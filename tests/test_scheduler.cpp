/**
 * @file test_scheduler.cpp
 * @brief Phase 6 scheduler tests.
 *
 * All tests exercise SchedulingPolicy::decide() with synthetic DeviceContext
 * values — no real GPU hardware required.  Two integration tests build small
 * CPU-only graphs through ScheduledGraph::execute() and verify that outputs
 * match Graph::execute().
 *
 * Test scenarios
 * ==============
 * 1. CPU preferred for small tensor (ReLU [1024]): transfer overhead > savings.
 * 2. GPU preferred for large MatMul [512×512]: compute savings > transfer cost.
 * 3. Residency changes the decision: inputs already on GPU removes H2D cost.
 * 4. VRAM limit blocks GPU even for large MatMul.
 * 5. Unsupported backend: profile with cuda_supported=false always picks CPU.
 * 6. Unknown op fallback: unrecognised op name → CPU, operator_name_known=false.
 * 7. Cost model verification: returned ns values match hand calculation.
 * 8. Integration (CPU): ScheduledGraph on a ReLU graph matches Graph::execute().
 * 9. Integration (CPU): multi-node graph (ReLU → Add) matches Graph::execute().
 */

#include "forgert/scheduler/scheduler.h"
#include "forgert/forgert.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace forgert;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void ok(bool condition, const char* msg) {
    if (!condition) {
        std::cerr << "FAIL: " << msg << "\n";
        std::exit(1);
    }
    std::cout << "PASS: " << msg << "\n";
}

static bool approxEq(double a, double b, double tol = 1e-6) {
    return std::fabs(a - b) <= tol * (std::fabs(a) + std::fabs(b) + 1.0);
}

// Hand-computed expected costs for a known profile + sizes.
// ReLU profile: cpu=0.18 ns/elem, speedup=5×
// Sizes: 1024 elements, float32 → 4096 bytes input, 4096 bytes output
// Transfer BW: 6 bytes/ns
static double expectedCpuCost(size_t elems, double ns_per_elem) {
    return ns_per_elem * static_cast<double>(elems);
}
// CUDA_LAUNCH_OVERHEAD_NS must match the constant in scheduler.cpp.
// Measured on GTX 1050 sm_61: ~20 us (benchmarks/measure_launch_overhead.cu).
static constexpr double EXPECTED_LAUNCH_OVERHEAD_NS = 20000.0;

static double expectedGpuCost(size_t elems, double ns_per_elem, double speedup,
                               size_t input_bytes, size_t output_bytes,
                               double bw, bool inputs_on_gpu) {
    double h2d    = inputs_on_gpu ? 0.0 : static_cast<double>(input_bytes) / bw;
    double kernel = EXPECTED_LAUNCH_OVERHEAD_NS
                  + (ns_per_elem / speedup) * static_cast<double>(elems);
    double d2h    = static_cast<double>(output_bytes) / bw;
    return h2d + kernel + d2h;
}

// ---------------------------------------------------------------------------
// Test 1: CPU preferred for small ReLU (transfer overhead > compute savings)
// ---------------------------------------------------------------------------
static void test_cpu_preferred_small_relu() {
    // ReLU, 1024 elements, float32
    const size_t elems        = 1024;
    const size_t input_bytes  = elems * sizeof(float);
    const size_t output_bytes = elems * sizeof(float);

    DeviceContext ctx;
    ctx.cuda_allocated_bytes          = 0;
    ctx.cuda_vram_budget_bytes        = static_cast<size_t>(3.8 * 1024 * 1024 * 1024);
    ctx.transfer_bandwidth_bytes_per_ns = 6.0;

    auto profiles = SchedulingPolicy::defaultProfiles();
    auto decision = SchedulingPolicy::decide(
        "ReLU", elems, input_bytes, output_bytes,
        /*inputs_already_on_gpu=*/false, ctx, profiles);

    // Expected: CPU cheaper because launch overhead + transfer dwarfs compute saving.
    // cpu    = 0.40 * 1024 = 409.6 ns
    // launch = 20000 ns  (measured: GTX 1050 sm_61)
    // gpu    = 4096/6 + 20000 + (0.40/3.0)*1024 + 4096/6
    //        = 682.7 + 20000 + 136.5 + 682.7 = ~21502 ns  → CPU wins by ~52x
    ok(decision.backend == Backend::CPU,
       "Test 1: ReLU [1024] → CPU preferred (transfer overhead > savings)");
    ok(decision.operator_name_known,
       "Test 1: operator_name_known=true for known op");
    ok(!decision.reason.empty(),
       "Test 1: reason is not empty");
}

// ---------------------------------------------------------------------------
// Test 2: GPU preferred for large MatMul
// ---------------------------------------------------------------------------
static void test_gpu_preferred_large_matmul() {
    // MatMul [512×512], float32 output → 512*512 = 262144 elems
    const size_t out_elems    = 512 * 512;
    // Typical inputs: [512×512] + [512×512] → 2 * 262144 * 4 bytes
    const size_t input_bytes  = 2 * out_elems * sizeof(float);
    const size_t output_bytes = out_elems * sizeof(float);

    DeviceContext ctx;
    ctx.cuda_allocated_bytes            = 0;
    ctx.cuda_vram_budget_bytes          = static_cast<size_t>(3.8 * 1024 * 1024 * 1024);
    ctx.transfer_bandwidth_bytes_per_ns = 6.0;

    auto profiles = SchedulingPolicy::defaultProfiles();
    auto decision = SchedulingPolicy::decide(
        "MatMul", out_elems, input_bytes, output_bytes,
        /*inputs_already_on_gpu=*/false, ctx, profiles);

    // cpu = 850 * 262144 = 222,822,400 ns (~223 ms)
    // gpu = (2*1MB)/6 + (850/120)*262144 + (1MB)/6
    //     = ~349 ms transfer + ~3.9 ms kernel = but let's just check backend
    // Actually: input_bytes = 2*262144*4 = 2,097,152
    //           h2d = 2097152 / 6 = 349,525 ns
    //           kernel = (850/120)*262144 = 1,857,813 ns
    //           d2h = 1048576 / 6 = 174,762 ns
    //           gpu_total = ~2,882,100 ns
    //           cpu_total = 850*262144 = 222,822,400 ns → GPU wins
    ok(decision.backend == Backend::CUDA,
       "Test 2: MatMul [512×512] → GPU preferred (compute savings > transfer)");
    ok(decision.estimated_cost_gpu_ns < decision.estimated_cost_cpu_ns,
       "Test 2: reported gpu cost < cpu cost");
    ok(!decision.reason.empty(),
       "Test 2: reason is not empty");
}

// ---------------------------------------------------------------------------
// Test 3: Residency changes the decision (inputs already on GPU)
// ---------------------------------------------------------------------------
static void test_residency_changes_decision() {
    // Use a medium MatMul where, without residency, GPU might not win,
    // but with residency (no H2D) it does.  Use small MatMul [64×64]:
    // out_elems = 4096, input_bytes = 2*4096*4 = 32768, output_bytes = 16384
    // Without residency:
    //   h2d    = 32768/6 = 5461 ns
    //   kernel = (850/120)*4096 = 29,013 ns
    //   d2h    = 16384/6 = 2730 ns
    //   gpu    = 37,204 ns
    //   cpu    = 850*4096 = 3,481,600 ns  → GPU wins
    // Let's verify with residency it ALSO wins (H2D skipped, even cheaper):
    const size_t out_elems    = 4096;
    const size_t input_bytes  = 2 * out_elems * sizeof(float);
    const size_t output_bytes = out_elems * sizeof(float);

    DeviceContext ctx;
    ctx.cuda_allocated_bytes            = 0;
    ctx.cuda_vram_budget_bytes          = static_cast<size_t>(3.8 * 1024 * 1024 * 1024);
    ctx.transfer_bandwidth_bytes_per_ns = 6.0;

    auto profiles = SchedulingPolicy::defaultProfiles();

    auto decision_no_res = SchedulingPolicy::decide(
        "MatMul", out_elems, input_bytes, output_bytes, false, ctx, profiles);
    auto decision_with_res = SchedulingPolicy::decide(
        "MatMul", out_elems, input_bytes, output_bytes, true,  ctx, profiles);

    // Both should pick GPU here, but with_res should have lower gpu cost.
    ok(decision_with_res.estimated_cost_gpu_ns <= decision_no_res.estimated_cost_gpu_ns,
       "Test 3: GPU cost with residency <= GPU cost without residency");

    // Now verify residency CHANGES the decision on a borderline case.
    //
    // Design: pick a bandwidth where H2D alone tips the GPU cost over the CPU cost,
    // but skipping H2D (residency) keeps the GPU total under the CPU cost.
    //
    // MatMul, N=10000 float32 elems:
    //   input_bytes  = 80000  (two [50×100] inputs)
    //   output_bytes = 40000
    //   cpu          = 850 * 10000 = 8,500,000 ns
    //
    // With bw=0.01 bytes/ns:
    //   h2d    = 80000 / 0.01 = 8,000,000 ns
    //   kernel = (850/120) * 10000 = 70,833 ns
    //   d2h    = 40000 / 0.01 = 4,000,000 ns
    //   gpu (no res) = 8,000,000 + 70,833 + 4,000,000 = 12,070,833 ns > cpu → CPU
    //   gpu (res)   =            70,833 + 4,000,000 =  4,070,833 ns < cpu → GPU
    const size_t bline_elems    = 10000;
    const size_t bline_in_bytes  = 80000;
    const size_t bline_out_bytes = 40000;

    DeviceContext bline_ctx;
    bline_ctx.cuda_allocated_bytes            = 0;
    bline_ctx.cuda_vram_budget_bytes          = static_cast<size_t>(3.8 * 1024 * 1024 * 1024);
    bline_ctx.transfer_bandwidth_bytes_per_ns = 0.01;  // very slow PCIe simulation

    auto dec_no_res = SchedulingPolicy::decide(
        "MatMul", bline_elems, bline_in_bytes, bline_out_bytes, false, bline_ctx, profiles);
    auto dec_with_res = SchedulingPolicy::decide(
        "MatMul", bline_elems, bline_in_bytes, bline_out_bytes, true, bline_ctx, profiles);

    ok(dec_no_res.backend == Backend::CPU,
       "Test 3: slow-BW + no residency → CPU preferred (H2D dominates)");
    ok(dec_with_res.backend == Backend::CUDA,
       "Test 3: slow-BW + residency → GPU preferred (H2D skipped, kernel+D2H < CPU)");
    ok(dec_with_res.estimated_cost_gpu_ns < dec_no_res.estimated_cost_gpu_ns,
       "Test 3: residency reduces GPU cost (H2D skipped)");
}

// ---------------------------------------------------------------------------
// Test 4: VRAM limit blocks GPU
// ---------------------------------------------------------------------------
static void test_vram_limit_blocks_gpu() {
    // Large MatMul that would normally go to GPU.
    const size_t out_elems    = 512 * 512;
    const size_t input_bytes  = 2 * out_elems * sizeof(float);
    const size_t output_bytes = out_elems * sizeof(float);

    // Set VRAM budget so small that even the output can't fit.
    DeviceContext ctx;
    ctx.cuda_allocated_bytes            = 0;
    ctx.cuda_vram_budget_bytes          = 1024;  // 1 KB — absurdly tiny
    ctx.transfer_bandwidth_bytes_per_ns = 6.0;

    auto profiles = SchedulingPolicy::defaultProfiles();
    auto decision = SchedulingPolicy::decide(
        "MatMul", out_elems, input_bytes, output_bytes, false, ctx, profiles);

    ok(decision.backend == Backend::CPU,
       "Test 4: VRAM limit → CPU fallback even for large MatMul");
    ok(decision.reason.find("vram-limit") != std::string::npos,
       "Test 4: reason mentions 'vram-limit'");
}

// ---------------------------------------------------------------------------
// Test 5: Operator with cuda_supported=false always picks CPU
// ---------------------------------------------------------------------------
static void test_unsupported_op_always_cpu() {
    // Inject a custom profile with cuda_supported=false.
    std::unordered_map<std::string, OperatorProfile> profiles = {
        {"CustomOp", {"CustomOp", 10.0, 20.0, /*cuda_supported=*/false}},
    };

    DeviceContext ctx;
    ctx.cuda_allocated_bytes            = 0;
    ctx.cuda_vram_budget_bytes          = static_cast<size_t>(3.8 * 1024 * 1024 * 1024);
    ctx.transfer_bandwidth_bytes_per_ns = 6.0;

    // Even with large tensors where GPU would otherwise win:
    const size_t out_elems    = 1024 * 1024;
    const size_t input_bytes  = out_elems * sizeof(float);
    const size_t output_bytes = out_elems * sizeof(float);

    auto decision = SchedulingPolicy::decide(
        "CustomOp", out_elems, input_bytes, output_bytes, false, ctx, profiles);

    ok(decision.backend == Backend::CPU,
       "Test 5: cuda_supported=false → always CPU");
    ok(decision.reason.find("no-cuda-impl") != std::string::npos,
       "Test 5: reason mentions 'no-cuda-impl'");
}

// ---------------------------------------------------------------------------
// Test 6: Unknown operator fallback
// ---------------------------------------------------------------------------
static void test_unknown_op_fallback() {
    auto profiles = SchedulingPolicy::defaultProfiles();

    DeviceContext ctx;
    ctx.cuda_allocated_bytes            = 0;
    ctx.cuda_vram_budget_bytes          = static_cast<size_t>(3.8 * 1024 * 1024 * 1024);
    ctx.transfer_bandwidth_bytes_per_ns = 6.0;

    auto decision = SchedulingPolicy::decide(
        "UnknownOpXYZ", 1024, 4096, 4096, false, ctx, profiles);

    ok(decision.backend == Backend::CPU,
       "Test 6: unknown op → CPU fallback");
    ok(!decision.operator_name_known,
       "Test 6: operator_name_known=false for unknown op");
    ok(decision.reason.find("unknown-op") != std::string::npos,
       "Test 6: reason mentions 'unknown-op'");
}

// ---------------------------------------------------------------------------
// Test 7: Cost model verification — numbers match hand calculation
// ---------------------------------------------------------------------------
static void test_cost_model_numbers() {
    // ReLU, 1024 float32 elements.
    const size_t elems        = 1024;
    const size_t input_bytes  = elems * sizeof(float);   // 4096
    const size_t output_bytes = elems * sizeof(float);   // 4096
    const double bw           = 6.0;
    const double ns_per_elem  = 0.40;
    const double speedup      = 3.0;

    double exp_cpu = expectedCpuCost(elems, ns_per_elem);
    double exp_gpu = expectedGpuCost(elems, ns_per_elem, speedup,
                                     input_bytes, output_bytes, bw, false);

    DeviceContext ctx;
    ctx.cuda_allocated_bytes            = 0;
    ctx.cuda_vram_budget_bytes          = static_cast<size_t>(3.8 * 1024 * 1024 * 1024);
    ctx.transfer_bandwidth_bytes_per_ns = bw;

    auto profiles = SchedulingPolicy::defaultProfiles();
    auto decision = SchedulingPolicy::decide(
        "ReLU", elems, input_bytes, output_bytes, false, ctx, profiles);

    ok(approxEq(decision.estimated_cost_cpu_ns, exp_cpu, 1e-4),
       "Test 7: CPU cost matches hand calculation");
    ok(approxEq(decision.estimated_cost_gpu_ns, exp_gpu, 1e-4),
       "Test 7: GPU cost matches hand calculation (h2d+kernel+d2h)");

    // Verify with residency: H2D skipped.
    double exp_gpu_res = expectedGpuCost(elems, ns_per_elem, speedup,
                                         input_bytes, output_bytes, bw, true);
    auto decision_res = SchedulingPolicy::decide(
        "ReLU", elems, input_bytes, output_bytes, true, ctx, profiles);
    ok(approxEq(decision_res.estimated_cost_gpu_ns, exp_gpu_res, 1e-4),
       "Test 7: GPU cost with residency matches hand calculation (d2h+kernel only)");
}

// ---------------------------------------------------------------------------
// Test 8: ScheduledGraph integration — single ReLU node, CPU-only
// ---------------------------------------------------------------------------
static void test_scheduled_graph_single_relu() {
    // Build a graph: input [4, 8] → ReLU → output
    Graph g;
    TensorShape shape({4, 8});
    size_t in_id = g.addInput(shape, DataType::Float32, "x");
    auto relu_op = std::make_unique<ReLUOp>();
    auto out_ids = g.addNode(std::move(relu_op), {in_id}, "relu");
    g.markOutput(out_ids[0]);
    g.validate();

    // Create input with mixed positive/negative values.
    Tensor input(shape, DataType::Float32, Device::CPU);
    float* data = static_cast<float*>(input.data());
    for (int i = 0; i < 32; ++i) {
        data[i] = static_cast<float>(i - 16);  // [-16 .. 15]
    }

    // Reference execution via Graph::execute().
    auto ref_outputs = g.execute({&input}, Backend::CPU);
    const float* ref = static_cast<const float*>(ref_outputs[0]->data());

    // ScheduledGraph execution.
    ScheduledGraph sg(g);
    // Force VRAM budget to 0 to ensure all decisions → CPU (no GPU available in CPU build).
    sg.setVramBudgetBytes(0);
    auto sched_outputs = sg.execute({&input});

    ok(sched_outputs.size() == 1,
       "Test 8: ScheduledGraph returns 1 output");
    ok(sched_outputs[0]->shape().numElements() == 32,
       "Test 8: output has correct element count");

    const float* got = static_cast<const float*>(sched_outputs[0]->data());
    bool values_match = true;
    for (int i = 0; i < 32; ++i) {
        if (std::fabs(got[i] - ref[i]) > 1e-5f) {
            values_match = false;
            std::cerr << "  element " << i << ": got=" << got[i]
                      << " ref=" << ref[i] << "\n";
            break;
        }
    }
    ok(values_match, "Test 8: ScheduledGraph output matches Graph::execute()");

    // Print schedule for inspection.
    sg.printSchedule();
}

// ---------------------------------------------------------------------------
// Test 9: ScheduledGraph integration — multi-node (ReLU → Add), CPU-only
// ---------------------------------------------------------------------------
static void test_scheduled_graph_multi_node() {
    // Graph: x → ReLU → y; (y + x) → output
    Graph g;
    TensorShape shape({4, 8});
    size_t x_id    = g.addInput(shape, DataType::Float32, "x");
    auto relu_op   = std::make_unique<ReLUOp>();
    auto relu_outs = g.addNode(std::move(relu_op), {x_id}, "relu");
    size_t y_id    = relu_outs[0];

    auto add_op   = std::make_unique<AddOp>();
    auto add_outs = g.addNode(std::move(add_op), {y_id, x_id}, "add");
    g.markOutput(add_outs[0]);
    g.validate();

    Tensor input(shape, DataType::Float32, Device::CPU);
    float* data = static_cast<float*>(input.data());
    for (int i = 0; i < 32; ++i) {
        data[i] = static_cast<float>(i - 16);
    }

    auto ref_outputs = g.execute({&input}, Backend::CPU);
    const float* ref = static_cast<const float*>(ref_outputs[0]->data());

    ScheduledGraph sg(g);
    sg.setVramBudgetBytes(0);  // force CPU
    auto sched_outputs = sg.execute({&input});

    ok(sched_outputs.size() == 1,
       "Test 9: multi-node ScheduledGraph returns 1 output");

    const float* got = static_cast<const float*>(sched_outputs[0]->data());
    bool values_match = true;
    for (int i = 0; i < 32; ++i) {
        if (std::fabs(got[i] - ref[i]) > 1e-5f) {
            values_match = false;
            std::cerr << "  element " << i << ": got=" << got[i]
                      << " ref=" << ref[i] << "\n";
            break;
        }
    }
    ok(values_match,
       "Test 9: multi-node ScheduledGraph output matches Graph::execute()");

    sg.printSchedule();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    std::cout << "=== Phase 6 Scheduler Tests ===\n\n";

    test_cpu_preferred_small_relu();
    test_gpu_preferred_large_matmul();
    test_residency_changes_decision();
    test_vram_limit_blocks_gpu();
    test_unsupported_op_always_cpu();
    test_unknown_op_fallback();
    test_cost_model_numbers();
    test_scheduled_graph_single_relu();
    test_scheduled_graph_multi_node();

    std::cout << "\n=== All Phase 6 tests passed ===\n";
    return 0;
}

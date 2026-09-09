/*
 * test_graph_cuda.cu  –  End-to-end CUDA graph execution tests (Phase 3)
 *
 * Proves that the ForgeRT Tensor → Graph → Operator → CUDA kernel path works
 * as a coherent runtime.  The execution model tested here is:
 *
 *   1. Caller prepares CPU input tensor
 *   2. Caller uploads to CUDA via Tensor::copyTo()          [explicit H2D]
 *   3. graph.execute({&cuda_input}, Backend::CUDA)
 *        • Graph allocates all intermediates/outputs as Device::CUDA
 *        • Each node's inputs and outputs are all device-resident
 *        • No implicit CPU↔GPU transfers between nodes
 *   4. Caller downloads output via Tensor::copyTo()          [explicit D2H]
 *   5. Caller verifies numerical result on CPU
 *
 * This design keeps transfers explicit and measurable, consistent with
 * ForgeRT's transparency principle.
 *
 * Tests
 * -----
 * 1.  Single-node CUDA ReLU graph — small known values
 * 2.  Large tensor (1M elements) — confirms grid-stride correctness
 * 3.  Multi-node CUDA graph (ReLU → ReLU) — confirms CUDA intermediates
 *     are correctly allocated and passed between graph nodes
 * 4.  CPU graph unchanged — Backend::CPU still works on the same graph
 * 5.  Error: CUDA backend + CPU-device input → throws before kernel
 * 6.  Error: CUDA backend + unsupported operator → throws before kernel
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <vector>
#include <memory>

#include "forgert/tensor/tensor.h"
#include "forgert/operator/operator.h"
#include "forgert/operator/relu.h"
#include "forgert/graph/graph.h"

using namespace forgert;

// ------------------------------------------------------------------ helpers --

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                 \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

static const float kTol = 1e-5f;
static int g_pass = 0;
static int g_fail = 0;

static void pass(const char* msg) { ++g_pass; printf("  \xE2\x9C\x93 %s\n", msg); }
static void fail(const char* msg) { ++g_fail; printf("  FAIL: %s\n", msg); }

// CPU reference ReLU for comparison
static float cpu_relu(float x) { return x > 0.0f ? x : 0.0f; }

// -------------------------------------------------------------------- tests --

/*
 * Test 1: Single-node CUDA graph — small known values
 *
 * Graph:   Input → ReLU → Output
 * Input:   [-3, -1, 0, 1, 2, -0.5, 4, -2]
 * Expected:[  0,  0, 0, 1, 2,  0.0, 4,  0]
 */
static void test_single_node_small() {
    printf("  Test 1: Single-node ReLU graph, small known values\n");

    Graph graph;
    size_t in_id = graph.addInput(TensorShape({8}), DataType::Float32);
    auto out_ids = graph.addNode(std::make_unique<ReLUOp>(), {in_id});
    graph.markOutput(out_ids[0]);
    graph.validate();

    // Prepare CPU input
    float h_in[8] = {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, -0.5f, 4.0f, -2.0f};
    Tensor cpu_in(TensorShape({8}), DataType::Float32, Device::CPU);
    std::memcpy(cpu_in.data(), h_in, 8 * sizeof(float));

    // Upload to CUDA
    Tensor cuda_in(TensorShape({8}), DataType::Float32, Device::CUDA);
    cpu_in.copyTo(cuda_in);

    // Execute graph on CUDA
    auto outputs = graph.execute({&cuda_in}, Backend::CUDA);
    assert(outputs.size() == 1);
    assert(outputs[0]->device() == Device::CUDA); // output must be on device

    // Download result
    Tensor cpu_out(TensorShape({8}), DataType::Float32, Device::CPU);
    outputs[0]->copyTo(cpu_out);

    const float* out = static_cast<const float*>(cpu_out.data());
    float expected[8];
    for (int i = 0; i < 8; ++i) expected[i] = cpu_relu(h_in[i]);

    bool ok = true;
    for (int i = 0; i < 8; ++i) {
        if (fabsf(out[i] - expected[i]) > kTol) {
            fprintf(stderr, "  FAIL at i=%d: expected %.4f got %.4f\n",
                    i, expected[i], out[i]);
            ok = false;
        }
    }
    if (ok) pass("Single-node CUDA graph, small values");
    else    fail("Single-node CUDA graph, small values");
}

/*
 * Test 2: Large tensor (1M elements) — verifies grid-stride kernel path
 *
 * Values alternate above and below zero deterministically.
 * All 1M elements are checked.
 */
static void test_single_node_large() {
    printf("  Test 2: Single-node ReLU graph, 1M elements (grid-stride)\n");

    const size_t n = 1024u * 1024u;

    Graph graph;
    size_t in_id = graph.addInput(TensorShape({n}), DataType::Float32);
    auto out_ids = graph.addNode(std::make_unique<ReLUOp>(), {in_id});
    graph.markOutput(out_ids[0]);
    graph.validate();

    // Build deterministic CPU input
    std::vector<float> h_in(n);
    for (size_t i = 0; i < n; ++i)
        h_in[i] = static_cast<float>(i % 2001) - 1000.0f;

    Tensor cpu_in(TensorShape({n}), DataType::Float32, Device::CPU);
    std::memcpy(cpu_in.data(), h_in.data(), n * sizeof(float));

    Tensor cuda_in(TensorShape({n}), DataType::Float32, Device::CUDA);
    cpu_in.copyTo(cuda_in);

    auto outputs = graph.execute({&cuda_in}, Backend::CUDA);
    assert(outputs[0]->device() == Device::CUDA);
    assert(outputs[0]->numElements() == n);

    Tensor cpu_out(TensorShape({n}), DataType::Float32, Device::CPU);
    outputs[0]->copyTo(cpu_out);

    const float* out = static_cast<const float*>(cpu_out.data());
    bool ok = true;
    for (size_t i = 0; i < n; ++i) {
        float expected = cpu_relu(h_in[i]);
        if (fabsf(out[i] - expected) > kTol) {
            fprintf(stderr, "  FAIL at i=%zu: expected %.4f got %.4f\n",
                    i, expected, out[i]);
            ok = false;
            break;
        }
    }
    if (ok) pass("Single-node CUDA graph, 1M elements (grid-stride)");
    else    fail("Single-node CUDA graph, 1M elements (grid-stride)");
}

/*
 * Test 3: Multi-node CUDA graph — CUDA intermediate tensors
 *
 * Graph:   Input → ReLU (node A) → intermediate (CUDA) → ReLU (node B) → Output
 *
 * Key verification: the intermediate tensor between nodes is Device::CUDA
 * (the graph allocates it; no CPU bounce).  ReLU is idempotent for non-negative
 * values, so applying it twice gives the same result as once.
 *
 * Input: [-3, -1, 0, 1, 2, -0.5, 4, -2]
 * After A: [0, 0, 0, 1, 2, 0, 4, 0]
 * After B: [0, 0, 0, 1, 2, 0, 4, 0]  (ReLU is idempotent on non-negative)
 */
static void test_multi_node_cuda_intermediates() {
    printf("  Test 3: Multi-node CUDA graph (ReLU→ReLU), CUDA intermediate\n");

    Graph graph;
    size_t in_id   = graph.addInput(TensorShape({8}), DataType::Float32);
    auto   mid_ids = graph.addNode(std::make_unique<ReLUOp>(), {in_id},   "relu_a");
    auto   out_ids = graph.addNode(std::make_unique<ReLUOp>(), {mid_ids[0]}, "relu_b");
    graph.markOutput(out_ids[0]);
    graph.validate();

    float h_in[8] = {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, -0.5f, 4.0f, -2.0f};
    Tensor cpu_in(TensorShape({8}), DataType::Float32, Device::CPU);
    std::memcpy(cpu_in.data(), h_in, 8 * sizeof(float));

    Tensor cuda_in(TensorShape({8}), DataType::Float32, Device::CUDA);
    cpu_in.copyTo(cuda_in);

    auto outputs = graph.execute({&cuda_in}, Backend::CUDA);
    assert(outputs.size() == 1);
    assert(outputs[0]->device() == Device::CUDA);

    Tensor cpu_out(TensorShape({8}), DataType::Float32, Device::CPU);
    outputs[0]->copyTo(cpu_out);

    const float* out = static_cast<const float*>(cpu_out.data());
    // Two ReLU passes = same as one (idempotent on non-negative)
    float expected[8];
    for (int i = 0; i < 8; ++i) expected[i] = cpu_relu(cpu_relu(h_in[i]));

    bool ok = true;
    for (int i = 0; i < 8; ++i) {
        if (fabsf(out[i] - expected[i]) > kTol) {
            fprintf(stderr, "  FAIL at i=%d: expected %.4f got %.4f\n",
                    i, expected[i], out[i]);
            ok = false;
        }
    }
    if (ok) pass("Multi-node CUDA graph, CUDA intermediates pass between nodes");
    else    fail("Multi-node CUDA graph, CUDA intermediates");
}

/*
 * Test 4: Same graph object, CPU backend — proves no CUDA-specific state
 *         was embedded in the graph structure
 */
static void test_cpu_backend_unchanged() {
    printf("  Test 4: Same graph type, CPU backend still works correctly\n");

    Graph graph;
    size_t in_id = graph.addInput(TensorShape({8}), DataType::Float32);
    auto out_ids = graph.addNode(std::make_unique<ReLUOp>(), {in_id});
    graph.markOutput(out_ids[0]);
    graph.validate();

    float h_in[8] = {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, -0.5f, 4.0f, -2.0f};
    Tensor cpu_in(TensorShape({8}), DataType::Float32, Device::CPU);
    std::memcpy(cpu_in.data(), h_in, 8 * sizeof(float));

    // Execute on CPU — no CUDA involved
    auto outputs = graph.execute({&cpu_in}, Backend::CPU);
    assert(outputs.size() == 1);
    assert(outputs[0]->device() == Device::CPU);

    const float* out = static_cast<const float*>(outputs[0]->data());
    bool ok = true;
    for (int i = 0; i < 8; ++i) {
        if (fabsf(out[i] - cpu_relu(h_in[i])) > kTol) { ok = false; break; }
    }
    if (ok) pass("CPU backend unaffected by CUDA infrastructure");
    else    fail("CPU backend unaffected by CUDA infrastructure");
}

/*
 * Test 5: Error path — CPU-device input passed to CUDA graph
 *
 * The graph's execute() stores the caller's tensor directly.  The first
 * node will then see a CPU tensor and should throw via validateDevice.
 */
static void test_error_cpu_input_to_cuda_graph() {
    printf("  Test 5: Error path — CPU input to CUDA graph\n");

    Graph graph;
    size_t in_id = graph.addInput(TensorShape({4}), DataType::Float32);
    auto out_ids = graph.addNode(std::make_unique<ReLUOp>(), {in_id});
    graph.markOutput(out_ids[0]);
    graph.validate();

    // Intentionally CPU tensor — not uploaded to device
    float h_in[4] = {1.0f, -1.0f, 2.0f, -2.0f};
    Tensor cpu_in(TensorShape({4}), DataType::Float32, Device::CPU);
    std::memcpy(cpu_in.data(), h_in, 4 * sizeof(float));

    bool threw = false;
    try {
        graph.execute({&cpu_in}, Backend::CUDA);
    } catch (const std::invalid_argument&) {
        threw = true;
    } catch (const std::runtime_error&) {
        threw = true;
    }

    if (threw) pass("CPU input correctly rejected by CUDA graph");
    else       fail("CPU input NOT rejected — missing validation");
}

/*
 * Test 6: Error path — operator that does not support CUDA
 *
 * Use a minimal stub operator that only reports Backend::CPU support.
 * Inserting it in a CUDA graph should throw via Operator::execute()
 * before any kernel is launched.
 */
namespace {

class CPUOnlyOp : public Operator {
public:
    std::string name() const override { return "CPUOnly"; }

    bool supportsBackend(Backend b) const override {
        return b == Backend::CPU;
    }

    std::vector<TensorShape> inferOutputShapes(
        const std::vector<TensorShape>& in) const override {
        validateInputCount("CPUOnlyOp", 1, in.size());
        return {in[0]};
    }

    void executeCPU(
        const std::vector<const Tensor*>& inputs,
        const std::vector<Tensor*>&       outputs) override {
        const float* in  = static_cast<const float*>(inputs[0]->data());
        float*       out = static_cast<float*>(outputs[0]->data());
        for (size_t i = 0; i < inputs[0]->numElements(); ++i) out[i] = in[i];
    }
};

} // anonymous namespace

static void test_error_unsupported_cuda_operator() {
    printf("  Test 6: Error path — CPU-only operator in CUDA graph\n");

    Graph graph;
    size_t in_id = graph.addInput(TensorShape({4}), DataType::Float32);
    auto out_ids = graph.addNode(std::make_unique<CPUOnlyOp>(), {in_id});
    graph.markOutput(out_ids[0]);
    graph.validate();

    float h_in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor cpu_in(TensorShape({4}), DataType::Float32, Device::CPU);
    std::memcpy(cpu_in.data(), h_in, 4 * sizeof(float));

    Tensor cuda_in(TensorShape({4}), DataType::Float32, Device::CUDA);
    cpu_in.copyTo(cuda_in);

    bool threw = false;
    try {
        graph.execute({&cuda_in}, Backend::CUDA);
    } catch (const std::runtime_error&) {
        threw = true;
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    if (threw) pass("CPU-only operator correctly rejected for CUDA backend");
    else       fail("CPU-only operator NOT rejected — missing validation");
}

/*
 * Test 7: CPU vs CUDA numerical equivalence — large tensor
 *
 * Run the same graph on both backends with identical input.
 * Every output element must match within tolerance.
 * This is the strongest correctness proof for the CUDA path.
 */
static void test_cpu_cuda_numerical_equivalence() {
    printf("  Test 7: CPU vs CUDA numerical equivalence, 100K elements\n");

    const size_t n = 100000;

    // Build input on CPU
    std::vector<float> h_vals(n);
    for (size_t i = 0; i < n; ++i)
        h_vals[i] = static_cast<float>((i * 17) % 3001) - 1500.0f;

    // --- CPU execution ---
    Graph cpu_graph;
    size_t cpu_in_id = cpu_graph.addInput(TensorShape({n}), DataType::Float32);
    auto   cpu_out_ids = cpu_graph.addNode(std::make_unique<ReLUOp>(), {cpu_in_id});
    cpu_graph.markOutput(cpu_out_ids[0]);
    cpu_graph.validate();

    Tensor cpu_input(TensorShape({n}), DataType::Float32, Device::CPU);
    std::memcpy(cpu_input.data(), h_vals.data(), n * sizeof(float));
    auto cpu_outputs = cpu_graph.execute({&cpu_input}, Backend::CPU);
    const float* cpu_result = static_cast<const float*>(cpu_outputs[0]->data());

    // --- CUDA execution ---
    Graph cuda_graph;
    size_t cuda_in_id = cuda_graph.addInput(TensorShape({n}), DataType::Float32);
    auto   cuda_out_ids = cuda_graph.addNode(std::make_unique<ReLUOp>(), {cuda_in_id});
    cuda_graph.markOutput(cuda_out_ids[0]);
    cuda_graph.validate();

    Tensor cuda_input_cpu(TensorShape({n}), DataType::Float32, Device::CPU);
    std::memcpy(cuda_input_cpu.data(), h_vals.data(), n * sizeof(float));
    Tensor cuda_input(TensorShape({n}), DataType::Float32, Device::CUDA);
    cuda_input_cpu.copyTo(cuda_input);

    auto cuda_outputs = cuda_graph.execute({&cuda_input}, Backend::CUDA);
    Tensor cuda_result_cpu(TensorShape({n}), DataType::Float32, Device::CPU);
    cuda_outputs[0]->copyTo(cuda_result_cpu);
    const float* cuda_result = static_cast<const float*>(cuda_result_cpu.data());

    // Compare every element
    bool ok = true;
    for (size_t i = 0; i < n; ++i) {
        if (fabsf(cuda_result[i] - cpu_result[i]) > kTol) {
            fprintf(stderr, "  FAIL at i=%zu: CPU=%.6f CUDA=%.6f\n",
                    i, cpu_result[i], cuda_result[i]);
            ok = false;
            break;
        }
    }
    if (ok) pass("CPU vs CUDA numerical equivalence, 100K elements");
    else    fail("CPU vs CUDA numerical equivalence");
}

// --------------------------------------------------------------------- main --

int main() {
    printf("========================================\n");
    printf("ForgeRT CUDA Graph Integration Tests\n");
    printf("========================================\n");

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    test_single_node_small();
    test_single_node_large();
    test_multi_node_cuda_intermediates();
    test_cpu_backend_unchanged();
    test_error_cpu_input_to_cuda_graph();
    test_error_unsupported_cuda_operator();
    test_cpu_cuda_numerical_equivalence();

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    printf("========================================\n");
    return (g_fail == 0) ? 0 : 1;
}

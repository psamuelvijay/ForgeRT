/**
 * @file test_profile_calibration.cpp
 * @brief Focused tests for scheduler profile validity, unit consistency,
 *        fallback behaviour, and deterministic placement decisions.
 *
 * These are CORRECTNESS tests (CPU-only, no GPU, no benchmarking). They operate
 * on in-code OperatorProfile structs and SchedulingPolicy::decide(); they do NOT
 * parse JSON or run the profiler. The measurement -> calibration pipeline lives
 * in benchmarks/benchmark_profile.cu + tools/calibrate_profiles.py and is kept
 * out of the CTest correctness gate on purpose.
 *
 * What is checked:
 *   1. defaultProfiles() are structurally valid (finite, positive, cuda flags).
 *   2. Unit consistency: for a measured-style profile, decide()'s component
 *      costs (h2d, kernel+launch, d2h, cpu) are individually sane and the GPU
 *      total is their sum -- no double-counting, consistent ns units.
 *   3. Fallback: an unknown op name degrades to CPU and is flagged as not known
 *      (estimates/measurements are never silently conflated).
 *   4. A profile with cuda_supported=false never yields a CUDA decision.
 *   5. Determinism: identical inputs yield identical decisions and costs.
 */

#include "forgert/scheduler/scheduler.h"

#include <cmath>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace forgert;

static int g_fail = 0;
static void ok(bool cond, const std::string& msg) {
    std::cout << (cond ? "PASS: " : "FAIL: ") << msg << "\n";
    if (!cond) ++g_fail;
}
static bool finitePos(double x) { return std::isfinite(x) && x > 0.0; }

// Build a DeviceContext with explicit, synthetic values (no hardware query).
static DeviceContext ctxWith(double bw, size_t budget = (size_t)4 << 30,
                             size_t allocated = 0) {
    DeviceContext c;
    c.transfer_bandwidth_bytes_per_ns = bw;
    c.cuda_vram_budget_bytes = budget;
    c.cuda_allocated_bytes = allocated;
    return c;
}

// ---------------------------------------------------------------------------
// 1. defaultProfiles() structural validity
// ---------------------------------------------------------------------------
static void test_default_profiles_valid() {
    auto profiles = SchedulingPolicy::defaultProfiles();
    ok(!profiles.empty(), "defaultProfiles() is non-empty");
    for (const auto& kv : profiles) {
        const OperatorProfile& p = kv.second;
        ok(p.op_name == kv.first,
           "profile key matches op_name for " + kv.first);
        ok(finitePos(p.cpu_ns_per_elem),
           kv.first + ": cpu_ns_per_elem finite and > 0");
        // gpu_speedup must be finite and > 0 (dimensionless ratio).
        ok(finitePos(p.gpu_speedup),
           kv.first + ": gpu_speedup finite and > 0");
        // Every default op is expected to have a CUDA implementation.
        ok(p.cuda_supported, kv.first + ": cuda_supported is true");
    }
}

// ---------------------------------------------------------------------------
// 2. Unit consistency: GPU total == h2d + (launch+kernel) + d2h, all in ns.
//    We cannot read the private component values, so we reconstruct them with
//    the SAME formula/constants the scheduler documents and require the public
//    estimated_cost_gpu_ns to equal that reconstruction. This guards against a
//    future change silently altering units or double-counting a term.
// ---------------------------------------------------------------------------
static constexpr double LAUNCH_NS = 20000.0;  // must match scheduler.cpp

static void test_unit_consistency() {
    OperatorProfile mm{"MatMul", 850.0, 120.0, true};
    std::unordered_map<std::string, OperatorProfile> profiles{{"MatMul", mm}};

    const size_t elems = 512 * 512;
    const size_t in_bytes = (512 * 512 + 512 * 512) * sizeof(float);
    const size_t out_bytes = 512 * 512 * sizeof(float);
    const double bw = 6.0;

    DeviceContext ctx = ctxWith(bw);
    ScheduleDecision d = SchedulingPolicy::decide(
        "MatMul", elems, in_bytes, out_bytes,
        /*inputs_already_on_gpu=*/false, ctx, profiles);

    const double h2d = in_bytes / bw;
    const double kernel = LAUNCH_NS + (mm.cpu_ns_per_elem / mm.gpu_speedup) * elems;
    const double d2h = out_bytes / bw;
    const double expected_gpu = h2d + kernel + d2h;
    const double expected_cpu = mm.cpu_ns_per_elem * elems;

    auto approx = [](double a, double b) {
        return std::fabs(a - b) <= 1e-6 * (std::fabs(a) + std::fabs(b) + 1.0);
    };
    ok(approx(d.estimated_cost_gpu_ns, expected_gpu),
       "GPU cost == h2d + (launch+kernel) + d2h (ns, no double-count)");
    ok(approx(d.estimated_cost_cpu_ns, expected_cpu),
       "CPU cost == cpu_ns_per_elem * elems (ns)");
    // Residency removes exactly the H2D term, nothing else.
    ScheduleDecision dres = SchedulingPolicy::decide(
        "MatMul", elems, in_bytes, out_bytes, /*inputs_already_on_gpu=*/true,
        ctx, profiles);
    ok(approx(dres.estimated_cost_gpu_ns, expected_gpu - h2d),
       "residency removes exactly the H2D term");
}

// ---------------------------------------------------------------------------
// 3. Fallback: unknown op -> CPU, flagged not-known.
// ---------------------------------------------------------------------------
static void test_unknown_op_fallback() {
    auto profiles = SchedulingPolicy::defaultProfiles();
    DeviceContext ctx = ctxWith(6.0);
    ScheduleDecision d = SchedulingPolicy::decide(
        "NoSuchOp", 1 << 20, 4 << 20, 4 << 20, false, ctx, profiles);
    ok(d.backend == Backend::CPU, "unknown op falls back to CPU");
    ok(!d.operator_name_known, "unknown op flagged operator_name_known=false");
    ok(d.reason.find("unknown-op") != std::string::npos,
       "unknown op reason is explicit (not silent)");
}

// ---------------------------------------------------------------------------
// 4. cuda_supported=false never yields CUDA.
// ---------------------------------------------------------------------------
static void test_cuda_unsupported_never_gpu() {
    OperatorProfile cpuonly{"CpuOnly", 1000.0, 50.0, false};
    std::unordered_map<std::string, OperatorProfile> profiles{{"CpuOnly", cpuonly}};
    DeviceContext ctx = ctxWith(6.0);
    // Even a huge, compute-heavy op must stay on CPU if cuda_supported=false.
    ScheduleDecision d = SchedulingPolicy::decide(
        "CpuOnly", 10 << 20, 4 << 20, 4 << 20, false, ctx, profiles);
    ok(d.backend == Backend::CPU, "cuda_supported=false forces CPU");
    ok(d.reason.find("no-cuda-impl") != std::string::npos,
       "no-cuda-impl reason is explicit");
}

// ---------------------------------------------------------------------------
// 5. Determinism: same inputs -> identical backend and costs.
// ---------------------------------------------------------------------------
static void test_determinism() {
    auto profiles = SchedulingPolicy::defaultProfiles();
    DeviceContext ctx = ctxWith(6.0);
    ScheduleDecision a = SchedulingPolicy::decide(
        "MatMul", 512 * 512, 2u * 512 * 512 * 4, 512u * 512 * 4, false, ctx, profiles);
    ScheduleDecision b = SchedulingPolicy::decide(
        "MatMul", 512 * 512, 2u * 512 * 512 * 4, 512u * 512 * 4, false, ctx, profiles);
    ok(a.backend == b.backend, "deterministic backend decision");
    ok(a.estimated_cost_cpu_ns == b.estimated_cost_cpu_ns &&
       a.estimated_cost_gpu_ns == b.estimated_cost_gpu_ns,
       "deterministic cost estimates");
}

int main() {
    std::cout << "========================================\n";
    std::cout << "ForgeRT Profile / Calibration Validation Tests\n";
    std::cout << "========================================\n";
    test_default_profiles_valid();
    test_unit_consistency();
    test_unknown_op_fallback();
    test_cuda_unsupported_never_gpu();
    test_determinism();
    std::cout << "========================================\n";
    std::cout << (g_fail == 0 ? "All profile/calibration tests PASSED\n"
                              : std::to_string(g_fail) + " test(s) FAILED\n");
    std::cout << "========================================\n";
    return g_fail == 0 ? 0 : 1;
}
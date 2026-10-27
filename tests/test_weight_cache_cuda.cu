/*
 * test_weight_cache_cuda.cu -- persistent device-resident weights (needs a GPU).
 *
 * Graph under test:  h = x @ W1 ; y = h @ W2   (x: runtime input, W1/W2: weights)
 * 256x256 float matrices (256 KiB each); the cost model places both MatMuls on CUDA,
 * which every case asserts so the test cannot pass by silently running on the CPU.
 *
 * Independent evidence: besides the cache's own counters, every case checks
 * MemoryStats' CPU->CUDA transfer byte/count totals (recorded by Tensor::copyTo),
 * so "avoided upload" is measured at the transfer layer, not self-reported.
 *
 *   1  first run uploads the frozen weights                  (bytes + misses)
 *   2  later runs upload only the runtime input              (exact byte delta)
 *   3  outputs match a CPU reference on every run
 *   4  changed runtime input => changed, correct output (inputs are never cached)
 *   5  separate graphs / swapped models never share cached data
 *   6  unfreeze+modify+refreeze invalidates; escaped writes caught in verify mode
 *   7  release / destruction return the VRAM
 *   8  budget limit falls back to transient upload, still correct
 *
 * Exit: 0 pass, 1 fail, 77 skipped (no usable CUDA device).
 */
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "forgert/graph/graph.h"
#include "forgert/memory/memory_stats.h"
#include "forgert/operator/matmul.h"
#include "forgert/scheduler/scheduler.h"
#include "forgert/tensor/tensor.h"

using namespace forgert;

static int g_fail = 0;
static void ok(bool c, const std::string& m) {
    std::printf("  %s %s\n", c ? "PASS:" : "FAIL:", m.c_str());
    if (!c) ++g_fail;
}

static constexpr size_t D = 256;
static constexpr size_t kMatBytes = D * D * sizeof(float);

static void fillLcg(Tensor& t, uint32_t seed) {
    float* p = static_cast<float*>(t.data());
    uint32_t s = seed;
    for (size_t i = 0; i < t.numElements(); ++i) {
        s = s * 1664525u + 1013904223u;
        p[i] = (static_cast<float>(s >> 8) / 16777216.0f) - 0.5f;   // [-0.5, 0.5)
    }
}
static Tensor mat(uint32_t seed) {
    Tensor t(TensorShape({D, D}), DataType::Float32, Device::CPU);
    fillLcg(t, seed);
    return t;
}
static double maxDiff(const Tensor& a, const Tensor& b) {
    const float* p = static_cast<const float*>(a.data());
    const float* q = static_cast<const float*>(b.data());
    double m = 0;
    for (size_t i = 0; i < a.numElements(); ++i) {
        const double d = std::fabs(double(p[i]) - double(q[i]));
        if (!(d <= m)) m = d;           // NaN propagates as a failure
    }
    return m;
}
static constexpr double kTol = 2e-3;

struct Net {                // owns the Graph; ids for x, w1, w2
    Graph g;
    Net() {
        size_t x  = g.addInput(TensorShape({D, D}), DataType::Float32, "x");
        size_t w1 = g.addInput(TensorShape({D, D}), DataType::Float32, "w1");
        size_t w2 = g.addInput(TensorShape({D, D}), DataType::Float32, "w2");
        auto h = g.addNode(std::make_unique<MatMulOp>(), {x, w1}, "mm1");
        auto y = g.addNode(std::make_unique<MatMulOp>(), {h[0], w2}, "mm2");
        g.markOutput(y[0]);
        g.validate();
    }
    std::unique_ptr<Tensor> cpuRef(const Tensor& x, const Tensor& w1, const Tensor& w2) {
        auto r = g.execute({&x, &w1, &w2}, Backend::CPU);
        return std::move(r[0]);
    }
};

static size_t h2dBytes() { return MemoryStats::instance().getTransferBytes(Device::CPU, Device::CUDA); }
static size_t h2dCount() { return MemoryStats::instance().getTransferCount(Device::CPU, Device::CUDA); }
static size_t cudaLive() { return MemoryStats::instance().getAllocatedBytes(Device::CUDA); }

static bool bothOnCuda(const ScheduledGraph& sg) {
    const auto& s = sg.lastSchedule();
    if (s.size() != 2) return false;
    for (const auto& n : s) if (n.decision.backend != Backend::CUDA) return false;
    return true;
}

struct RunDelta { size_t bytes, count; std::unique_ptr<Tensor> out; };
static RunDelta run(ScheduledGraph& sg, const Tensor& x, const Tensor& w1, const Tensor& w2) {
    const size_t b0 = h2dBytes(), c0 = h2dCount();
    auto r = sg.execute({&x, &w1, &w2});
    return {h2dBytes() - b0, h2dCount() - c0, std::move(r[0])};
}

// ---------------------------------------------------------------------------
static void test_first_run_uploads_then_reuses() {
    std::printf(" [1-3] first run uploads, later runs reuse, outputs correct\n");
    Net n; Tensor x = mat(1), w1 = mat(2), w2 = mat(3);
    w1.freeze(); w2.freeze();
    auto ref = n.cpuRef(x, w1, w2);

    ScheduledGraph sg(n.g);
    auto r1 = run(sg, x, w1, w2);
    ok(bothOnCuda(sg), "both MatMuls were placed on CUDA (no silent CPU fallback)");
    ok(r1.bytes == 3 * kMatBytes && r1.count == 3,
       "run 1 uploaded x + W1 + W2 (3 host->device copies, 3x256 KiB) per MemoryStats");
    auto s1 = sg.weightCacheStats();
    ok(s1.misses == 2 && s1.hits == 0 && s1.uploaded_bytes == 2 * kMatBytes &&
       s1.resident_bytes == 2 * kMatBytes && s1.entries == 2,
       "run 1: cache counted 2 misses, 512 KiB resident, 0 hits");
    ok(maxDiff(*r1.out, *ref) < kTol, "run 1 output matches the CPU reference");

    bool correct = true, only_x = true;
    for (int i = 2; i <= 4; ++i) {
        auto r = run(sg, x, w1, w2);
        only_x &= (r.bytes == kMatBytes && r.count == 1);
        correct &= maxDiff(*r.out, *ref) < kTol;
        correct &= bothOnCuda(sg);
    }
    ok(only_x, "runs 2-4 each uploaded ONLY the runtime input (1 copy, 256 KiB) per MemoryStats");
    ok(correct, "runs 2-4 outputs match the CPU reference and stayed on CUDA");
    auto s = sg.weightCacheStats();
    ok(s.misses == 2 && s.hits == 6 && s.uploaded_bytes == 2 * kMatBytes &&
       s.avoided_bytes == 6 * kMatBytes,
       "cache totals: 2 misses, 6 hits, 1.5 MiB of weight uploads avoided over 3 warm runs");

    // Baseline: same workload with the cache disabled re-uploads weights every run.
    ScheduledGraph off(n.g);
    off.setWeightCacheEnabled(false);
    run(off, x, w1, w2);
    auto b = run(off, x, w1, w2);
    ok(b.bytes == 3 * kMatBytes && b.count == 3,
       "control: cache disabled re-uploads x + W1 + W2 on a warm run (counters are measuring uploads)");
    ok(maxDiff(*b.out, *ref) < kTol, "control output also correct");
}

static void test_runtime_input_never_cached() {
    std::printf(" [4] changed runtime input => changed output\n");
    Net n; Tensor x = mat(11), w1 = mat(12), w2 = mat(13);
    w1.freeze(); w2.freeze();
    ScheduledGraph sg(n.g);
    auto r1 = run(sg, x, w1, w2);
    auto ref1 = n.cpuRef(x, w1, w2);

    Tensor x2 = mat(99);                       // different tensor, different data
    auto r2 = run(sg, x2, w1, w2);
    auto ref2 = n.cpuRef(x2, w1, w2);
    ok(maxDiff(*r1.out, *r2.out) > 1e-2, "different input tensor => clearly different output");
    ok(maxDiff(*r2.out, *ref2) < kTol, "new-input output matches its CPU reference (not stale)");
    ok(r2.bytes == kMatBytes, "only the new input was uploaded; weights still resident");

    x.fill(0.25f);                              // same tensor object, mutated IN PLACE
    auto r3 = run(sg, x, w1, w2);
    auto ref3 = n.cpuRef(x, w1, w2);
    ok(maxDiff(*r3.out, *ref3) < kTol && maxDiff(*r3.out, *r1.out) > 1e-2,
       "in-place mutation of the same input tensor is picked up (unfrozen inputs are never cached)");
    ok(maxDiff(*r1.out, *ref1) < kTol, "sanity: first output was correct for the original data");
    ok(sg.weightCacheStats().entries == 2, "cache still holds only the 2 weights, never an input");
}

static void test_separate_graphs_and_model_swap() {
    std::printf(" [5] separate graphs / swapped models never share data\n");
    Net na, nb;
    Tensor x = mat(21);
    Tensor a1 = mat(31), a2 = mat(32), b1 = mat(41), b2 = mat(42);   // same shapes, different values
    a1.freeze(); a2.freeze(); b1.freeze(); b2.freeze();
    auto refA = na.cpuRef(x, a1, a2), refB = nb.cpuRef(x, b1, b2);

    ScheduledGraph sa(na.g), sb(nb.g);
    auto ra = run(sa, x, a1, a2);
    auto rb = run(sb, x, b1, b2);
    ok(maxDiff(*ra.out, *refA) < kTol && maxDiff(*rb.out, *refB) < kTol,
       "two models with same-shaped weights each match their own reference");
    ok(maxDiff(*ra.out, *rb.out) > 1e-2, "the two models produce different outputs");
    ok(rb.bytes == 3 * kMatBytes, "second graph uploaded its own weights (caches are per ScheduledGraph)");
    auto ra2 = run(sa, x, a1, a2), rb2 = run(sb, x, b1, b2);
    ok(ra2.bytes == kMatBytes && rb2.bytes == kMatBytes &&
       maxDiff(*ra2.out, *refA) < kTol && maxDiff(*rb2.out, *refB) < kTol,
       "warm runs of both graphs hit their own caches and stay correct");

    // Swap the model behind ONE ScheduledGraph: same shapes, new tensors.
    auto sw = run(sa, x, b1, b2);
    ok(maxDiff(*sw.out, *refB) < kTol && maxDiff(*sw.out, *refA) > 1e-2,
       "swapping to different weights on one ScheduledGraph yields the new model's output");
    ok(sw.bytes == 3 * kMatBytes, "swapped weights were uploaded fresh (not served from old entries)");
    auto s = sa.weightCacheStats();
    ok(s.entries == 2 && s.resident_bytes == 2 * kMatBytes && s.evictions == 2,
       "old model's 2 device copies were evicted; VRAM held stays at 2 weights");
}

static void test_invalidation() {
    std::printf(" [6] invalidation on unfreeze/modify/refreeze and on escaped writes\n");
    Net n; Tensor x = mat(51), w1 = mat(52), w2 = mat(53);
    w1.freeze(); w2.freeze();
    ScheduledGraph sg(n.g);
    auto before = run(sg, x, w1, w2);
    run(sg, x, w1, w2);

    w1.unfreeze();
    fillLcg(w1, 777);                           // legitimately change the weight
    w1.freeze();                                // new freeze generation
    auto refNew = n.cpuRef(x, w1, w2);
    auto after = run(sg, x, w1, w2);
    auto s = sg.weightCacheStats();
    ok(s.invalidations == 1, "re-frozen weight was detected as stale (1 invalidation)");
    ok(after.bytes == 2 * kMatBytes, "only the changed weight + input were re-uploaded (W2 still resident)");
    ok(maxDiff(*after.out, *refNew) < kTol && maxDiff(*after.out, *before.out) > 1e-2,
       "output reflects the NEW weight values, not the stale device copy");
    auto warm = run(sg, x, w1, w2);
    ok(warm.bytes == kMatBytes && maxDiff(*warm.out, *refNew) < kTol,
       "after re-upload the new weight is resident again (warm run uploads input only)");

    // Escaped write: bypasses Tensor's guards via const_cast (simulates stale raw pointer).
    ScheduledGraph v(n.g);
    v.setWeightCacheVerify(true);
    run(v, x, w1, w2);
    float* sneaky = const_cast<float*>(static_cast<const float*>(static_cast<const Tensor&>(w2).data()));
    sneaky[0] += 3.0f;
    auto refEsc = n.cpuRef(x, w1, w2);
    auto esc = run(v, x, w1, w2);
    ok(v.weightCacheStats().integrity_failures == 1,
       "verify mode: write that bypassed freeze() was caught by the content hash");
    ok(maxDiff(*esc.out, *refEsc) < kTol, "verify mode: output uses the actual (modified) weight, not stale VRAM");
}

static void test_release_and_lifetime() {
    std::printf(" [7] release and destruction return VRAM\n");
    Net n; Tensor x = mat(61), w1 = mat(62), w2 = mat(63);
    w1.freeze(); w2.freeze();
    const size_t base = cudaLive();
    {
        ScheduledGraph sg(n.g);
        run(sg, x, w1, w2);
        ok(cudaLive() - base == 2 * kMatBytes && sg.weightCacheStats().resident_bytes == 2 * kMatBytes,
           "resident weights hold exactly 512 KiB of device memory after a run");
        sg.releaseDeviceWeights();
        ok(cudaLive() == base && sg.weightCacheStats().entries == 0,
           "releaseDeviceWeights() frees all of it immediately");
        auto r = run(sg, x, w1, w2);
        ok(r.bytes == 3 * kMatBytes && sg.weightCacheStats().misses == 4,
           "after release the next run re-uploads and re-caches (correctly)");
        sg.setWeightCacheEnabled(false);
        ok(cudaLive() == base, "disabling the cache frees resident weights");
    }
    {
        ScheduledGraph sg(n.g);
        run(sg, x, w1, w2);
        ok(cudaLive() > base, "(fresh ScheduledGraph holds weights again)");
    }
    ok(cudaLive() == base, "destroying the ScheduledGraph frees its cached device weights");
    // Host weights are untouched by all of this.
    auto ref = n.cpuRef(x, w1, w2);
    ScheduledGraph again(n.g);
    ok(maxDiff(*run(again, x, w1, w2).out, *ref) < kTol && w1.isFrozen() && w2.isFrozen(),
       "host weights intact and still frozen; a new ScheduledGraph works");
}

static void test_budget_fallback() {
    std::printf(" [8] budget limit => transient upload fallback, still correct\n");
    Net n; Tensor x = mat(71), w1 = mat(72), w2 = mat(73);
    w1.freeze(); w2.freeze();
    auto ref = n.cpuRef(x, w1, w2);
    ScheduledGraph sg(n.g);
    sg.setWeightCacheBudgetBytes(kMatBytes);          // room for exactly one weight
    run(sg, x, w1, w2);
    auto warm = run(sg, x, w1, w2);
    auto s = sg.weightCacheStats();
    ok(s.entries == 1 && s.resident_bytes == kMatBytes &&
       s.budget_skips >= 2, "only one weight fits the budget; the other is skipped, not cached");
    ok(warm.bytes == 2 * kMatBytes, "warm run uploads the input + the one uncacheable weight (cached one reused)");
    ok(maxDiff(*warm.out, *ref) < kTol && bothOnCuda(sg), "output correct and still on CUDA under a tight budget");
}

int main() {
    std::printf("========================================\n"
                "ForgeRT Persistent Device-Weight Cache (CUDA)\n"
                "========================================\n");
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    try {
        test_first_run_uploads_then_reuses();
        test_runtime_input_never_cached();
        test_separate_graphs_and_model_swap();
        test_invalidation();
        test_release_and_lifetime();
        test_budget_fallback();
    } catch (const std::exception& e) {
        std::printf("  FAIL: unexpected exception: %s\n", e.what());
        ++g_fail;
    }
    std::printf("========================================\n%s\n========================================\n",
                g_fail == 0 ? "All weight-cache CUDA tests PASSED" : "SOME TESTS FAILED");
    return g_fail == 0 ? 0 : 1;
}
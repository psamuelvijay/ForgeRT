/**
 * @file test_tensor_freeze.cpp
 * @brief CPU-only (no GPU, links forgert_cpu_stub) tests for the immutability
 *        contract and the device-weight cache's non-GPU behaviour.
 *
 * Covers: Tensor freeze/unfreeze semantics and mutation guards, unique ids and
 * generations, move semantics, cache budget accounting, the explicit error when a
 * CUDA transfer is needed but no CUDA backend is linked (never a silent CPU
 * fallback), and that CPU-only ScheduledGraph execution with frozen weights is
 * unaffected. GPU-backed behaviour (hits, avoided bytes, invalidation against real
 * device memory) is in test_weight_cache_cuda.cu.
 */
#include "forgert/forgert.h"
#include "forgert/operator/matmul.h"
#include "forgert/scheduler/scheduler.h"
#include "forgert/scheduler/weight_cache.h"

#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

using namespace forgert;

static int g_fail = 0;
static void ok(bool c, const std::string& m) {
    std::cout << (c ? "PASS: " : "FAIL: ") << m << "\n";
    if (!c) ++g_fail;
}
template <typename Ex, typename F>
static bool throwsAs(F&& f, const char* must_contain = nullptr) {
    try { f(); }
    catch (const Ex& e) {
        return !must_contain || std::string(e.what()).find(must_contain) != std::string::npos;
    } catch (...) { return false; }
    return false;
}

static Tensor makeCpu(size_t n, float v) {
    Tensor t(TensorShape({n}), DataType::Float32, Device::CPU);
    float* p = static_cast<float*>(t.data());
    for (size_t i = 0; i < n; ++i) p[i] = v + static_cast<float>(i);
    return t;
}

static void test_freeze_semantics() {
    Tensor t = makeCpu(8, 1.0f);
    ok(!t.isFrozen() && t.freezeGeneration() == 0, "new tensor is mutable, generation 0");
    t.freeze();
    ok(t.isFrozen() && t.freezeGeneration() != 0, "freeze() freezes and assigns a generation");
    const uint64_t gen = t.freezeGeneration();
    t.freeze();
    ok(t.freezeGeneration() == gen, "freeze() is idempotent (same generation)");

    ok(throwsAs<std::logic_error>([&] { (void)t.data(); }, "frozen"),
       "mutable data() on a frozen tensor throws");
    const Tensor& ct = t;
    ok(ct.data() != nullptr && static_cast<const float*>(ct.data())[3] == 4.0f,
       "const read of a frozen tensor works");
    ok(throwsAs<std::logic_error>([&] { t.zero(); }, "frozen"), "zero() on frozen throws");
    ok(throwsAs<std::logic_error>([&] { t.fill(2.0f); }, "frozen"), "fill() on frozen throws");

    Tensor dst = makeCpu(8, 0.0f);
    ok(throwsAs<std::logic_error>([&] { dst.copyTo(t); }, "frozen"),
       "frozen tensor cannot be a copyTo destination");
    ok(static_cast<const float*>(ct.data())[0] == 1.0f, "frozen contents unchanged by rejected writes");
    t.copyTo(dst);
    ok(static_cast<float*>(dst.data())[7] == 8.0f, "frozen tensor can be a copyTo source");

    t.unfreeze();
    ok(!t.isFrozen(), "unfreeze() makes it mutable");
    t.fill(5.0f);
    ok(static_cast<float*>(t.data())[0] == 5.0f, "mutation works after unfreeze");
    t.freeze();
    ok(t.freezeGeneration() != gen, "re-freeze after unfreeze yields a NEW generation");

    float raw[4] = {1, 2, 3, 4};
    Tensor external(TensorShape({4}), DataType::Float32, raw, Device::CPU);
    ok(throwsAs<std::logic_error>([&] { external.freeze(); }, "non-owning"),
       "non-owning tensor cannot be frozen (cannot be protected)");
}

static void test_ids_and_moves() {
    std::set<uint64_t> ids;
    bool unique = true, nonzero = true;
    for (int i = 0; i < 200; ++i) {
        Tensor t = makeCpu(4, 0.0f);
        nonzero &= t.id() != 0;
        unique &= ids.insert(t.id()).second;   // ids stay unique even when addresses are reused
    }
    ok(nonzero && unique, "200 create/destroy cycles: every tensor id unique and non-zero");

    Tensor a = makeCpu(4, 0.0f);
    a.freeze();
    const uint64_t id = a.id(), gen = a.freezeGeneration();
    Tensor b(std::move(a));
    ok(b.id() == id && b.isFrozen() && b.freezeGeneration() == gen,
       "move construction carries id and frozen state with the buffer");
    ok(a.id() == 0 && !a.isFrozen(), "moved-from tensor has id 0 and is not frozen");
    Tensor c = makeCpu(4, 0.0f);
    c = std::move(b);
    ok(c.id() == id && c.isFrozen(), "move assignment carries id and frozen state");
}

static void test_cache_without_cuda() {
    DeviceWeightCache cache;
    Tensor mut = makeCpu(16, 1.0f);
    ok(cache.acquire(mut) == nullptr, "unfrozen tensor is never cached (acquire -> nullptr)");
    ok(!cache.canAdmit(mut) && !DeviceWeightCache::isCacheable(mut),
       "unfrozen tensor is not cacheable/admissible");

    Tensor w = makeCpu(16, 1.0f);
    w.freeze();
    ok(DeviceWeightCache::isCacheable(w) && cache.canAdmit(w) && !cache.isResident(w),
       "frozen CPU tensor is cacheable and admissible but not resident");
    ok(throwsAs<std::runtime_error>([&] { cache.acquire(w); }, "CUDA"),
       "no CUDA backend linked: acquire() throws an explicit error (no silent fallback)");
    auto s = cache.stats();
    ok(s.entries == 0 && s.misses == 0 && s.uploaded_bytes == 0 && s.resident_bytes == 0,
       "failed acquire leaves the cache empty and counters untouched");

    DeviceWeightCache tiny(8);   // 8-byte budget < 64-byte tensor
    ok(!tiny.canAdmit(w), "tensor larger than budget is not admissible");
    ok(tiny.acquire(w) == nullptr, "over-budget tensor: acquire() -> nullptr, no throw");
    ok(tiny.stats().budget_skips == 1 && tiny.stats().entries == 0, "budget skip is counted");
    tiny.clear();
    ok(tiny.stats().resident_bytes == 0, "clear() on an empty cache is safe");
}

// y = x @ W with a frozen W, in a binary that has no CUDA provider.
static void test_cpu_only_scheduled_graph() {
    const size_t M = 4, K = 8, N = 6;
    Graph g;
    size_t xi = g.addInput(TensorShape({M, K}), DataType::Float32, "x");
    size_t wi = g.addInput(TensorShape({K, N}), DataType::Float32, "w");
    auto out = g.addNode(std::make_unique<MatMulOp>(), {xi, wi}, "mm");
    g.markOutput(out[0]);
    g.validate();

    Tensor x(TensorShape({M, K}), DataType::Float32, Device::CPU);
    Tensor w(TensorShape({K, N}), DataType::Float32, Device::CPU);
    for (size_t i = 0; i < M * K; ++i) static_cast<float*>(x.data())[i] = 0.1f * (i % 7) - 0.2f;
    for (size_t i = 0; i < K * N; ++i) static_cast<float*>(w.data())[i] = 0.05f * (i % 5) - 0.1f;
    w.freeze();

    auto ref = g.execute({&x, &w}, Backend::CPU);

    ScheduledGraph sg(g);
    sg.setVramBudgetBytes(0);   // force CPU placement (no GPU in this binary)
    bool match = true;
    for (int run = 0; run < 3; ++run) {
        auto got = sg.execute({&x, &w});
        const float* a = static_cast<const float*>(ref[0]->data());
        const float* b = static_cast<const float*>(got[0]->data());
        for (size_t i = 0; i < M * N; ++i) match &= std::fabs(a[i] - b[i]) < 1e-6f;
    }
    ok(match, "CPU-only ScheduledGraph with a frozen weight matches Graph::execute over 3 runs");
    auto s = sg.weightCacheStats();
    ok(s.hits == 0 && s.misses == 0 && s.entries == 0, "no cache activity when everything runs on CPU");
    ok(static_cast<const float*>(static_cast<const Tensor&>(w).data())[0] == -0.1f &&
       w.isFrozen(), "weight untouched and still frozen after execution");
}

// A large matmul would be placed on CUDA; with no CUDA backend this must fail
// loudly (frozen or not), never quietly run on the CPU and report a GPU result.
static void test_cuda_unavailable_is_explicit() {
    const size_t M = 256, K = 256, N = 256;
    Graph g;
    size_t xi = g.addInput(TensorShape({M, K}), DataType::Float32, "x");
    size_t wi = g.addInput(TensorShape({K, N}), DataType::Float32, "w");
    auto out = g.addNode(std::make_unique<MatMulOp>(), {xi, wi}, "mm");
    g.markOutput(out[0]);
    g.validate();
    Tensor x(TensorShape({M, K}), DataType::Float32, Device::CPU);
    Tensor w(TensorShape({K, N}), DataType::Float32, Device::CPU);
    x.zero(); w.zero();

    ScheduledGraph unfrozen(g);
    ok(throwsAs<std::runtime_error>([&] { unfrozen.execute({&x, &w}); }, "CUDA"),
       "CUDA-placed node without a CUDA backend throws explicitly (unfrozen weight)");
    w.freeze();
    ScheduledGraph frozen(g);
    ok(throwsAs<std::runtime_error>([&] { frozen.execute({&x, &w}); }, "CUDA"),
       "CUDA-placed node without a CUDA backend throws explicitly (frozen weight)");
    ok(frozen.weightCacheStats().entries == 0, "failed run leaves no cache entry behind");
    ok(w.isFrozen(), "weight remains frozen after the failed run");

    ScheduledGraph off(g);
    off.setWeightCacheEnabled(false);
    ok(throwsAs<std::runtime_error>([&] { off.execute({&x, &w}); }, "CUDA"),
       "cache disabled: still an explicit error, no silent fallback");
}

int main() {
    std::cout << "========================================\n"
              << "ForgeRT Tensor Freeze / Weight Cache (CPU-only)\n"
              << "========================================\n";
    test_freeze_semantics();
    test_ids_and_moves();
    test_cache_without_cuda();
    test_cpu_only_scheduled_graph();
    test_cuda_unavailable_is_explicit();
    std::cout << "========================================\n"
              << (g_fail == 0 ? "All freeze/cache CPU tests PASSED\n"
                              : std::to_string(g_fail) + " test(s) FAILED\n")
              << "========================================\n";
    return g_fail == 0 ? 0 : 1;
}
/*
 * benchmark_weight_residency.cu -- repeated SqueezeNet inference on the mixed
 * CPU/CUDA path, with and without persistent device-resident weights.
 *
 * What it measures (all on the same model, same input, same process):
 *   A. weight-upload cost: the time to copy EXACTLY the model's weights host->device,
 *      by CUDA events -- the per-run cost the cache removes.
 *   B. end-to-end latency per execute(), cache OFF vs ON, interleaved ABAB so drift
 *      (thermals, background load) hits both equally. First ON run reported
 *      separately (it pays the one-time upload).
 *   C. transfer counters per run from MemoryStats (CPU->CUDA bytes/copies) and from
 *      the cache itself (hits, misses, avoided bytes).
 *   D. correctness: every run's output is compared with the CPU reference (outside
 *      the timed region), so a fast wrong answer cannot pass.
 *
 * Bounded: --runs pairs (default 12) and a hard wall-clock guard (--max-seconds, 90).
 * This is a MEASUREMENT tool, not a correctness gate; it is not registered with CTest.
 * Results are only as reliable as the host's state: they are reported with their
 * spread (min / median / max) and must be treated as provisional on a loaded laptop.
 *
 * Usage: benchmark_weight_residency [model.fgx] [--runs N] [--max-seconds S]
 * Exit: 0 ok and correct, 1 incorrect output, 77 skipped (no model / no CUDA).
 */
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "forgert/import/importer.h"
#include "forgert/memory/memory_stats.h"
#include "forgert/scheduler/scheduler.h"
#include "forgert/tensor/tensor.h"

using namespace forgert;
using Clock = std::chrono::steady_clock;

static double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
struct Summary { double min, med, max; };
static Summary summarize(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return {v.front(), (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]), v.back()};
}
static bool fileExists(const std::string& p) {
    if (FILE* f = std::fopen(p.c_str(), "rb")) { std::fclose(f); return true; }
    return false;
}
static double maxDiff(const Tensor& a, const Tensor& b) {
    if (a.numElements() != b.numElements()) return 1e30;
    const float* p = static_cast<const float*>(a.data());
    const float* q = static_cast<const float*>(b.data());
    double m = 0;
    for (size_t i = 0; i < a.numElements(); ++i) {
        const double d = std::fabs(double(p[i]) - double(q[i]));
        if (!(d <= m)) m = d;   // NaN counts as a failure
    }
    return m;
}
static size_t h2dBytes() { return MemoryStats::instance().getTransferBytes(Device::CPU, Device::CUDA); }
static size_t h2dCount() { return MemoryStats::instance().getTransferCount(Device::CPU, Device::CUDA); }

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); return 2; } } while (0)

int main(int argc, char** argv) {
    std::string path;
    int runs = 12;
    double max_seconds = 90.0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--runs") && i + 1 < argc) runs = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--max-seconds") && i + 1 < argc) max_seconds = std::atof(argv[++i]);
        else path = argv[i];
    }
    runs = std::max(3, std::min(runs, 100));
    if (path.empty()) {
        for (const char* c : {"tests/e2e_work/squeezenet1.1-7.fgx", "tests/fgx_work/squeezenet1.1-7.fgx"})
            if (fileExists(c)) { path = c; break; }
    }
    if (path.empty() || !fileExists(path)) { std::printf("SKIP: no .fgx model (pass a path)\n"); return 77; }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("SKIP: no CUDA device\n"); return 77; }
    CK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));

    ImportedModel model = ImportedModel::load(path);
    size_t weight_bytes = 0;
    for (const auto& w : model.weights()) weight_bytes += w->numBytes();
    std::printf("Weight residency benchmark\n  GPU   : %s (sm_%d%d)\n  model : %s\n"
                "  weights: %zu tensors, %.2f MiB (frozen: %s)\n  pairs : %d  (guard %.0f s)\n\n",
                prop.name, prop.major, prop.minor, path.c_str(), model.numWeights(),
                weight_bytes / 1048576.0, model.weights().front()->isFrozen() ? "yes" : "NO", runs, max_seconds);

    // Deterministic input and CPU reference (the already-validated CPU path).
    Tensor input(model.inputShapes()[0], DataType::Float32, Device::CPU);
    {
        float* p = static_cast<float*>(input.data());
        for (size_t i = 0; i < input.numElements(); ++i)
            p[i] = (static_cast<float>((i * 2654435761u) % 3001) - 1500.0f) / 1000.0f;
    }
    auto ref = model.run({&input});
    const std::vector<const Tensor*> all = model.graphInputs({&input});

    // ---- A. weight-upload cost (CUDA events), exactly the data the cache avoids ----
    std::vector<std::unique_ptr<Tensor>> dev;
    for (const auto& w : model.weights())
        dev.push_back(std::make_unique<Tensor>(w->shape(), w->dtype(), Device::CUDA));
    auto uploadAll = [&]() { for (size_t i = 0; i < dev.size(); ++i) model.weights()[i]->copyTo(*dev[i]); };
    for (int i = 0; i < 3; ++i) uploadAll();                       // warm-up
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    std::vector<double> up;
    for (int i = 0; i < 15; ++i) {
        CK(cudaEventRecord(e0)); uploadAll(); CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float t = 0; CK(cudaEventElapsedTime(&t, e0, e1)); up.push_back(t);
    }
    dev.clear();
    const Summary U = summarize(up);
    std::printf("A. Uploading all %zu weights (%.2f MiB), cudaEvent, 15 samples:\n"
                "     min %.3f  median %.3f  max %.3f ms   (~%.2f GB/s at median)\n\n",
                model.numWeights(), weight_bytes / 1048576.0, U.min, U.med, U.max,
                weight_bytes / (U.med * 1e6));

    // ---- A2. what a cache-OFF run really does to the weights every time:
    //          cudaMalloc + upload + cudaFree for each tensor (host clock, synchronized).
    {
        auto cycle = [&]() {
            std::vector<std::unique_ptr<Tensor>> tmp;
            for (const auto& w : model.weights()) {
                tmp.push_back(std::make_unique<Tensor>(w->shape(), w->dtype(), Device::CUDA));
                w->copyTo(*tmp.back());
            }
            cudaDeviceSynchronize();
            tmp.clear();                       // cudaFree each
            cudaDeviceSynchronize();
        };
        for (int i = 0; i < 3; ++i) cycle();
        std::vector<double> t;
        for (int i = 0; i < 15; ++i) { const auto a = Clock::now(); cycle(); t.push_back(ms(a, Clock::now())); }
        const Summary S = summarize(t);
        std::printf("A2. Allocate + upload + free all %zu weights (what a cache-OFF run does), 15 samples:\n"
                    "     min %.3f  median %.3f  max %.3f ms\n\n", model.numWeights(), S.min, S.med, S.max);
    }

    // ---- B/C/D. interleaved end-to-end, cache OFF vs ON ----
    ScheduledGraph off(model.graph()), on(model.graph());
    off.setWeightCacheEnabled(false);

    struct Run { double ms; size_t bytes, count; double diff; };
    auto one = [&](ScheduledGraph& sg) -> Run {
        const size_t b0 = h2dBytes(), c0 = h2dCount();
        const auto t0 = Clock::now();
        auto out = sg.execute(all);
        const auto t1 = Clock::now();
        Run r{ms(t0, t1), h2dBytes() - b0, h2dCount() - c0, 0.0};
        r.diff = maxDiff(*out[0], *ref[0]);       // consumes the output; outside the timed region
        return r;
    };

    one(off);                                      // untimed warm-up of the OFF path (context, allocator)
    const Run first_on = one(on);                  // FIRST cached run: pays the one-time upload
    std::vector<double> t_off, t_on;
    std::vector<size_t> b_off, b_on, c_off, c_on;
    double worst = std::max(first_on.diff, 0.0);
    const auto wall0 = Clock::now();
    int done = 0;
    for (int i = 0; i < runs; ++i) {
        if (ms(wall0, Clock::now()) > max_seconds * 1000.0) { std::printf("  (stopped early: wall-clock guard)\n"); break; }
        const bool off_first = (i % 2) == 0;       // alternate order to cancel position bias
        Run a = off_first ? one(off) : one(on);
        Run b = off_first ? one(on) : one(off);
        const Run& ro = off_first ? a : b;
        const Run& rn = off_first ? b : a;
        t_off.push_back(ro.ms); b_off.push_back(ro.bytes); c_off.push_back(ro.count);
        t_on.push_back(rn.ms);  b_on.push_back(rn.bytes);  c_on.push_back(rn.count);
        worst = std::max({worst, ro.diff, rn.diff});
        ++done;
    }
    if (done < 3) { std::printf("FAIL: too few samples\n"); return 1; }

    const Summary S_off = summarize(t_off), S_on = summarize(t_on);
    auto bytes_min = [](const std::vector<size_t>& v) { return *std::min_element(v.begin(), v.end()); };
    auto bytes_max = [](const std::vector<size_t>& v) { return *std::max_element(v.begin(), v.end()); };
    const WeightCacheStats cs = on.weightCacheStats();

    std::printf("B. End-to-end execute() latency, %d interleaved pairs (host clock, includes CPU ops):\n", done);
    std::printf("     cache OFF : min %8.2f  median %8.2f  max %8.2f ms\n", S_off.min, S_off.med, S_off.max);
    std::printf("     cache ON  : min %8.2f  median %8.2f  max %8.2f ms\n", S_on.min, S_on.med, S_on.max);
    std::printf("     first ON run (one-time upload): %.2f ms\n", first_on.ms);
    std::printf("     median difference OFF - ON    : %+.2f ms (%.1f%% of OFF median)\n\n",
                S_off.med - S_on.med, 100.0 * (S_off.med - S_on.med) / S_off.med);

    std::printf("C. Host->device transfers per run (MemoryStats, CPU->CUDA):\n");
    std::printf("     cache OFF : %zu copies, %.2f MiB per run (min..max bytes %.2f..%.2f MiB)\n",
                c_off[0], b_off[0] / 1048576.0, bytes_min(b_off) / 1048576.0, bytes_max(b_off) / 1048576.0);
    std::printf("     cache ON  : %zu copies, %.2f MiB per run (min..max bytes %.2f..%.2f MiB)\n",
                c_on[0], b_on[0] / 1048576.0, bytes_min(b_on) / 1048576.0, bytes_max(b_on) / 1048576.0);
    std::printf("     first ON run: %zu copies, %.2f MiB\n", first_on.count, first_on.bytes / 1048576.0);
    std::printf("     cache stats: hits %llu, misses %llu, uploaded %.2f MiB, AVOIDED %.2f MiB, "
                "resident %.2f MiB in %zu tensors\n\n",
                (unsigned long long)cs.hits, (unsigned long long)cs.misses, cs.uploaded_bytes / 1048576.0,
                cs.avoided_bytes / 1048576.0, cs.resident_bytes / 1048576.0, cs.entries);

    std::printf("D. Correctness: worst max-abs-diff vs CPU reference over all %d runs = %.3e (tol 1e-3)\n",
                2 * done + 2, worst);
    const bool correct = worst < 1e-3;
    std::printf("\n%s\n", correct ? "RESULT: outputs correct on every run." : "RESULT: FAIL - output mismatch.");
    std::printf("Note: single-host, interleaved, spread shown; provisional unless the host was idle.\n");
    return correct ? 0 : 1;
}
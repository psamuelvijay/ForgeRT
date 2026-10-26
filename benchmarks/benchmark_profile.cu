/*
 * benchmark_profile.cu -- machine-readable profiling run for scheduler calibration.
 *
 * Emits a single JSON object (to stdout, or to the file given as argv[1])
 * capturing the raw measurements the scheduler cost model is derived from:
 *
 *   - CPU operator latency           (chrono, pre-allocated host tensors)
 *   - GPU kernel-only latency        (cudaEvent, tensors already resident)
 *   - GPU end-to-end latency         (H2D + kernel + D2H, pre-allocated device)
 *   - H2D and D2H transfer latency   (isolated, -> effective bandwidth bytes/ns)
 *   - CUDA launch overhead           (null kernel)
 *
 * Methodology
 * -----------
 *   * Warm-up before every measured series (CUDA context + caches).
 *   * Each measurement is an AVERAGE over `iters` back-to-back ops, repeated for
 *     `samples` independent series. We report median / min / stddev ACROSS the
 *     per-series averages -- a single series is never presented as the result.
 *   * GPU timing uses cudaEvent_t with cudaEventSynchronize.
 *   * CPU timing uses std::chrono::high_resolution_clock.
 *   * Allocation (cudaMalloc / host malloc) is done ONCE outside the timed loop,
 *     so steady-state execution is separated from setup cost.
 *
 * This program MEASURES only. It does not derive scheduler profiles or decide
 * CPU/GPU placement. Derivation is a separate, reviewable step:
 * tools/calibrate_profiles.py reads this JSON and emits derived profiles.
 *
 * IMPORTANT: results are only as trustworthy as the host. On a thermally
 * constrained or loaded machine, numbers are PROVISIONAL. The JSON records the
 * host so provenance travels with the data; it does not and cannot certify that
 * the host was quiescent. See docs/PROFILING.md.
 */

#include "forgert/operator/relu.h"
#include "forgert/operator/add.h"
#include "forgert/operator/matmul.h"
#include "forgert/operator/softmax.h"
#include "forgert/operator/layernorm.h"
#include "forgert/operator/conv2d.h"
#include "forgert/tensor/tensor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

using namespace forgert;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s:%d  %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(_e));                              \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------- statistics --
struct Stats {
    double median = 0.0, min = 0.0, max = 0.0, mean = 0.0, stddev = 0.0;
    int samples = 0;
    int iters = 0;     // inner iterations actually used per sample (adaptive)
    int warmup = 0;
};

static Stats summarize(std::vector<double> xs) {
    Stats s;
    if (xs.empty()) return s;
    s.samples = static_cast<int>(xs.size());
    std::sort(xs.begin(), xs.end());
    s.min = xs.front();
    s.max = xs.back();
    const size_t n = xs.size();
    s.median = (n % 2) ? xs[n / 2] : 0.5 * (xs[n / 2 - 1] + xs[n / 2]);
    double sum = 0.0; for (double v : xs) sum += v;
    s.mean = sum / n;
    double acc = 0.0; for (double v : xs) acc += (v - s.mean) * (v - s.mean);
    s.stddev = (n > 1) ? std::sqrt(acc / (n - 1)) : 0.0;
    return s;
}

// --------------------------------------------------------------------- timing --
// One "series" = average ms over `iters` back-to-back calls. We collect
// `samples` series and summarize across them.
// Cost-adaptive iteration count. A fixed count that suits a 1K-element ReLU
// (microseconds) would run a 512x512 naive CPU matmul (hundreds of ms per call)
// for minutes. One probe call sets iters so a single sample targets ~kSampleMs,
// capped at the requested maximum; heavy ops get 1 warm-up and iters=1.
static const double kSampleMs = 20.0;
static Stats timeCPU(const std::function<void()>& fn, int warmup, int iters, int samples) {
    auto p0 = std::chrono::high_resolution_clock::now();
    fn();                                            // probe (also a warm-up call)
    auto p1 = std::chrono::high_resolution_clock::now();
    const double probe_ms = std::chrono::duration<double, std::milli>(p1 - p0).count();
    if (probe_ms > 0.0) {
        const int fit = static_cast<int>(kSampleMs / probe_ms);
        iters = std::max(1, std::min(iters, fit));
    }
    if (probe_ms > 50.0) warmup = 1;
    for (int i = 0; i < warmup; ++i) fn();
    std::vector<double> series;
    series.reserve(samples);
    for (int s = 0; s < samples; ++s) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        series.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / iters);
    }
    Stats st = summarize(series); st.iters = iters; st.warmup = warmup; return st;
}

static Stats timeGPU(const std::function<void()>& fn, int warmup, int iters, int samples) {
    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> series;
    series.reserve(samples);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int s = 0; s < samples; ++s) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; ++i) fn();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        series.push_back(static_cast<double>(ms) / iters);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    Stats st = summarize(series); st.iters = iters; st.warmup = warmup; return st;
}

// null kernel for launch overhead
__global__ void null_kernel(float* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = 1.0f;
}

static void fillHost(Tensor& t, float v = 0.1f) {
    float* p = static_cast<float*>(t.data());
    for (size_t i = 0; i < t.numElements(); ++i) p[i] = v;
}

// Anti-elision machinery.
//
// An earlier version consumed only out[0]; the compiler then kept just that one
// element's computation live and elided the rest, so a 1-element and a 1M-element
// ReLU timed identically (~0.0006 ns/elem, physically impossible). To force the
// WHOLE operator to run, consume() reduces over EVERY output element into a
// volatile sink -- the optimizer cannot prove any element dead.
//
// This means CPU timings include one linear read pass over the output in
// addition to the operator itself. That is a deliberate, documented tradeoff:
// it is the standard "consume the result" benchmark discipline, it is O(N) like
// the op, and it is honest (it slightly over-counts memory-bound ops rather than
// wildly under-counting them via elision). perturb() additionally feeds the sink
// back into one input element so a loop-invariant call cannot be hoisted by LICM.
static volatile float g_sink = 0.0f;
static inline void consume(const Tensor& t) {
    const float* p = static_cast<const float*>(t.data());
    const size_t n = t.numElements();
    float acc = 0.0f;
    for (size_t i = 0; i < n; ++i) acc += p[i];
    g_sink = acc;
}
static inline void perturb(Tensor& in) {
    float* p = static_cast<float*>(in.data());
    p[0] = 0.1f + 1e-7f * g_sink;
}

// ----------------------------------------------------------------- JSON output --
// Minimal hand-written JSON writer (emit only; no parser needed here).
struct Json {
    std::string s;
    bool firstInObj = true;
    void raw(const std::string& t) { s += t; }
    void key(const char* k) {
        if (!firstInObj) s += ",";
        firstInObj = false;
        s += "\""; s += k; s += "\":";
    }
    void beginObj() { s += "{"; firstInObj = true; }
    void endObj()   { s += "}"; firstInObj = false; }
};

static void emitStats(Json& j, const char* name, const Stats& st) {
    j.key(name); j.beginObj();
    j.key("median_ms"); j.raw(std::to_string(st.median));
    j.key("min_ms");    j.raw(std::to_string(st.min));
    j.key("max_ms");    j.raw(std::to_string(st.max));
    j.key("mean_ms");   j.raw(std::to_string(st.mean));
    j.key("stddev_ms"); j.raw(std::to_string(st.stddev));
    j.key("samples");   j.raw(std::to_string(st.samples));
    j.key("inner_iters"); j.raw(std::to_string(st.iters));
    j.key("warmup");    j.raw(std::to_string(st.warmup));
    j.endObj();
}

// measurement config
static const int WARMUP  = 15;
static const int ITERS   = 100;
static const int SAMPLES = 15;

// Each op measurement produces a JSON object with shape, elems, and the stats.
struct Fns {
    std::function<void()> cpu, kernel, e2e;
};

static void benchOp(Json& j, bool& firstEntry, const char* op, const std::string& shape,
                    size_t elems, size_t in_bytes, size_t out_bytes, const Fns& fns) {
    Stats cpu    = timeCPU(fns.cpu,       WARMUP, ITERS, SAMPLES);
    Stats kernel = timeGPU(fns.kernel,    WARMUP, ITERS, SAMPLES);
    Stats e2e    = timeGPU(fns.e2e,       WARMUP, ITERS, SAMPLES);

    if (!firstEntry) j.raw(",");
    firstEntry = false;
    j.beginObj();
    j.key("op");        j.raw(std::string("\"") + op + "\"");
    j.key("shape");     j.raw(std::string("\"") + shape + "\"");
    j.key("elems");     j.raw(std::to_string(elems));
    j.key("input_bytes");  j.raw(std::to_string(in_bytes));
    j.key("output_bytes"); j.raw(std::to_string(out_bytes));
    emitStats(j, "cpu", cpu);
    emitStats(j, "gpu_kernel", kernel);
    emitStats(j, "gpu_e2e", e2e);
    j.endObj();
}


int main(int argc, char** argv) {
    const char* out_path = (argc > 1) ? argv[1] : nullptr;

    // Warm up CUDA context.
    { Tensor dummy(TensorShape({1}), DataType::Float32, Device::CUDA);
      CUDA_CHECK(cudaDeviceSynchronize()); }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int rtVer = 0, drVer = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&rtVer));
    CUDA_CHECK(cudaDriverGetVersion(&drVer));

    Json j;
    j.beginObj();

    // ---- provenance (queried, not hardcoded) ----
    j.key("schema"); j.raw("\"forgert.profile.raw/v1\"");
    j.key("provenance"); j.beginObj();
    j.key("gpu_name");      j.raw(std::string("\"") + prop.name + "\"");
    j.key("gpu_sm");        j.raw(std::string("\"sm_") + std::to_string(prop.major) + std::to_string(prop.minor) + "\"");
    j.key("gpu_vram_mb");   j.raw(std::to_string(static_cast<unsigned long long>(prop.totalGlobalMem >> 20)));
    j.key("cuda_runtime");  j.raw(std::to_string(rtVer));
    j.key("cuda_driver");   j.raw(std::to_string(drVer));
    j.key("warmup_iters");  j.raw(std::to_string(WARMUP));
    j.key("inner_iters");   j.raw(std::to_string(ITERS));
    j.key("samples");       j.raw(std::to_string(SAMPLES));
    j.key("note"); j.raw("\"PROVISIONAL unless the host was on AC power, thermally stable, and idle; this program cannot certify that.\"");
    j.endObj();

    // ---- transfer latency / bandwidth (isolated H2D and D2H) ----
    j.key("transfers"); j.raw("[");
    bool firstT = true;
    for (size_t n : {size_t(65536), size_t(1048576), size_t(4194304)}) {
        TensorShape shape({n});
        size_t bytes = n * sizeof(float);
        Tensor host(shape, DataType::Float32, Device::CPU);
        Tensor dev(shape, DataType::Float32, Device::CUDA);
        fillHost(host);
        Stats h2d = timeGPU([&]() { host.copyTo(dev); }, WARMUP, ITERS, SAMPLES);
        Stats d2h = timeGPU([&]() { dev.copyTo(host); }, WARMUP, ITERS, SAMPLES);
        if (!firstT) j.raw(","); firstT = false;
        j.beginObj();
        j.key("bytes"); j.raw(std::to_string(bytes));
        emitStats(j, "h2d", h2d);
        emitStats(j, "d2h", d2h);
        // effective bytes/ns from median ms
        double h2d_bw = (h2d.median > 0) ? bytes / (h2d.median * 1e6) : 0.0;  // ms->ns = *1e6
        double d2h_bw = (d2h.median > 0) ? bytes / (d2h.median * 1e6) : 0.0;
        j.key("h2d_bytes_per_ns"); j.raw(std::to_string(h2d_bw));
        j.key("d2h_bytes_per_ns"); j.raw(std::to_string(d2h_bw));
        j.endObj();
    }
    j.raw("]");

    // ---- launch overhead (null kernel) ----
    float* d_null = nullptr;
    CUDA_CHECK(cudaMalloc(&d_null, 16));
    Stats launch = timeGPU([&]() { null_kernel<<<1, 256>>>(d_null); }, 50, 1000, SAMPLES);
    CUDA_CHECK(cudaFree(d_null));
    j.key("launch_overhead"); j.beginObj();
    emitStats(j, "null_kernel", launch);
    j.key("launch_overhead_ns"); j.raw(std::to_string(launch.median * 1e6));  // ms->ns
    j.endObj();

    // ---- per-operator latency ----
    //
    // Every (op, size) case OWNS its tensors for the duration of its measurement.
    // An earlier version declared the tensors `static` inside a lambda; a
    // function-local static is initialised exactly once, so every larger size
    // silently re-measured the FIRST (smallest) tensors while reporting the larger
    // element count. That produced flat, physically impossible ns/elem figures and
    // was misdiagnosed as compiler elision. tests/test_profile_calibration.py
    // pins the symptom (time must scale with size) so it cannot regress silently.
    j.key("operators"); j.raw("[");
    bool firstOp = true;

    // ReLU (unary elementwise)
    for (size_t n : {size_t(1024), size_t(65536), size_t(1048576)}) {
        TensorShape sh({n});
        Tensor h_in(sh, DataType::Float32, Device::CPU);
        Tensor h_out(sh, DataType::Float32, Device::CPU);
        Tensor d_in(sh, DataType::Float32, Device::CUDA);
        Tensor d_out(sh, DataType::Float32, Device::CUDA);
        fillHost(h_in); h_in.copyTo(d_in);
        ReLUOp op;
        Fns f;
        f.cpu    = [&]() { perturb(h_in); op.execute(Backend::CPU,  {&h_in}, {&h_out}); consume(h_out); };
        f.kernel = [&]() { op.execute(Backend::CUDA, {&d_in}, {&d_out}); };
        f.e2e    = [&]() { h_in.copyTo(d_in); op.execute(Backend::CUDA, {&d_in}, {&d_out}); d_out.copyTo(h_out); consume(h_out); };
        benchOp(j, firstOp, "ReLU", "[" + std::to_string(n) + "]", n,
                n * sizeof(float), n * sizeof(float), f);
    }

    // MatMul (square, compute bound)
    for (size_t m : {size_t(256), size_t(512)}) {
        const size_t k = m, n = m;
        Tensor h_a(TensorShape({m, k}), DataType::Float32, Device::CPU);
        Tensor h_b(TensorShape({k, n}), DataType::Float32, Device::CPU);
        Tensor h_c(TensorShape({m, n}), DataType::Float32, Device::CPU);
        Tensor d_a(TensorShape({m, k}), DataType::Float32, Device::CUDA);
        Tensor d_b(TensorShape({k, n}), DataType::Float32, Device::CUDA);
        Tensor d_c(TensorShape({m, n}), DataType::Float32, Device::CUDA);
        fillHost(h_a); fillHost(h_b, 0.2f); h_a.copyTo(d_a); h_b.copyTo(d_b);
        MatMulOp op;
        Fns f;
        f.cpu    = [&]() { perturb(h_a); op.execute(Backend::CPU,  {&h_a, &h_b}, {&h_c}); consume(h_c); };
        f.kernel = [&]() { op.execute(Backend::CUDA, {&d_a, &d_b}, {&d_c}); };
        f.e2e    = [&]() { h_a.copyTo(d_a); h_b.copyTo(d_b); op.execute(Backend::CUDA, {&d_a, &d_b}, {&d_c}); d_c.copyTo(h_c); consume(h_c); };
        std::string shp = "[" + std::to_string(m) + "," + std::to_string(k) + "]@[" +
                          std::to_string(k) + "," + std::to_string(n) + "]";
        benchOp(j, firstOp, "MatMul", shp, m * n,
                (m * k + k * n) * sizeof(float), m * n * sizeof(float), f);
    }
    j.raw("]");
    j.endObj();

    if (out_path) {
        FILE* f = std::fopen(out_path, "wb");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", out_path); return 1; }
        std::fwrite(j.s.data(), 1, j.s.size(), f);
        std::fclose(f);
        std::fprintf(stderr, "wrote %s (%zu bytes)\n", out_path, j.s.size());
    } else {
        std::fwrite(j.s.data(), 1, j.s.size(), stdout);
        std::fputc('\n', stdout);
    }
    return 0;
}
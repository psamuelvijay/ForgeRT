/*
 * gpu_stress.cu -- bounded, verifiable GPU workload for ForgeRT.
 *
 * Repeatedly runs forgert::MatMulOp on Backend::CUDA (which dispatches to
 * forgert_matmul_cuda -> matmul_kernel) so GPU utilisation is visible in
 * Task Manager / nvidia-smi, and proves real GPU work happened:
 *   - reports CUDA device discovery (count, name, compute capability, VRAM)
 *   - output buffer is pre-filled with NaN so an un-run kernel cannot pass
 *   - kernel time is measured with CUDA events, not host clocks
 *   - the result is checked against a double-precision CPU reference
 *   - achieved GFLOP/s is compared against the device's theoretical peak
 *
 * Bounded: stops after --seconds of wall time (hard-capped at 60 s) or
 * kMaxIters iterations, whichever is first. Memory: 3 * N*N floats.
 *
 * Usage: gpu_stress [--seconds S=15] [--n N=2048] [--countdown C=0]
 * Exit status: 0 = GPU work verified correct, non-zero otherwise.
 */
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <thread>
#include <vector>

#include "forgert/operator/matmul.h"
#include "forgert/tensor/tensor.h"

using namespace forgert;

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t e_ = (call);                                              \
        if (e_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__,          \
                         __LINE__, cudaGetErrorString(e_));                   \
            return 2;                                                         \
        }                                                                     \
    } while (0)

static constexpr int kMaxSeconds = 60;
static constexpr int kMaxN = 4096;        // 3 * 64 MB = 192 MB VRAM at the cap
static constexpr long kMaxIters = 100000; // secondary stop condition
static constexpr int kWarmup = 3;

// Deterministic values in [-1, 1).
static void fill(float* p, size_t n, uint32_t seed) {
    uint32_t s = seed;
    for (size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        p[i] = (static_cast<float>(s >> 8) / 8388608.0f) - 1.0f;
    }
}

int main(int argc, char** argv) {
    int seconds = 15, n = 2048, countdown = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--seconds")) seconds = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--n")) n = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--countdown")) countdown = std::atoi(argv[i + 1]);
        else { std::fprintf(stderr, "unknown arg %s\n", argv[i]); return 64; }
    }
    if (seconds < 1) seconds = 1;
    if (seconds > kMaxSeconds) seconds = kMaxSeconds;
    if (n < 64) n = 64;
    if (n > kMaxN) n = kMaxN;
    if (countdown < 0) countdown = 0;
    if (countdown > 60) countdown = 60;

    // ---- 1. device discovery ------------------------------------------------
    int dev_count = 0;
    cudaError_t derr = cudaGetDeviceCount(&dev_count);
    std::printf("[discovery] cudaGetDeviceCount -> %d (%s)\n", dev_count,
                cudaGetErrorString(derr));
    if (derr != cudaSuccess || dev_count < 1) {
        std::fprintf(stderr, "FAIL: no CUDA device usable\n");
        return 2;
    }
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int clock_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    // FP32 cores/SM: 128 for cc 6.1 (Pascal GP107). Peak = 2 * cores * clock.
    const int cores_per_sm = (prop.major == 6 && prop.minor == 0) ? 64 : 128;
    const double peak_gflops =
        2.0 * prop.multiProcessorCount * cores_per_sm * (clock_khz / 1e6);
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    std::printf("[discovery] device 0: %s, sm_%d%d, %d SMs, %.0f MHz, VRAM %.0f MiB (free %.0f MiB)\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                clock_khz / 1e3, total_b / 1048576.0, free_b / 1048576.0);
    std::printf("[discovery] theoretical FP32 peak ~= %.0f GFLOP/s\n", peak_gflops);

    const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
    const double vram_mib = 3.0 * bytes / 1048576.0;
    std::printf("[config] N=%d, matmul %dx%d @ %dx%d, VRAM use ~%.0f MiB, run %d s\n",
                n, n, n, n, n, vram_mib, seconds);
    if (3 * bytes > free_b / 2) {
        std::fprintf(stderr, "FAIL: would use more than half of free VRAM\n");
        return 2;
    }

    // ---- 2. tensors, uploaded through ForgeRT's own transfer path -----------
    const size_t un = static_cast<size_t>(n);
    int rc = 0;
    try {
        Tensor h_a(TensorShape({un, un}), DataType::Float32, Device::CPU);
        Tensor h_b(TensorShape({un, un}), DataType::Float32, Device::CPU);
        Tensor h_c(TensorShape({un, un}), DataType::Float32, Device::CPU);
        fill(static_cast<float*>(h_a.data()), un * un, 1u);
        fill(static_cast<float*>(h_b.data()), un * un, 2u);

        Tensor d_a(TensorShape({un, un}), DataType::Float32, Device::CUDA);
        Tensor d_b(TensorShape({un, un}), DataType::Float32, Device::CUDA);
        Tensor d_c(TensorShape({un, un}), DataType::Float32, Device::CUDA);
        h_a.copyTo(d_a);
        h_b.copyTo(d_b);
        // 0xFF bytes == NaN: a kernel that never ran cannot pass verification.
        CUDA_CHECK(cudaMemset(d_c.data(), 0xFF, bytes));
        CUDA_CHECK(cudaDeviceSynchronize());

        MatMulOp op;
        std::vector<const Tensor*> in = {&d_a, &d_b};
        std::vector<Tensor*> out = {&d_c};

        // ---- 3. warm-up (also first correctness check target) --------------
        for (int i = 0; i < kWarmup; ++i) op.execute(Backend::CUDA, in, out);
        CUDA_CHECK(cudaDeviceSynchronize());

        if (countdown > 0) {
            std::printf("[ready] Open Task Manager -> Performance -> GPU 1. Starting in %d s...\n",
                        countdown);
            std::fflush(stdout);
            std::this_thread::sleep_for(std::chrono::seconds(countdown));
        }

        // ---- 4. timed, bounded loop ----------------------------------------
        cudaEvent_t ev0, ev1;
        CUDA_CHECK(cudaEventCreate(&ev0));
        CUDA_CHECK(cudaEventCreate(&ev1));
        double gpu_ms_total = 0.0;
        long iters = 0;
        const auto t0 = std::chrono::steady_clock::now();
        auto last_report = t0;
        std::printf("[run] GPU workload START\n");
        std::fflush(stdout);
        while (iters < kMaxIters) {
            CUDA_CHECK(cudaEventRecord(ev0));
            op.execute(Backend::CUDA, in, out); // launches + syncs internally
            CUDA_CHECK(cudaEventRecord(ev1));
            CUDA_CHECK(cudaEventSynchronize(ev1));
            float ms = 0.f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
            gpu_ms_total += ms;
            ++iters;
            const auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - last_report).count() >= 1.0) {
                last_report = now;
                std::printf("[run] t=%4.1fs iters=%ld last=%.2f ms\n",
                            std::chrono::duration<double>(now - t0).count(), iters, ms);
                std::fflush(stdout);
            }
            if (std::chrono::duration<double>(now - t0).count() >= seconds) break;
        }
        const double wall_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("[run] GPU workload END\n");

        // ---- 5. result verification vs double-precision CPU reference ------
        d_c.copyTo(h_c);
        const float* A = static_cast<const float*>(h_a.data());
        const float* B = static_cast<const float*>(h_b.data());
        const float* C = static_cast<const float*>(h_c.data());
        size_t checked = 0, bad = 0;
        double max_abs = 0.0;
        const size_t rows[] = {0, un / 7, un / 3, un / 2, (2 * un) / 3, un - 1};
        for (size_t r : rows) {
            for (size_t j = 0; j < un; ++j) {
                double ref = 0.0;
                for (size_t k = 0; k < un; ++k)
                    ref += static_cast<double>(A[r * un + k]) * B[k * un + j];
                const double got = C[r * un + j];
                const double err = std::fabs(got - ref); // NaN -> comparison false
                ++checked;
                if (!(err <= 1e-3 + 1e-4 * std::fabs(ref))) ++bad;
                else if (err > max_abs) max_abs = err;
            }
        }

        const double flops_per_iter = 2.0 * un * un * un;
        const double avg_ms = gpu_ms_total / iters;
        const double gflops = flops_per_iter / (avg_ms * 1e6);
        std::printf("\n[result] iterations          : %ld\n", iters);
        std::printf("[result] wall time           : %.2f s\n", wall_s);
        std::printf("[result] GPU time (events)   : %.2f s (%.0f%% of wall)\n",
                    gpu_ms_total / 1e3, 100.0 * gpu_ms_total / 1e3 / wall_s);
        std::printf("[result] avg kernel+sync     : %.2f ms\n", avg_ms);
        std::printf("[result] throughput          : %.1f GFLOP/s (%.0f%% of theoretical peak)\n",
                    gflops, 100.0 * gflops / peak_gflops);
        std::printf("[result] correctness         : %zu/%zu sampled outputs within tol, max abs err %.2e\n",
                    checked - bad, checked, max_abs);

        const bool plausible = gflops > 20.0 && gflops < peak_gflops * 1.05;
        if (bad != 0) { std::printf("[result] FAIL: mismatches vs CPU reference\n"); rc = 1; }
        else if (!plausible) { std::printf("[result] FAIL: throughput implausible for GPU work\n"); rc = 1; }
        else std::printf("[result] PASS: MatMulOp ran on CUDA, results verified\n");

        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: exception: %s\n", e.what());
        return 3;
    }
    return rc;
}

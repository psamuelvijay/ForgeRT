/*
 * measure_launch_overhead.cu
 *
 * Measures actual CUDA kernel launch overhead by timing a do-nothing kernel
 * across a range of iterations, then computing the fixed cost per launch.
 *
 * Also shows how launch overhead relates to operator size breakeven points.
 */
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ \
                  << "  " << cudaGetErrorString(_e) << "\n"; \
        std::exit(1); \
    } } while(0)

// Minimal kernel that does exactly one memory touch — isolates launch overhead
__global__ void null_kernel(float* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = 1.0f;
}

// Tiny ReLU to show overhead at small element counts
__global__ void relu_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (in[i] > 0.0f) ? in[i] : 0.0f;
}

template<typename Fn>
static double timeGPU(Fn fn, int warmup = 20, int iters = 1000) {
    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return static_cast<double>(ms) / iters * 1000.0;  // return microseconds
}

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << "\n\n";

    // Warm up CUDA context
    float* dummy;
    CUDA_CHECK(cudaMalloc(&dummy, 16));

    // --- 1. Null kernel overhead (pure launch cost, ~1 thread) ---
    double null_us = timeGPU([&]() {
        null_kernel<<<1, 1>>>(dummy);
    });
    std::cout << "Null kernel (1 block, 1 thread): " << std::fixed
              << std::setprecision(2) << null_us << " us\n";

    // --- 2. Null kernel with 256 threads ---
    double null256_us = timeGPU([&]() {
        null_kernel<<<1, 256>>>(dummy);
    });
    std::cout << "Null kernel (1 block, 256 threads): "
              << null256_us << " us\n\n";

    // --- 3. ReLU at small sizes: overhead vs work ---
    std::cout << std::left
              << std::setw(14) << "ReLU elems"
              << std::setw(16) << "Kernel (us)"
              << std::setw(20) << "Launch frac (%)"
              << "Decision implication\n"
              << std::string(72, '-') << "\n";

    float* d_in;  float* d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  1048576 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, 1048576 * sizeof(float)));

    const double LAUNCH_US = null_us;
    for (int n : {32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, 262144, 1048576}) {
        int blocks = (n + 255) / 256;
        double kernel_us = timeGPU([&]() {
            relu_kernel<<<blocks, 256>>>(d_in, d_out, n);
        });
        double launch_frac = (LAUNCH_US / kernel_us) * 100.0;

        // Breakeven analysis:
        // CPU ReLU cost = 0.40 ns/elem * n (ns) = 0.0004 * n (us)
        // GPU cost = launch_overhead + compute
        double cpu_us = 0.40e-3 * n;  // 0.40 ns/elem converted to us
        const char* decision = (cpu_us < kernel_us) ? "CPU wins" : "GPU wins";

        std::cout << std::setw(14) << n
                  << std::setw(16) << std::setprecision(3) << kernel_us
                  << std::setw(20) << std::setprecision(1) << launch_frac
                  << decision << "\n";
    }

    // --- 4. Compute breakeven element count ---
    std::cout << "\nBreakeven analysis for ReLU (scheduler model):\n";
    std::cout << "  CPU:  0.40 ns/elem\n";
    std::cout << "  GPU:  launch_overhead + (0.40/3.0) ns/elem\n";
    std::cout << "  Transfer: H2D + D2H at 6 bytes/ns\n\n";

    // With transfers, breakeven is already handled by transfer cost.
    // The question is: when inputs ARE already on GPU (residency=true),
    // does launch overhead push the breakeven higher?
    std::cout << "  With residency (no H2D), breakeven element count:\n";
    std::cout << "    Without launch overhead model: ";
    // cpu_cost = gpu_cost (no h2d)
    // 0.40*N = launch + (0.40/3)*N + (N*4)/6
    // Kernel cost only: 0.40*N = launch + 0.133*N
    // 0.267*N = launch => N = launch / 0.267
    double launch_ns = null_us * 1000.0;  // convert us back to ns
    double breakeven_no_launch = 0;       // no threshold needed if launch not modeled
    double breakeven_with_launch = launch_ns / (0.40 - (0.40/3.0));
    std::cout << "~0 (GPU always wins when transfer is free)\n";
    std::cout << "    With launch overhead (" << std::setprecision(0) << launch_ns
              << " ns): N >= " << static_cast<int>(breakeven_with_launch) << " elems\n";
    std::cout << "    At " << static_cast<int>(breakeven_with_launch) << " elems: "
              << static_cast<int>(breakeven_with_launch * 4) << " bytes ("
              << std::setprecision(1)
              << (breakeven_with_launch * 4 / 1024.0) << " KB)\n";

    CUDA_CHECK(cudaFree(dummy));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    std::cout << "\nConclusion:\n";
    std::cout << "  Current scheduler omits launch overhead.\n";
    std::cout << "  For residency=true cases, this causes incorrect GPU decisions\n";
    std::cout << "  for tensors below the breakeven element count above.\n";
    std::cout << "  Fix: add CUDA_LAUNCH_OVERHEAD_NS constant to kernel_ns.\n";

    return 0;
}

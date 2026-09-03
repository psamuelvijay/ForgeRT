/*
 * relu_kernel.cu  –  CUDA ReLU kernel for ForgeRT (Phase 3)
 *
 * Target architecture: Pascal sm_61 (GTX 1050)
 * Strategy: grid-stride loop, 256 threads/block
 *
 * The launcher returns a plain int (0 = success) so the host header never
 * needs to include cuda_runtime.h.  cudaError_t is cast at the boundary.
 */

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>

// ------------------------------------------------------------------ kernel ---
/*
 * Each thread handles one (or more) elements via grid-stride loop.
 * Using int for n/idx to avoid __int128 operations on sm_61; max single
 * allocation we'll ever pass is well under INT_MAX for Phase 3 work.
 */
__global__ void relu_kernel(const float* __restrict__ input,
                             float*       __restrict__ output,
                             int                       n) {
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float v = input[i];
        output[i] = (v > 0.0f) ? v : 0.0f;
    }
}

// ----------------------------------------------------------- host launcher ---
/*
 * Returns 0 on success, non-zero (the raw cudaError_t integer) on failure.
 * Declared extern "C" so MSVC/g++ can call it without CUDA headers.
 */
extern "C" int forgert_relu_cuda(const float* input, float* output, size_t n) {
    if (n == 0) return 0;

    if (n > static_cast<size_t>(INT_MAX)) {
        fprintf(stderr, "forgert_relu_cuda: n=%zu exceeds INT_MAX\n", n);
        return static_cast<int>(cudaErrorInvalidValue);
    }

    const int N             = static_cast<int>(n);
    const int threadsPerBlock = 256;                          // good for sm_61
    const int blocks          = (N + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocks, threadsPerBlock>>>(input, output, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "relu_kernel launch error: %s\n", cudaGetErrorString(err));
        return static_cast<int>(err);
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "relu_kernel sync error: %s\n", cudaGetErrorString(err));
        return static_cast<int>(err);
    }

    return 0;
}

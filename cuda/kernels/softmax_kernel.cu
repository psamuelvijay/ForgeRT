/*
 * softmax_kernel.cu  –  CUDA Softmax kernel for ForgeRT (Phase 3)
 *
 * Target architecture: Pascal sm_61 (GTX 1050)
 * Strategy: Multi-pass reduction approach using shared memory
 *
 * Computes numerically stable softmax along the last dimension:
 *   softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
 *
 * Algorithm:
 * 1. Find max value in each row (reduction)
 * 2. Compute exp(x - max) and sum (reduction) 
 * 3. Normalize by dividing by sum
 *
 * Pascal sm_61 specifications:
 * - 48 KB shared memory per block
 * - 1024 threads per block max
 * - 32-wide warps
 * - No tensor cores
 */

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>
#include <cmath>
#include <cfloat>

// Block size optimized for sm_61
#define BLOCK_SIZE 256

// ------------------------------------------------------------------ helpers --

// Warp-level reduction for Pascal (no __shfl_down_sync yet)
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down(val, offset));
    }
    return val;
}

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Block-level reduction using shared memory
__device__ float blockReduceMax(float val) {
    static __shared__ float shared[32]; // One per warp
    int lane = threadIdx.x % 32;
    int warpid = threadIdx.x / 32;
    
    val = warpReduceMax(val);
    
    if (lane == 0) shared[warpid] = val;
    __syncthreads();
    
    if (threadIdx.x < blockDim.x / 32) {
        val = shared[lane];
    } else {
        val = -FLT_MAX; // Use FLT_MAX instead of INFINITY
    }
    
    if (warpid == 0) val = warpReduceMax(val);
    return val;
}

__device__ float blockReduceSum(float val) {
    static __shared__ float shared[32]; // One per warp
    int lane = threadIdx.x % 32;
    int warpid = threadIdx.x / 32;
    
    val = warpReduceSum(val);
    
    if (lane == 0) shared[warpid] = val;
    __syncthreads();
    
    if (threadIdx.x < blockDim.x / 32) {
        val = shared[lane];
    } else {
        val = 0.0f;
    }
    
    if (warpid == 0) val = warpReduceSum(val);
    return val;
}

// ------------------------------------------------------------------ kernels --

/*
 * Kernel 1: Find max value in each row
 * Each block processes one row, result written to max_vals[row_idx]
 */
__global__ void softmax_find_max_kernel(const float* __restrict__ input,
                                         float* __restrict__ max_vals,
                                         int outer_size, int inner_size) {
    int row = blockIdx.x;
    if (row >= outer_size) return;
    
    int tid = threadIdx.x;
    const float* row_input = input + row * inner_size;
    
    // Each thread finds max of its assigned elements
    float thread_max = -FLT_MAX;
    for (int i = tid; i < inner_size; i += blockDim.x) {
        thread_max = fmaxf(thread_max, row_input[i]);
    }
    
    // Block-level reduction to find row max
    thread_max = blockReduceMax(thread_max);
    
    // Thread 0 writes result
    if (tid == 0) {
        max_vals[row] = thread_max;
    }
}

/*
 * Kernel 2: Compute exp(x - max) and find sum
 * Each block processes one row, reads max from max_vals, 
 * computes exp values in-place, and writes sum to sum_vals
 */
__global__ void softmax_exp_sum_kernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        const float* __restrict__ max_vals,
                                        float* __restrict__ sum_vals,
                                        int outer_size, int inner_size) {
    int row = blockIdx.x;
    if (row >= outer_size) return;
    
    int tid = threadIdx.x;
    const float* row_input = input + row * inner_size;
    float* row_output = output + row * inner_size;
    float row_max = max_vals[row];
    
    // Each thread computes exp(x - max) for its assigned elements and accumulates sum
    float thread_sum = 0.0f;
    for (int i = tid; i < inner_size; i += blockDim.x) {
        float exp_val = expf(row_input[i] - row_max);
        row_output[i] = exp_val;
        thread_sum += exp_val;
    }
    
    // Block-level reduction to find row sum
    thread_sum = blockReduceSum(thread_sum);
    
    // Thread 0 writes result
    if (tid == 0) {
        sum_vals[row] = thread_sum;
    }
}

/*
 * Kernel 3: Normalize by dividing by sum
 * Each block processes one row, reads sum from sum_vals, normalizes output
 */
__global__ void softmax_normalize_kernel(float* __restrict__ output,
                                          const float* __restrict__ sum_vals,
                                          int outer_size, int inner_size) {
    int row = blockIdx.x;
    if (row >= outer_size) return;
    
    int tid = threadIdx.x;
    float* row_output = output + row * inner_size;
    float row_sum = sum_vals[row];
    
    // Avoid division by zero (though should be very rare with exp)
    if (row_sum <= 0.0f) {
        // Fallback to uniform distribution
        float uniform = 1.0f / static_cast<float>(inner_size);
        for (int i = tid; i < inner_size; i += blockDim.x) {
            row_output[i] = uniform;
        }
        return;
    }
    
    // Normalize each element
    for (int i = tid; i < inner_size; i += blockDim.x) {
        row_output[i] /= row_sum;
    }
}

// ----------------------------------------------------------- host launcher ---

/*
 * Returns 0 on success, non-zero (the raw cudaError_t integer) on failure.
 * Declared extern "C" so MSVC/g++ can call it without CUDA headers.
 */
extern "C" int forgert_softmax_cuda(const float* input, float* output,
                                    size_t outer_size, size_t inner_size) {
    
    if (outer_size == 0 || inner_size == 0) return 0;
    
    if (outer_size > static_cast<size_t>(INT_MAX) || 
        inner_size > static_cast<size_t>(INT_MAX)) {
        fprintf(stderr, "forgert_softmax_cuda: dimensions exceed INT_MAX\n");
        return static_cast<int>(cudaErrorInvalidValue);
    }
    
    const int outer = static_cast<int>(outer_size);
    const int inner = static_cast<int>(inner_size);
    
    // Allocate temporary storage for max and sum values
    float *d_max_vals, *d_sum_vals;
    cudaError_t err = cudaMalloc(&d_max_vals, outer_size * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "softmax_cuda: failed to allocate max_vals\n");
        return static_cast<int>(err);
    }
    
    err = cudaMalloc(&d_sum_vals, outer_size * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "softmax_cuda: failed to allocate sum_vals\n");
        cudaFree(d_max_vals);
        return static_cast<int>(err);
    }
    
    // Launch configuration: one block per row
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize(outer);
    
    // Phase 1: Find max in each row
    softmax_find_max_kernel<<<gridSize, blockSize>>>(input, d_max_vals, outer, inner);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "softmax_find_max_kernel launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_max_vals);
        cudaFree(d_sum_vals);
        return static_cast<int>(err);
    }
    
    // Phase 2: Compute exp(x - max) and sum
    softmax_exp_sum_kernel<<<gridSize, blockSize>>>(input, output, d_max_vals, d_sum_vals, outer, inner);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "softmax_exp_sum_kernel launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_max_vals);
        cudaFree(d_sum_vals);
        return static_cast<int>(err);
    }
    
    // Phase 3: Normalize
    softmax_normalize_kernel<<<gridSize, blockSize>>>(output, d_sum_vals, outer, inner);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "softmax_normalize_kernel launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_max_vals);
        cudaFree(d_sum_vals);
        return static_cast<int>(err);
    }
    
    // Wait for completion
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "softmax_cuda sync error: %s\n", cudaGetErrorString(err));
        cudaFree(d_max_vals);
        cudaFree(d_sum_vals);
        return static_cast<int>(err);
    }
    
    // Cleanup temporary storage
    cudaFree(d_max_vals);
    cudaFree(d_sum_vals);
    
    return 0;
}
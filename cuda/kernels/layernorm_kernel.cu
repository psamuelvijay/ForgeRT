#include <cuda_runtime.h>
#include <cmath>

/*
 * CUDA LayerNorm implementation with dual-reduction approach.
 * 
 * Algorithm: output = (input - mean) / sqrt(variance + epsilon)
 * 
 * Implementation:
 * 1. Block-parallel reduction to compute mean per row
 * 2. Block-parallel reduction to compute variance per row  
 * 3. Element-wise normalization using computed statistics
 * 
 * Each thread block processes one row (inner_size elements).
 * Uses shared memory for efficient block-level reductions.
 * Optimized for Pascal sm_61 architecture.
 */

constexpr int BLOCK_SIZE = 256;

// Block-level sum reduction using shared memory
__device__ float block_reduce_sum(float val) {
    __shared__ float shared[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();
    
    // Tree reduction in shared memory
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    
    return shared[0];
}

// LayerNorm kernel: each block processes one row
__global__ void layernorm_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                size_t outer_size,
                                size_t inner_size,
                                float epsilon) {
    
    size_t row = blockIdx.x;
    if (row >= outer_size) return;
    
    const float* row_input = input + row * inner_size;
    float* row_output = output + row * inner_size;
    
    float inv_inner_size = 1.0f / static_cast<float>(inner_size);
    
    // Pass 1: Compute mean
    float sum = 0.0f;
    for (size_t i = threadIdx.x; i < inner_size; i += blockDim.x) {
        sum += row_input[i];
    }
    
    sum = block_reduce_sum(sum);
    
    // Broadcast mean to all threads
    __shared__ float mean_shared;
    if (threadIdx.x == 0) {
        mean_shared = sum * inv_inner_size;
    }
    __syncthreads();
    float mean = mean_shared;
    
    // Pass 2: Compute variance
    float var_sum = 0.0f;
    for (size_t i = threadIdx.x; i < inner_size; i += blockDim.x) {
        float diff = row_input[i] - mean;
        var_sum += diff * diff;
    }
    
    var_sum = block_reduce_sum(var_sum);
    
    // Broadcast normalization factor to all threads
    __shared__ float inv_std_shared;
    if (threadIdx.x == 0) {
        float variance = var_sum * inv_inner_size;
        inv_std_shared = 1.0f / sqrtf(variance + epsilon);
    }
    __syncthreads();
    float inv_std = inv_std_shared;
    
    // Pass 3: Normalize
    for (size_t i = threadIdx.x; i < inner_size; i += blockDim.x) {
        row_output[i] = (row_input[i] - mean) * inv_std;
    }
}

extern "C" int forgert_layernorm_cuda(const float* input, float* output,
                                      size_t outer_size, size_t inner_size,
                                      float epsilon) {
    
    if (!input || !output) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    
    if (outer_size == 0 || inner_size == 0) {
        return static_cast<int>(cudaSuccess);  // Nothing to do
    }
    
    // Launch one block per row, each with BLOCK_SIZE threads
    dim3 grid_size(static_cast<unsigned int>(outer_size));
    dim3 block_size(BLOCK_SIZE);
    
    layernorm_kernel<<<grid_size, block_size>>>(input, output, outer_size, inner_size, epsilon);
    
    // Check for kernel launch errors
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        return static_cast<int>(launch_err);
    }
    
    // Synchronize to catch execution errors
    cudaError_t sync_err = cudaDeviceSynchronize();
    return static_cast<int>(sync_err);
}
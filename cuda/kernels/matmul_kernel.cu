/*
 * matmul_kernel.cu  –  CUDA matrix multiplication kernel for ForgeRT (Phase 3)
 *
 * Target architecture: Pascal sm_61 (GTX 1050)
 * Strategy: Tiled matrix multiplication using shared memory
 *
 * Computes C = A @ B where A[M,K], B[K,N] -> C[M,N]
 * All matrices in row-major layout.
 *
 * Pascal sm_61 specifications:
 * - 48 KB shared memory per block
 * - 1024 threads per block max
 * - 32-wide warps
 * - No tensor cores (pre-Volta)
 */

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>

// Tile size optimized for sm_61: 16x16 = 256 threads, fits well in shared memory
#define TILE_SIZE 16

// ------------------------------------------------------------------ kernel ---
/*
 * Tiled matrix multiplication kernel.
 * Each block computes a TILE_SIZE x TILE_SIZE submatrix of C.
 * Uses shared memory to reduce global memory accesses.
 */
__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float*       __restrict__ C,
                              int M, int K, int N) {
    
    // Shared memory tiles for A and B
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    // Thread and block indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Output coordinates
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles of A and B
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tile of A into shared memory
        int a_row = row;
        int a_col = tile * TILE_SIZE + tx;
        if (a_row < M && a_col < K) {
            tile_A[ty][tx] = A[a_row * K + a_col];
        } else {
            tile_A[ty][tx] = 0.0f;
        }
        
        // Load tile of B into shared memory
        int b_row = tile * TILE_SIZE + ty;
        int b_col = col;
        if (b_row < K && b_col < N) {
            tile_B[ty][tx] = B[b_row * N + b_col];
        } else {
            tile_B[ty][tx] = 0.0f;
        }
        
        // Synchronize to ensure tiles are loaded
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_A[ty][k] * tile_B[k][tx];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ----------------------------------------------------------- host launcher ---
/*
 * Returns 0 on success, non-zero (the raw cudaError_t integer) on failure.
 * Declared extern "C" so MSVC/g++ can call it without CUDA headers.
 */
extern "C" int forgert_matmul_cuda(const float* A, const float* B, float* C,
                                   size_t M, size_t K, size_t N) {
    
    // Validate inputs
    if (M == 0 || K == 0 || N == 0) return 0;
    
    if (M > static_cast<size_t>(INT_MAX) || 
        K > static_cast<size_t>(INT_MAX) || 
        N > static_cast<size_t>(INT_MAX)) {
        fprintf(stderr, "forgert_matmul_cuda: dimensions exceed INT_MAX\n");
        return static_cast<int>(cudaErrorInvalidValue);
    }
    
    const int m = static_cast<int>(M);
    const int k = static_cast<int>(K);
    const int n = static_cast<int>(N);
    
    // Configure grid and block dimensions
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((n + TILE_SIZE - 1) / TILE_SIZE, 
                  (m + TILE_SIZE - 1) / TILE_SIZE);
    
    // Launch kernel
    matmul_kernel<<<gridSize, blockSize>>>(A, B, C, m, k, n);
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "matmul_kernel launch error: %s\n", cudaGetErrorString(err));
        return static_cast<int>(err);
    }
    
    // Wait for completion
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "matmul_kernel sync error: %s\n", cudaGetErrorString(err));
        return static_cast<int>(err);
    }
    
    return 0;
}
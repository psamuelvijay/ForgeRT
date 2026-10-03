/*
 * add_kernel.cu  –  CUDA Add kernel for ForgeRT (Phase 3)
 *
 * Target architecture: Pascal sm_61 (GTX 1050)
 *
 * Computes C = A + B with NumPy-style broadcasting.
 *
 * Two dispatch paths:
 *
 *   1. Fast path  – A, B, C all have the same element count.
 *      Single kernel; one thread per element; no index arithmetic.
 *
 *   2. Broadcast path – A and/or B have size-1 dimensions that are
 *      broadcast to match C.  Each thread computes the flat output index,
 *      decomposes it into per-dimension indices, clamps broadcast dims to
 *      0, then re-folds to flat A/B indices.
 *
 * Shape metadata is small (≤ FORGERT_ADD_MAX_DIMS ints per tensor) so it is
 * passed directly as kernel parameters.  No global-memory round-trips for
 * shape data.
 *
 * Pascal sm_61 details:
 *   - 48 KB shared memory (unused here – Add is memory-bound)
 *   - Grid-stride loop keeps all SMs busy for large tensors
 *   - 256 threads/block is a good occupancy target for sm_61
 */

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>

#define BLOCK_SIZE      256
#define MAX_DIMS        8   // max tensor rank we support

// ------------------------------------------------------------------
// Fast path: element-wise add, no broadcasting
// ------------------------------------------------------------------
__global__ void add_elementwise_kernel(const float* __restrict__ a,
                                        const float* __restrict__ b,
                                        float* __restrict__ c,
                                        size_t n)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += gridDim.x * blockDim.x)
    {
        c[i] = a[i] + b[i];
    }
}

// ------------------------------------------------------------------
// Broadcast path
//
// All three tensors are described after left-padding their shapes with
// 1s to the same rank (ndim).  Passed as flat arrays of length ndim.
// ------------------------------------------------------------------
__global__ void add_broadcast_kernel(const float* __restrict__ a,
                                      const float* __restrict__ b,
                                      float* __restrict__ c,
                                      // shape arrays – passed by value into registers
                                      size_t a_dims_0,  size_t a_dims_1,
                                      size_t a_dims_2,  size_t a_dims_3,
                                      size_t a_dims_4,  size_t a_dims_5,
                                      size_t a_dims_6,  size_t a_dims_7,
                                      size_t b_dims_0,  size_t b_dims_1,
                                      size_t b_dims_2,  size_t b_dims_3,
                                      size_t b_dims_4,  size_t b_dims_5,
                                      size_t b_dims_6,  size_t b_dims_7,
                                      size_t c_dims_0,  size_t c_dims_1,
                                      size_t c_dims_2,  size_t c_dims_3,
                                      size_t c_dims_4,  size_t c_dims_5,
                                      size_t c_dims_6,  size_t c_dims_7,
                                      int    ndim,
                                      size_t n)
{
    // Rebuild local arrays from scalar params (avoids pointer in params)
    size_t a_dims[MAX_DIMS] = {a_dims_0, a_dims_1, a_dims_2, a_dims_3,
                                a_dims_4, a_dims_5, a_dims_6, a_dims_7};
    size_t b_dims[MAX_DIMS] = {b_dims_0, b_dims_1, b_dims_2, b_dims_3,
                                b_dims_4, b_dims_5, b_dims_6, b_dims_7};
    size_t c_dims[MAX_DIMS] = {c_dims_0, c_dims_1, c_dims_2, c_dims_3,
                                c_dims_4, c_dims_5, c_dims_6, c_dims_7};

    for (size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
         out_idx < n;
         out_idx += gridDim.x * blockDim.x)
    {
        // Decompose flat output index into per-dim indices
        size_t tmp = out_idx;
        size_t idx[MAX_DIMS];
        for (int d = ndim - 1; d >= 0; --d) {
            idx[d] = tmp % c_dims[d];
            tmp    /= c_dims[d];
        }

        // Fold to A's flat index (broadcast dims clamped to 0)
        size_t a_flat = 0, a_stride = 1;
        for (int d = ndim - 1; d >= 0; --d) {
            size_t i_a = (a_dims[d] == 1) ? 0 : idx[d];
            a_flat   += i_a * a_stride;
            a_stride *= a_dims[d];
        }

        // Fold to B's flat index
        size_t b_flat = 0, b_stride = 1;
        for (int d = ndim - 1; d >= 0; --d) {
            size_t i_b = (b_dims[d] == 1) ? 0 : idx[d];
            b_flat   += i_b * b_stride;
            b_stride *= b_dims[d];
        }

        c[out_idx] = a[a_flat] + b[b_flat];
    }
}

// ------------------------------------------------------------------
// Host launcher
// ------------------------------------------------------------------
extern "C" int forgert_add_cuda(const float* a, const float* b, float* c,
                                 const size_t* a_dims, const size_t* b_dims,
                                 const size_t* c_dims, int ndim,
                                 size_t c_size)
{
    if (c_size == 0) return 0;

    if (ndim <= 0 || ndim > MAX_DIMS) {
        fprintf(stderr, "forgert_add_cuda: ndim=%d out of range [1,%d]\n",
                ndim, MAX_DIMS);
        return static_cast<int>(cudaErrorInvalidValue);
    }

    // Determine whether all tensors have the same element count (fast path)
    size_t a_size = 1, b_size = 1;
    for (int d = 0; d < ndim; ++d) {
        a_size *= a_dims[d];
        b_size *= b_dims[d];
    }

    // Grid-stride: cap grid so we don't exceed 65535 blocks
    const size_t max_blocks = 65535;
    size_t needed = (c_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 block(BLOCK_SIZE);
    dim3 grid(static_cast<unsigned int>(needed < max_blocks ? needed : max_blocks));

    cudaError_t err;

    if (a_size == c_size && b_size == c_size) {
        // ---- Fast path ------------------------------------------------
        add_elementwise_kernel<<<grid, block>>>(a, b, c, c_size);
    } else {
        // ---- Broadcast path -------------------------------------------
        // Left-pad dim arrays with 1s to MAX_DIMS for the kernel
        size_t ad[MAX_DIMS] = {1,1,1,1,1,1,1,1};
        size_t bd[MAX_DIMS] = {1,1,1,1,1,1,1,1};
        size_t cd[MAX_DIMS] = {1,1,1,1,1,1,1,1};
        int offset = MAX_DIMS - ndim;
        for (int d = 0; d < ndim; ++d) {
            ad[offset + d] = a_dims[d];
            bd[offset + d] = b_dims[d];
            cd[offset + d] = c_dims[d];
        }
        add_broadcast_kernel<<<grid, block>>>(
            a, b, c,
            ad[0], ad[1], ad[2], ad[3], ad[4], ad[5], ad[6], ad[7],
            bd[0], bd[1], bd[2], bd[3], bd[4], bd[5], bd[6], bd[7],
            cd[0], cd[1], cd[2], cd[3], cd[4], cd[5], cd[6], cd[7],
            MAX_DIMS,   // always pass full 8-dim arrays to the kernel
            c_size);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "forgert_add_cuda kernel launch error: %s\n",
                cudaGetErrorString(err));
        return static_cast<int>(err);
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "forgert_add_cuda sync error: %s\n",
                cudaGetErrorString(err));
        return static_cast<int>(err);
    }

    return 0;
}

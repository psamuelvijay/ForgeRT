/*
 * conv2d_kernel.cu  --  CUDA Conv2D for ForgeRT (Phase 8)
 *
 * Strategy: im2col (device) + tiled SGEMM.
 *
 * im2col:  Reshape the input patch windows into rows of a matrix
 *          col_matrix [C_in*kH*kW, H_out*W_out*N].
 *          This is a standard approach that reuses the already-verified
 *          tiled MatMul path for the compute-intensive GEMM.
 *
 * GEMM:    weight [C_out, C_in*kH*kW]  x  col [C_in*kH*kW, P]  = out [C_out, P]
 *          where P = N * H_out * W_out.
 *
 * Bias:    Added with a separate element-wise kernel after GEMM.
 *
 * Why not cuDNN?  ForgeRT has no external dependencies; this kernel is
 * sufficient for SqueezeNet 1.1 correctness validation.
 *
 * Target: Pascal sm_61 (GTX 1050, 4 GB VRAM)
 * Built with: CUDA 12.6, MSVC 14.39, -O3 -use_fast_math
 */

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>

// Tile size for GEMM — 16 gives good occupancy on sm_61 (2048 threads/SM,
// 64 KB shared memory).  Each block uses 2 * 16*16 * 4 = 2 KB smem.
static constexpr int TILE = 16;

// ---------------------------------------------------------------------------
// Kernel 1: im2col
// ---------------------------------------------------------------------------
// Writes a col_matrix of shape [K_cols, P] where
//   K_cols = C_in * kH * kW
//   P      = N * H_out * W_out
// Each element col[k, p] = input patch value (or 0 for padding).
// ---------------------------------------------------------------------------
__global__ void im2col_kernel(
    const float* __restrict__ input,  // [N, C_in, H_in, W_in]
    float*       __restrict__ col,    // [K_cols, P]
    int N, int C_in, int H_in, int W_in,
    int kH, int kW,
    int stride_h, int stride_w,
    int pad_h,    int pad_w,
    int H_out,    int W_out,
    int K_cols,   int P)
{
    // Each thread writes one cell of col.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K_cols * P) return;

    const int k = idx / P;   // row in col: which patch element
    const int p = idx % P;   // col in col: which output position

    // Decode k -> (ci, kh, kw)
    const int ci = k / (kH * kW);
    const int rem = k % (kH * kW);
    const int kh = rem / kW;
    const int kw = rem % kW;

    // Decode p -> (n, oh, ow)
    const int n  = p / (H_out * W_out);
    const int sp = p % (H_out * W_out);
    const int oh = sp / W_out;
    const int ow = sp % W_out;

    const int ih = oh * stride_h - pad_h + kh;
    const int iw = ow * stride_w - pad_w + kw;

    float val = 0.0f;
    if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
        val = input[((n * C_in + ci) * H_in + ih) * W_in + iw];
    }
    col[k * P + p] = val;
}

// ---------------------------------------------------------------------------
// Kernel 2: tiled SGEMM  C = A * B
//   A [M, K],  B [K, P],  C [M, P]  -- all row-major
// ---------------------------------------------------------------------------
__global__ void tiled_sgemm(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int K, int P)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        const int aCol = t * TILE + threadIdx.x;
        const int bRow = t * TILE + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < K && col < P) ? B[bRow * P + col] : 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE; ++i) acc += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < P)
        C[row * P + col] = acc;
}

// ---------------------------------------------------------------------------
// Kernel 3: add bias
//   out [C_out, P] += bias [C_out]  broadcast over P
// ---------------------------------------------------------------------------
__global__ void add_bias_kernel(
    float*       __restrict__ out,   // [C_out, P]
    const float* __restrict__ bias,  // [C_out]
    int C_out, int P)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C_out * P) return;
    out[idx] += bias[idx / P];
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
extern "C" int forgert_conv2d_cuda(
    const float* input,   // [N, C_in,  H_in,  W_in]
    const float* weight,  // [C_out, C_in, kH, kW]
    const float* bias,    // [C_out] or nullptr
    float*       output,  // [N, C_out, H_out, W_out]
    int N, int C_in, int H_in, int W_in,
    int C_out, int kH, int kW,
    int stride_h, int stride_w,
    int pad_h,    int pad_w)
{
    const int H_out = (H_in + 2*pad_h - kH) / stride_h + 1;
    const int W_out = (W_in + 2*pad_w - kW) / stride_w + 1;
    const int K_cols = C_in * kH * kW;  // GEMM inner dimension
    const int P      = N * H_out * W_out;  // GEMM columns (output positions)

    // ------ 1. im2col ------
    float* d_col = nullptr;
    cudaError_t err = cudaMalloc(&d_col, (size_t)K_cols * P * sizeof(float));
    if (err != cudaSuccess) return static_cast<int>(err);

    {
        const int total = K_cols * P;
        const int threads = 256;
        const int blocks  = (total + threads - 1) / threads;
        im2col_kernel<<<blocks, threads>>>(
            input, d_col,
            N, C_in, H_in, W_in,
            kH, kW, stride_h, stride_w, pad_h, pad_w,
            H_out, W_out, K_cols, P);
        err = cudaGetLastError();
        if (err != cudaSuccess) { cudaFree(d_col); return static_cast<int>(err); }
    }

    // ------ 2. GEMM: weight [C_out, K_cols] x col [K_cols, P] -> out [C_out, P] ------
    // output is already allocated as [N, C_out, H_out, W_out] which is the same
    // memory as [C_out, P] when N==1 (or we treat the batch dimension as folded
    // into P already).  For N>1 P already encodes n, so we compute one big GEMM.
    {
        dim3 threads2(TILE, TILE);
        dim3 blocks2((P + TILE-1)/TILE, (C_out + TILE-1)/TILE);
        tiled_sgemm<<<blocks2, threads2>>>(weight, d_col, output, C_out, K_cols, P);
        err = cudaGetLastError();
        if (err != cudaSuccess) { cudaFree(d_col); return static_cast<int>(err); }
    }

    // ------ 3. Bias ------
    if (bias) {
        const int total = C_out * P;
        const int threads = 256;
        const int blocks  = (total + threads - 1) / threads;
        add_bias_kernel<<<blocks, threads>>>(output, bias, C_out, P);
        err = cudaGetLastError();
        if (err != cudaSuccess) { cudaFree(d_col); return static_cast<int>(err); }
    }

    err = cudaDeviceSynchronize();
    cudaFree(d_col);
    return static_cast<int>(err);
}

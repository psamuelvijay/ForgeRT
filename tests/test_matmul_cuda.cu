/*
 * test_matmul_cuda.cu  –  CUDA MatMul kernel and operator tests (Phase 3)
 *
 * Tests both the standalone CUDA kernel and the MatMulOp CUDA dispatch.
 * Verifies numerical correctness against CPU reference implementation.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <vector>
#include <memory>

#include "forgert/tensor/tensor.h"
#include "forgert/operator/matmul.h"
#include "kernels/matmul_kernel.h"

using namespace forgert;

// ------------------------------------------------------------------ helpers --

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                 \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

static const float kTol = 1e-4f;  // Slightly relaxed for accumulated errors in MatMul
static int g_pass = 0;
static int g_fail = 0;

static void pass(const char* msg) { ++g_pass; printf("  \xE2\x9C\x93 %s\n", msg); }
static void fail(const char* msg) { ++g_fail; printf("  FAIL: %s\n", msg); }

// CPU reference matrix multiplication
static void cpu_matmul_ref(const float* A, const float* B, float* C,
                          size_t M, size_t K, size_t N) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Initialize matrix with deterministic values
static void init_matrix(float* mat, size_t rows, size_t cols, float scale = 1.0f) {
    for (size_t i = 0; i < rows * cols; ++i) {
        mat[i] = scale * (static_cast<float>(i % 17) - 8.0f) / 8.0f;
    }
}

// Check matrices are equal within tolerance
static bool matrices_equal(const float* A, const float* B, size_t M, size_t N, float tol = kTol) {
    for (size_t i = 0; i < M * N; ++i) {
        if (fabsf(A[i] - B[i]) > tol) {
            size_t row = i / N;
            size_t col = i % N;
            fprintf(stderr, "  FAIL at [%zu,%zu]: expected %.6f got %.6f (diff=%.6f)\n",
                    row, col, A[i], B[i], fabsf(A[i] - B[i]));
            return false;
        }
    }
    return true;
}

// -------------------------------------------------------------------- tests --

/*
 * Test 1: Small matrix multiplication - kernel correctness
 * A[2,3] @ B[3,2] = C[2,2]
 */
static void test_kernel_small() {
    printf("  Test 1: CUDA kernel small matrix (2x3 @ 3x2 = 2x2)\n");
    
    const size_t M = 2, K = 3, N = 2;
    
    // Host matrices
    std::vector<float> h_A(M * K), h_B(K * N), h_C_expected(M * N), h_C_cuda(M * N);
    
    // Initialize inputs
    init_matrix(h_A.data(), M, K);
    init_matrix(h_B.data(), K, N, 0.7f);
    
    // Compute CPU reference
    cpu_matmul_ref(h_A.data(), h_B.data(), h_C_expected.data(), M, K, N);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel
    int err = forgert_matmul_cuda(d_A, d_B, d_C, M, K, N);
    if (err != 0) {
        fprintf(stderr, "Kernel launch failed: %d\n", err);
        fail("CUDA kernel small matrix");
        return;
    }
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C_cuda.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify
    if (matrices_equal(h_C_expected.data(), h_C_cuda.data(), M, N)) {
        pass("CUDA kernel small matrix (2x3 @ 3x2 = 2x2)");
    } else {
        fail("CUDA kernel small matrix");
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

/*
 * Test 2: Medium matrix multiplication - kernel performance
 * A[64,32] @ B[32,64] = C[64,64]
 */
static void test_kernel_medium() {
    printf("  Test 2: CUDA kernel medium matrix (64x32 @ 32x64 = 64x64)\n");
    
    const size_t M = 64, K = 32, N = 64;
    
    std::vector<float> h_A(M * K), h_B(K * N), h_C_expected(M * N), h_C_cuda(M * N);
    
    init_matrix(h_A.data(), M, K, 0.5f);
    init_matrix(h_B.data(), K, N, 0.3f);
    cpu_matmul_ref(h_A.data(), h_B.data(), h_C_expected.data(), M, K, N);
    
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    int err = forgert_matmul_cuda(d_A, d_B, d_C, M, K, N);
    if (err != 0) {
        fail("CUDA kernel medium matrix - launch failed");
        return;
    }
    
    CUDA_CHECK(cudaMemcpy(h_C_cuda.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    if (matrices_equal(h_C_expected.data(), h_C_cuda.data(), M, N)) {
        pass("CUDA kernel medium matrix (64x32 @ 32x64 = 64x64)");
    } else {
        fail("CUDA kernel medium matrix");
    }
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

/*
 * Test 3: MatMulOp CUDA dispatch - operator integration
 * Tests the full MatMulOp::executeCUDA() path
 */
static void test_operator_cuda_dispatch() {
    printf("  Test 3: MatMulOp CUDA dispatch (16x8 @ 8x16 = 16x16)\n");
    
    const size_t M = 16, K = 8, N = 16;
    
    // Create input tensors on CPU
    std::vector<float> h_A(M * K), h_B(K * N);
    init_matrix(h_A.data(), M, K, 1.2f);
    init_matrix(h_B.data(), K, N, 0.8f);
    
    Tensor cpu_A(TensorShape({M, K}), DataType::Float32, Device::CPU);
    Tensor cpu_B(TensorShape({K, N}), DataType::Float32, Device::CPU);
    std::memcpy(cpu_A.data(), h_A.data(), M * K * sizeof(float));
    std::memcpy(cpu_B.data(), h_B.data(), K * N * sizeof(float));
    
    // Upload to CUDA
    Tensor cuda_A(TensorShape({M, K}), DataType::Float32, Device::CUDA);
    Tensor cuda_B(TensorShape({K, N}), DataType::Float32, Device::CUDA);
    Tensor cuda_C(TensorShape({M, N}), DataType::Float32, Device::CUDA);
    
    cpu_A.copyTo(cuda_A);
    cpu_B.copyTo(cuda_B);
    
    // Execute via operator
    MatMulOp op;
    assert(op.supportsBackend(Backend::CUDA));
    
    try {
        op.execute(Backend::CUDA, {&cuda_A, &cuda_B}, {&cuda_C});
    } catch (const std::exception& e) {
        fprintf(stderr, "MatMulOp::executeCUDA failed: %s\n", e.what());
        fail("MatMulOp CUDA dispatch");
        return;
    }
    
    // Download result and compare with CPU reference
    Tensor cpu_C(TensorShape({M, N}), DataType::Float32, Device::CPU);
    cuda_C.copyTo(cpu_C);
    
    std::vector<float> h_C_expected(M * N);
    cpu_matmul_ref(h_A.data(), h_B.data(), h_C_expected.data(), M, K, N);
    
    const float* result = static_cast<const float*>(cpu_C.data());
    if (matrices_equal(h_C_expected.data(), result, M, N)) {
        pass("MatMulOp CUDA dispatch (16x8 @ 8x16 = 16x16)");
    } else {
        fail("MatMulOp CUDA dispatch");
    }
}

/*
 * Test 4: CPU vs CUDA numerical equivalence - large matrix
 */
static void test_cpu_cuda_equivalence() {
    printf("  Test 4: CPU vs CUDA numerical equivalence (32x24 @ 24x32 = 32x32)\n");
    
    const size_t M = 32, K = 24, N = 32;
    
    // Create input tensors
    std::vector<float> h_A(M * K), h_B(K * N);
    init_matrix(h_A.data(), M, K, 1.5f);
    init_matrix(h_B.data(), K, N, 0.6f);
    
    Tensor cpu_A(TensorShape({M, K}), DataType::Float32, Device::CPU);
    Tensor cpu_B(TensorShape({K, N}), DataType::Float32, Device::CPU);
    Tensor cpu_C(TensorShape({M, N}), DataType::Float32, Device::CPU);
    std::memcpy(cpu_A.data(), h_A.data(), M * K * sizeof(float));
    std::memcpy(cpu_B.data(), h_B.data(), K * N * sizeof(float));
    
    // Execute on CPU
    MatMulOp op;
    op.execute(Backend::CPU, {&cpu_A, &cpu_B}, {&cpu_C});
    
    // Execute on CUDA
    Tensor cuda_A(TensorShape({M, K}), DataType::Float32, Device::CUDA);
    Tensor cuda_B(TensorShape({K, N}), DataType::Float32, Device::CUDA);
    Tensor cuda_C(TensorShape({M, N}), DataType::Float32, Device::CUDA);
    
    cpu_A.copyTo(cuda_A);
    cpu_B.copyTo(cuda_B);
    op.execute(Backend::CUDA, {&cuda_A, &cuda_B}, {&cuda_C});
    
    // Compare results
    Tensor cuda_result_cpu(TensorShape({M, N}), DataType::Float32, Device::CPU);
    cuda_C.copyTo(cuda_result_cpu);
    
    const float* cpu_result = static_cast<const float*>(cpu_C.data());
    const float* gpu_result = static_cast<const float*>(cuda_result_cpu.data());
    
    if (matrices_equal(cpu_result, gpu_result, M, N)) {
        pass("CPU vs CUDA numerical equivalence (32x24 @ 24x32 = 32x32)");
    } else {
        fail("CPU vs CUDA numerical equivalence");
    }
}

/*
 * Test 5: Edge cases - identity and zero matrices
 */
static void test_edge_cases() {
    printf("  Test 5: Edge cases (identity, zero matrices)\n");
    
    // Test A @ I = A (identity matrix)
    const size_t N = 4;
    std::vector<float> h_A(N * N), h_I(N * N, 0.0f), h_result(N * N);
    
    // Initialize A with test values
    for (size_t i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(i + 1);
    }
    
    // Create identity matrix
    for (size_t i = 0; i < N; ++i) {
        h_I[i * N + i] = 1.0f;
    }
    
    Tensor cuda_A(TensorShape({N, N}), DataType::Float32, Device::CUDA);
    Tensor cuda_I(TensorShape({N, N}), DataType::Float32, Device::CUDA);
    Tensor cuda_result(TensorShape({N, N}), DataType::Float32, Device::CUDA);
    
    Tensor cpu_A(TensorShape({N, N}), DataType::Float32, Device::CPU);
    Tensor cpu_I(TensorShape({N, N}), DataType::Float32, Device::CPU);
    std::memcpy(cpu_A.data(), h_A.data(), N * N * sizeof(float));
    std::memcpy(cpu_I.data(), h_I.data(), N * N * sizeof(float));
    
    cpu_A.copyTo(cuda_A);
    cpu_I.copyTo(cuda_I);
    
    MatMulOp op;
    op.execute(Backend::CUDA, {&cuda_A, &cuda_I}, {&cuda_result});
    
    Tensor cpu_result(TensorShape({N, N}), DataType::Float32, Device::CPU);
    cuda_result.copyTo(cpu_result);
    
    const float* result_data = static_cast<const float*>(cpu_result.data());
    if (matrices_equal(h_A.data(), result_data, N, N)) {
        pass("Edge cases (A @ I = A)");
    } else {
        fail("Edge cases (identity matrix)");
    }
}

// --------------------------------------------------------------------- main --

int main() {
    printf("========================================\n");
    printf("ForgeRT CUDA MatMul Tests\n");
    printf("========================================\n");
    
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);
    
    test_kernel_small();
    test_kernel_medium();
    test_operator_cuda_dispatch();
    test_cpu_cuda_equivalence();
    test_edge_cases();
    
    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    printf("========================================\n");
    return (g_fail == 0) ? 0 : 1;
}
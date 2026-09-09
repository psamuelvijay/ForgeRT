/*
 * test_softmax_cuda.cu  –  CUDA Softmax kernel and operator tests (Phase 3)
 *
 * Tests both the standalone CUDA kernel and the SoftmaxOp CUDA dispatch.
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
#include "forgert/operator/softmax.h"
#include "kernels/softmax_kernel.h"

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

static const float kTol = 1e-5f;
static int g_pass = 0;
static int g_fail = 0;

static void pass(const char* msg) { ++g_pass; printf("  \xE2\x9C\x93 %s\n", msg); }
static void fail(const char* msg) { ++g_fail; printf("  FAIL: %s\n", msg); }

// CPU reference softmax for 1D
static void cpu_softmax_1d(const float* input, float* output, size_t size) {
    if (size == 0) return;
    
    // Find max
    float max_val = input[0];
    for (size_t i = 1; i < size; ++i) {
        max_val = std::max(max_val, input[i]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    
    // Normalize
    for (size_t i = 0; i < size; ++i) {
        output[i] /= sum;
    }
}

// Check if arrays are equal within tolerance
static bool arrays_equal(const float* a, const float* b, size_t n, float tol = kTol) {
    for (size_t i = 0; i < n; ++i) {
        if (fabsf(a[i] - b[i]) > tol) {
            fprintf(stderr, "  FAIL at i=%zu: expected %.6f got %.6f (diff=%.6f)\n",
                    i, a[i], b[i], fabsf(a[i] - b[i]));
            return false;
        }
    }
    return true;
}

// Check if probability distribution sums to 1
static bool check_probability_sum(const float* probs, size_t n, float tol = kTol) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        if (probs[i] < 0.0f || probs[i] > 1.0f) {
            fprintf(stderr, "  FAIL: probability out of range: %.6f\n", probs[i]);
            return false;
        }
        sum += probs[i];
    }
    if (fabsf(sum - 1.0f) > tol) {
        fprintf(stderr, "  FAIL: probabilities sum to %.6f, expected 1.0\n", sum);
        return false;
    }
    return true;
}

// -------------------------------------------------------------------- tests --

/*
 * Test 1: Simple known distribution [1, 2, 3]
 */
static void test_kernel_simple_1d() {
    printf("  Test 1: CUDA kernel simple 1D softmax [1, 2, 3]\n");
    
    const size_t n = 3;
    std::vector<float> h_input = {1.0f, 2.0f, 3.0f};
    std::vector<float> h_expected(n), h_cuda(n);
    
    // Compute CPU reference
    cpu_softmax_1d(h_input.data(), h_expected.data(), n);
    
    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel (1D case: outer_size=1, inner_size=3)
    int err = forgert_softmax_cuda(d_input, d_output, 1, n);
    if (err != 0) {
        fprintf(stderr, "Kernel launch failed: %d\n", err);
        fail("CUDA kernel simple 1D");
        return;
    }
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_cuda.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify
    if (arrays_equal(h_expected.data(), h_cuda.data(), n) &&
        check_probability_sum(h_cuda.data(), n)) {
        pass("CUDA kernel simple 1D softmax [1, 2, 3]");
    } else {
        fail("CUDA kernel simple 1D");
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

/*
 * Test 2: Large values for numerical stability [1000, 1001, 1002]
 */
static void test_kernel_large_values() {
    printf("  Test 2: CUDA kernel numerical stability [1000, 1001, 1002]\n");
    
    const size_t n = 3;
    std::vector<float> h_input = {1000.0f, 1001.0f, 1002.0f};
    std::vector<float> h_expected(n), h_cuda(n);
    
    cpu_softmax_1d(h_input.data(), h_expected.data(), n);
    
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    
    int err = forgert_softmax_cuda(d_input, d_output, 1, n);
    if (err != 0) {
        fail("CUDA kernel large values - launch failed");
        return;
    }
    
    CUDA_CHECK(cudaMemcpy(h_cuda.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Check for NaN/Inf
    bool has_nan_inf = false;
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(h_cuda[i]) || std::isinf(h_cuda[i])) {
            has_nan_inf = true;
            break;
        }
    }
    
    if (!has_nan_inf && arrays_equal(h_expected.data(), h_cuda.data(), n) &&
        check_probability_sum(h_cuda.data(), n)) {
        pass("CUDA kernel numerical stability [1000, 1001, 1002]");
    } else {
        fail("CUDA kernel large values");
    }
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

/*
 * Test 3: 2D batch processing - multiple rows independently normalized
 */
static void test_kernel_2d_batch() {
    printf("  Test 3: CUDA kernel 2D batch (2x3 matrix)\n");
    
    const size_t outer = 2, inner = 3;
    const size_t total = outer * inner;
    
    // Input: [[1, 2, 3], [4, 5, 6]]
    std::vector<float> h_input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> h_expected(total), h_cuda(total);
    
    // Compute CPU reference for each row
    cpu_softmax_1d(&h_input[0], &h_expected[0], inner); // Row 0
    cpu_softmax_1d(&h_input[3], &h_expected[3], inner); // Row 1
    
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, total * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    
    int err = forgert_softmax_cuda(d_input, d_output, outer, inner);
    if (err != 0) {
        fail("CUDA kernel 2D batch - launch failed");
        return;
    }
    
    CUDA_CHECK(cudaMemcpy(h_cuda.data(), d_output, total * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify each row separately
    bool row0_ok = check_probability_sum(&h_cuda[0], inner);
    bool row1_ok = check_probability_sum(&h_cuda[3], inner);
    bool values_ok = arrays_equal(h_expected.data(), h_cuda.data(), total);
    
    if (row0_ok && row1_ok && values_ok) {
        pass("CUDA kernel 2D batch (2x3 matrix)");
    } else {
        fail("CUDA kernel 2D batch");
    }
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

/*
 * Test 4: Large tensor to exercise reduction
 */
static void test_kernel_large_tensor() {
    printf("  Test 4: CUDA kernel large tensor (10 x 1000)\n");
    
    const size_t outer = 10, inner = 1000;
    const size_t total = outer * inner;
    
    std::vector<float> h_input(total), h_expected(total), h_cuda(total);
    
    // Initialize with deterministic values
    for (size_t i = 0; i < total; ++i) {
        h_input[i] = static_cast<float>((i * 17) % 101) - 50.0f; // Values in [-50, 50]
    }
    
    // Compute CPU reference
    for (size_t row = 0; row < outer; ++row) {
        cpu_softmax_1d(&h_input[row * inner], &h_expected[row * inner], inner);
    }
    
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, total * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    
    int err = forgert_softmax_cuda(d_input, d_output, outer, inner);
    if (err != 0) {
        fail("CUDA kernel large tensor - launch failed");
        return;
    }
    
    CUDA_CHECK(cudaMemcpy(h_cuda.data(), d_output, total * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Check each row sums to 1
    bool sums_ok = true;
    for (size_t row = 0; row < outer; ++row) {
        if (!check_probability_sum(&h_cuda[row * inner], inner)) {
            sums_ok = false;
            break;
        }
    }
    
    // Check values match CPU (with slightly relaxed tolerance for large tensors)
    bool values_ok = arrays_equal(h_expected.data(), h_cuda.data(), total, 2e-5f);
    
    if (sums_ok && values_ok) {
        pass("CUDA kernel large tensor (10 x 1000)");
    } else {
        fail("CUDA kernel large tensor");
    }
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

/*
 * Test 5: SoftmaxOp CUDA dispatch - operator integration
 */
static void test_operator_cuda_dispatch() {
    printf("  Test 5: SoftmaxOp CUDA dispatch (1D case)\n");
    
    const size_t n = 5;
    std::vector<float> h_input = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    
    // Create tensors
    Tensor cpu_input(TensorShape({n}), DataType::Float32, Device::CPU);
    std::memcpy(cpu_input.data(), h_input.data(), n * sizeof(float));
    
    Tensor cuda_input(TensorShape({n}), DataType::Float32, Device::CUDA);
    Tensor cuda_output(TensorShape({n}), DataType::Float32, Device::CUDA);
    
    cpu_input.copyTo(cuda_input);
    
    // Execute via operator
    SoftmaxOp op;
    assert(op.supportsBackend(Backend::CUDA));
    
    try {
        op.execute(Backend::CUDA, {&cuda_input}, {&cuda_output});
    } catch (const std::exception& e) {
        fprintf(stderr, "SoftmaxOp::executeCUDA failed: %s\n", e.what());
        fail("SoftmaxOp CUDA dispatch");
        return;
    }
    
    // Download result and verify
    Tensor cpu_output(TensorShape({n}), DataType::Float32, Device::CPU);
    cuda_output.copyTo(cpu_output);
    
    const float* result = static_cast<const float*>(cpu_output.data());
    if (check_probability_sum(result, n)) {
        pass("SoftmaxOp CUDA dispatch (1D case)");
    } else {
        fail("SoftmaxOp CUDA dispatch");
    }
}

/*
 * Test 6: CPU vs CUDA numerical equivalence
 */
static void test_cpu_cuda_equivalence() {
    printf("  Test 6: CPU vs CUDA numerical equivalence (2D case)\n");
    
    const size_t rows = 8, cols = 16;
    const size_t total = rows * cols;
    
    // Create input tensors
    std::vector<float> h_input(total);
    for (size_t i = 0; i < total; ++i) {
        h_input[i] = static_cast<float>((i * 13) % 37) - 18.0f;
    }
    
    // Execute on CPU
    Tensor cpu_input(TensorShape({rows, cols}), DataType::Float32, Device::CPU);
    Tensor cpu_output(TensorShape({rows, cols}), DataType::Float32, Device::CPU);
    std::memcpy(cpu_input.data(), h_input.data(), total * sizeof(float));
    
    SoftmaxOp op;
    op.execute(Backend::CPU, {&cpu_input}, {&cpu_output});
    
    // Execute on CUDA
    Tensor cuda_input(TensorShape({rows, cols}), DataType::Float32, Device::CUDA);
    Tensor cuda_output(TensorShape({rows, cols}), DataType::Float32, Device::CUDA);
    
    cpu_input.copyTo(cuda_input);
    op.execute(Backend::CUDA, {&cuda_input}, {&cuda_output});
    
    // Compare results
    Tensor cuda_result_cpu(TensorShape({rows, cols}), DataType::Float32, Device::CPU);
    cuda_output.copyTo(cuda_result_cpu);
    
    const float* cpu_result = static_cast<const float*>(cpu_output.data());
    const float* gpu_result = static_cast<const float*>(cuda_result_cpu.data());
    
    if (arrays_equal(cpu_result, gpu_result, total)) {
        pass("CPU vs CUDA numerical equivalence (2D case)");
    } else {
        fail("CPU vs CUDA numerical equivalence");
    }
}

// --------------------------------------------------------------------- main --

int main() {
    printf("========================================\n");
    printf("ForgeRT CUDA Softmax Tests\n");
    printf("========================================\n");
    
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);
    
    test_kernel_simple_1d();
    test_kernel_large_values();
    test_kernel_2d_batch();
    test_kernel_large_tensor();
    test_operator_cuda_dispatch();
    test_cpu_cuda_equivalence();
    
    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    printf("========================================\n");
    return (g_fail == 0) ? 0 : 1;
}
/*
 * test_relu_op_cuda.cu  –  ReLUOp CUDA backend integration tests (Phase 3)
 *
 * Tests that ReLUOp::execute(Backend::CUDA, ...) correctly dispatches through
 * the operator abstraction to the CUDA ReLU kernel.
 *
 * Memory model (Phase 3 / pre-Phase-4):
 *   Tensor does not yet own CUDA device memory (that is Phase 4).
 *   We allocate device memory with cudaMalloc and wrap it in non-owning
 *   Tensor objects (the Tensor(shape, dtype, void*, Device) constructor).
 *   This is the correct architectural boundary — no fake Tensor CUDA
 *   allocation, no skipping the Phase 4 work.
 *
 * Tests:
 *   1. supportsBackend() reports CUDA
 *   2. Small tensor: correctness via ReLUOp::execute(Backend::CUDA)
 *   3. Large tensor (1M): exercises grid-stride loop through the op path
 *   4. CPU vs CUDA numerical equivalence through the operator interface
 *   5. Error path: wrong Device on input tensor throws
 *   6. Error path: wrong Device on output tensor throws
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <string>

#include "forgert/operator/relu.h"    // ReLUOp  (includes operator.h, tensor.h)
#include "forgert/tensor/tensor.h"

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

// Simple pass/fail counter
static int g_pass = 0;
static int g_fail = 0;

static void pass(const char* msg) {
    ++g_pass;
    printf("  \xE2\x9C\x93 %s\n", msg);
}

static void fail(const char* msg, const char* detail = "") {
    ++g_fail;
    printf("  FAIL: %s  %s\n", msg, detail);
}

// Allocate device buffer, upload host data, return device pointer.
// Caller owns the device memory.
static float* upload(const float* host, size_t n) {
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, host, n * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

// Download device buffer to newly-allocated host array.
static float* download(const float* d, size_t n) {
    float* h = new float[n];
    CUDA_CHECK(cudaMemcpy(h, d, n * sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

// -------------------------------------------------------------------- tests --

/*
 * Test 1: supportsBackend()
 */
static void test_supports_backend() {
    printf("  Test 1: supportsBackend()\n");
    forgert::ReLUOp op;

    if (!op.supportsBackend(forgert::Backend::CPU))
        fail("supportsBackend(CPU) should be true");
    else if (!op.supportsBackend(forgert::Backend::CUDA))
        fail("supportsBackend(CUDA) should be true");
    else
        pass("supportsBackend: CPU=true, CUDA=true");
}

/*
 * Test 2: Small tensor correctness via ReLUOp::execute(Backend::CUDA)
 */
static void test_small_tensor() {
    printf("  Test 2: Small tensor (1024 elements) via op dispatch\n");

    const int n = 1024;
    float h_in[n];
    for (int i = 0; i < n; ++i) h_in[i] = static_cast<float>(i) - 512.0f;

    float* d_in  = upload(h_in, n);
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));

    forgert::TensorShape shape({static_cast<size_t>(n)});

    // Non-owning Tensor wrappers around device memory
    forgert::Tensor t_in (shape, forgert::DataType::Float32, d_in,  forgert::Device::CUDA);
    forgert::Tensor t_out(shape, forgert::DataType::Float32, d_out, forgert::Device::CUDA);

    forgert::ReLUOp op;
    op.execute(forgert::Backend::CUDA, {&t_in}, {&t_out});

    float* h_out = download(d_out, n);

    bool ok = true;
    for (int i = 0; i < n; ++i) {
        float expected = (h_in[i] > 0.0f) ? h_in[i] : 0.0f;
        if (fabsf(h_out[i] - expected) > kTol) {
            fprintf(stderr, "  FAIL at i=%d: expected %.6f got %.6f\n",
                    i, expected, h_out[i]);
            ok = false;
            break;
        }
    }

    delete[] h_out;
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    if (ok) pass("Small tensor correctness via op dispatch");
    else    fail("Small tensor correctness via op dispatch");
}

/*
 * Test 3: Large tensor (1M elements) — grid-stride loop
 */
static void test_large_tensor() {
    printf("  Test 3: Large tensor (1M elements, grid-stride) via op dispatch\n");

    const size_t n = 1024u * 1024u;
    float* h_in = new float[n];
    for (size_t i = 0; i < n; ++i)
        h_in[i] = static_cast<float>(i % 2000) - 1000.0f;

    float* d_in  = upload(h_in, n);
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));

    forgert::TensorShape shape({n});
    forgert::Tensor t_in (shape, forgert::DataType::Float32, d_in,  forgert::Device::CUDA);
    forgert::Tensor t_out(shape, forgert::DataType::Float32, d_out, forgert::Device::CUDA);

    forgert::ReLUOp op;
    op.execute(forgert::Backend::CUDA, {&t_in}, {&t_out});

    float* h_out = download(d_out, n);

    // Spot-check every 4096th element
    bool ok = true;
    for (size_t i = 0; i < n; i += 4096) {
        float expected = (h_in[i] > 0.0f) ? h_in[i] : 0.0f;
        if (fabsf(h_out[i] - expected) > kTol) {
            ok = false;
            break;
        }
    }

    delete[] h_in;
    delete[] h_out;
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    if (ok) pass("Large tensor (grid-stride) via op dispatch");
    else    fail("Large tensor (grid-stride) via op dispatch");
}

/*
 * Test 4: CPU vs CUDA numerical equivalence through operator interface
 */
static void test_cpu_vs_cuda_equivalence() {
    printf("  Test 4: CPU vs CUDA equivalence through operator interface\n");

    const int n = 10000;
    float h_in[n];
    for (int i = 0; i < n; ++i)
        h_in[i] = static_cast<float>((i * 17) % 2001) - 1000.0f;

    // CPU reference via ReLUOp::execute(Backend::CPU)
    float h_cpu_out[n];
    {
        forgert::TensorShape shape({static_cast<size_t>(n)});
        forgert::Tensor t_in (shape, forgert::DataType::Float32, h_in,     forgert::Device::CPU);
        forgert::Tensor t_out(shape, forgert::DataType::Float32, h_cpu_out, forgert::Device::CPU);
        forgert::ReLUOp op;
        op.execute(forgert::Backend::CPU, {&t_in}, {&t_out});
    }

    // CUDA via ReLUOp::execute(Backend::CUDA)
    float* d_in  = upload(h_in, n);
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(n) * sizeof(float)));

    forgert::TensorShape shape({static_cast<size_t>(n)});
    forgert::Tensor t_in (shape, forgert::DataType::Float32, d_in,  forgert::Device::CUDA);
    forgert::Tensor t_out(shape, forgert::DataType::Float32, d_out, forgert::Device::CUDA);
    forgert::ReLUOp op;
    op.execute(forgert::Backend::CUDA, {&t_in}, {&t_out});

    float* h_cuda_out = download(d_out, n);

    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (fabsf(h_cuda_out[i] - h_cpu_out[i]) > kTol) {
            fprintf(stderr, "  FAIL at i=%d: CPU=%.6f CUDA=%.6f\n",
                    i, h_cpu_out[i], h_cuda_out[i]);
            ok = false;
            break;
        }
    }

    delete[] h_cuda_out;
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    if (ok) pass("CPU vs CUDA equivalence through operator interface");
    else    fail("CPU vs CUDA equivalence through operator interface");
}

/*
 * Test 5: Error path — CPU-device input tensor rejected for CUDA backend
 */
static void test_error_wrong_input_device() {
    printf("  Test 5: Error path — CPU tensor rejected by CUDA backend\n");

    const int n = 4;
    float h_buf[n] = {1.0f, -1.0f, 2.0f, -2.0f};
    float d_buf_raw[n] = {};  // placeholder; won't actually be used

    forgert::TensorShape shape({static_cast<size_t>(n)});
    // input on CPU, output on CUDA — should throw
    forgert::Tensor t_in (shape, forgert::DataType::Float32, h_buf,   forgert::Device::CPU);
    forgert::Tensor t_out(shape, forgert::DataType::Float32, d_buf_raw, forgert::Device::CUDA);

    forgert::ReLUOp op;
    bool threw = false;
    try {
        op.execute(forgert::Backend::CUDA, {&t_in}, {&t_out});
    } catch (const std::invalid_argument&) {
        threw = true;
    } catch (const std::runtime_error&) {
        threw = true;
    }

    if (threw) pass("CPU-device input rejected by CUDA backend");
    else        fail("CPU-device input NOT rejected — should have thrown");
}

/*
 * Test 6: Error path — CPU-device output tensor rejected for CUDA backend
 */
static void test_error_wrong_output_device() {
    printf("  Test 6: Error path — CPU output tensor rejected by CUDA backend\n");

    const int n = 4;
    float h_buf[n]    = {1.0f, -1.0f, 2.0f, -2.0f};
    float h_out_buf[n] = {};

    float* d_in = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_buf, n * sizeof(float), cudaMemcpyHostToDevice));

    forgert::TensorShape shape({static_cast<size_t>(n)});
    // input on CUDA, output on CPU — should throw
    forgert::Tensor t_in (shape, forgert::DataType::Float32, d_in,      forgert::Device::CUDA);
    forgert::Tensor t_out(shape, forgert::DataType::Float32, h_out_buf,  forgert::Device::CPU);

    forgert::ReLUOp op;
    bool threw = false;
    try {
        op.execute(forgert::Backend::CUDA, {&t_in}, {&t_out});
    } catch (const std::invalid_argument&) {
        threw = true;
    } catch (const std::runtime_error&) {
        threw = true;
    }

    CUDA_CHECK(cudaFree(d_in));

    if (threw) pass("CPU output tensor rejected by CUDA backend");
    else        fail("CPU output tensor NOT rejected — should have thrown");
}

// --------------------------------------------------------------------- main --

int main() {
    printf("========================================\n");
    printf("ForgeRT ReLUOp CUDA Backend Integration Tests\n");
    printf("========================================\n");

    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    test_supports_backend();
    test_small_tensor();
    test_large_tensor();
    test_cpu_vs_cuda_equivalence();
    test_error_wrong_input_device();
    test_error_wrong_output_device();

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    printf("========================================\n");

    return (g_fail == 0) ? 0 : 1;
}

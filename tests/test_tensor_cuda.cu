/*
 * test_tensor_cuda.cu  –  Tensor CUDA memory ownership tests (Phase 4)
 *
 * Validates that Tensor can allocate, use, and correctly destroy CUDA device
 * memory.  All tests use Device::CUDA through the normal Tensor API; the raw
 * cuda_runtime.h calls here are only for verification (reading back values).
 *
 * Tests:
 *   1. Owning CUDA allocation: constructor succeeds, device() == CUDA
 *   2. RAII: destructor frees device memory (verified via cudaPointerAttributes)
 *   3. Move constructor: pointer transfers, source becomes null
 *   4. Move assignment: same guarantee as move constructor
 *   5. zero(): cudaMemset zeroes device buffer
 *   6. fill(): H2D-staged fill produces correct values on device
 *   7. copyTo() CPU→CUDA (H2D)
 *   8. copyTo() CUDA→CPU (D2H)
 *   9. copyTo() CUDA→CUDA (D2D)
 *  10. copyTo() CPU→CPU (memcpy through Tensor API)
 *  11. Non-owning tensor wrapping external device pointer: dtor does not free
 *  12. Graph + ReLUOp end-to-end: execute(Backend::CUDA) via owned tensors
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <memory>

#include "forgert/tensor/tensor.h"
#include "forgert/operator/relu.h"
#include "forgert/graph/graph.h"

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

static void pass(const char* msg) {
    ++g_pass;
    printf("  \xE2\x9C\x93 %s\n", msg);
}
static void fail(const char* msg) {
    ++g_fail;
    printf("  FAIL: %s\n", msg);
}

// Read a single float from device memory into host.
static float read_device_float(const void* d_ptr, size_t idx = 0) {
    float v = 0.0f;
    CUDA_CHECK(cudaMemcpy(&v,
        static_cast<const float*>(d_ptr) + idx,
        sizeof(float), cudaMemcpyDeviceToHost));
    return v;
}

// -------------------------------------------------------------------- tests --

static void test_cuda_allocation() {
    printf("  Test 1: Owning CUDA allocation\n");
    TensorShape shape({64, 16});
    Tensor t(shape, DataType::Float32, Device::CUDA);
    bool ok = (t.device() == Device::CUDA)
           && (t.numElements() == 1024)
           && (t.numBytes()    == 4096)
           && (t.data()        != nullptr);
    if (ok) pass("CUDA allocation"); else fail("CUDA allocation");
}

static void test_raii_frees_memory() {
    printf("  Test 2: RAII – destructor frees device memory\n");
    void* raw_ptr = nullptr;
    {
        Tensor t(TensorShape({32}), DataType::Float32, Device::CUDA);
        raw_ptr = t.data();
        // t goes out of scope here → cudaFree called
    }
    // After free, cudaPointerAttributes should report the pointer as invalid
    cudaPointerAttributes attrs{};
    cudaError_t err = cudaPointerGetAttributes(&attrs, raw_ptr);
    // Either the call errors or reports the memory is no longer device memory.
    bool freed = (err != cudaSuccess) || (attrs.type == cudaMemoryTypeUnregistered);
    cudaGetLastError(); // clear any error
    if (freed) pass("RAII frees device memory");
    else       fail("RAII frees device memory – pointer still valid after dtor");
}

static void test_move_constructor() {
    printf("  Test 3: Move constructor\n");
    Tensor t1(TensorShape({16}), DataType::Float32, Device::CUDA);
    void* original = t1.data();

    Tensor t2(std::move(t1));
    bool ok = (t2.data()   == original)
           && (t1.data()   == nullptr)
           && (t2.device() == Device::CUDA)
           && (t2.numElements() == 16);
    if (ok) pass("Move constructor"); else fail("Move constructor");
}

static void test_move_assignment() {
    printf("  Test 4: Move assignment\n");
    Tensor t1(TensorShape({8}), DataType::Float32, Device::CUDA);
    Tensor t2(TensorShape({8}), DataType::Float32, Device::CUDA);
    void* ptr1 = t1.data();

    t2 = std::move(t1);
    bool ok = (t2.data() == ptr1) && (t1.data() == nullptr);
    if (ok) pass("Move assignment"); else fail("Move assignment");
}

static void test_zero() {
    printf("  Test 5: zero() zeroes device buffer\n");
    const int n = 64;
    Tensor t(TensorShape({static_cast<size_t>(n)}), DataType::Float32, Device::CUDA);
    // Fill with non-zero first
    float h[n];
    for (int i = 0; i < n; ++i) h[i] = static_cast<float>(i + 1);
    CUDA_CHECK(cudaMemcpy(t.data(), h, n * sizeof(float), cudaMemcpyHostToDevice));

    t.zero();

    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (read_device_float(t.data(), i) != 0.0f) { ok = false; break; }
    }
    if (ok) pass("zero() on CUDA tensor"); else fail("zero() on CUDA tensor");
}

static void test_fill_cuda() {
    printf("  Test 6: fill() stages H2D copy\n");
    const int n = 128;
    Tensor t(TensorShape({static_cast<size_t>(n)}), DataType::Float32, Device::CUDA);
    t.fill(2.718f);

    bool ok = true;
    for (int i = 0; i < n; i += 8) {
        float v = read_device_float(t.data(), i);
        if (fabsf(v - 2.718f) > kTol) { ok = false; break; }
    }
    if (ok) pass("fill() on CUDA tensor"); else fail("fill() on CUDA tensor");
}

static void test_copy_h2d() {
    printf("  Test 7: copyTo() CPU → CUDA (H2D)\n");
    const int n = 32;
    Tensor cpu(TensorShape({static_cast<size_t>(n)}), DataType::Float32, Device::CPU);
    Tensor gpu(TensorShape({static_cast<size_t>(n)}), DataType::Float32, Device::CUDA);

    float* h = static_cast<float*>(cpu.data());
    for (int i = 0; i < n; ++i) h[i] = static_cast<float>(i) * 0.5f;

    cpu.copyTo(gpu);

    bool ok = true;
    for (int i = 0; i < n; ++i) {
        float v = read_device_float(gpu.data(), i);
        if (fabsf(v - h[i]) > kTol) { ok = false; break; }
    }
    if (ok) pass("copyTo() H2D"); else fail("copyTo() H2D");
}

static void test_copy_d2h() {
    printf("  Test 8: copyTo() CUDA → CPU (D2H)\n");
    const int n = 32;
    Tensor gpu(TensorShape({static_cast<size_t>(n)}), DataType::Float32, Device::CUDA);
    Tensor cpu(TensorShape({static_cast<size_t>(n)}), DataType::Float32, Device::CPU);

    // Write known values to device
    float h_src[n];
    for (int i = 0; i < n; ++i) h_src[i] = static_cast<float>(i) * 3.0f;
    CUDA_CHECK(cudaMemcpy(gpu.data(), h_src, n * sizeof(float), cudaMemcpyHostToDevice));

    gpu.copyTo(cpu);

    float* h_dst = static_cast<float*>(cpu.data());
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (fabsf(h_dst[i] - h_src[i]) > kTol) { ok = false; break; }
    }
    if (ok) pass("copyTo() D2H"); else fail("copyTo() D2H");
}

static void test_copy_d2d() {
    printf("  Test 9: copyTo() CUDA → CUDA (D2D)\n");
    const int n = 64;
    Tensor src(TensorShape({static_cast<size_t>(n)}), DataType::Float32, Device::CUDA);
    Tensor dst(TensorShape({static_cast<size_t>(n)}), DataType::Float32, Device::CUDA);

    src.fill(7.0f);
    src.copyTo(dst);

    bool ok = true;
    for (int i = 0; i < n; i += 4) {
        float v = read_device_float(dst.data(), i);
        if (fabsf(v - 7.0f) > kTol) { ok = false; break; }
    }
    if (ok) pass("copyTo() D2D"); else fail("copyTo() D2D");
}

static void test_copy_cpu_to_cpu() {
    printf("  Test 10: copyTo() CPU → CPU (memcpy via Tensor API)\n");
    const int n = 16;
    Tensor src(TensorShape({static_cast<size_t>(n)}), DataType::Float32, Device::CPU);
    Tensor dst(TensorShape({static_cast<size_t>(n)}), DataType::Float32, Device::CPU);

    float* s = static_cast<float*>(src.data());
    for (int i = 0; i < n; ++i) s[i] = static_cast<float>(i);
    src.copyTo(dst);

    float* d = static_cast<float*>(dst.data());
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (fabsf(d[i] - s[i]) > kTol) { ok = false; break; }
    }
    if (ok) pass("copyTo() CPU→CPU"); else fail("copyTo() CPU→CPU");
}

static void test_non_owning_device_tensor() {
    printf("  Test 11: Non-owning CUDA tensor does not double-free\n");
    // Allocate raw device memory manually
    float* raw = nullptr;
    CUDA_CHECK(cudaMalloc(&raw, 8 * sizeof(float)));

    {
        // Wrap in non-owning Tensor; dtor must NOT call cudaFree on raw
        Tensor t(TensorShape({8}), DataType::Float32,
                 static_cast<void*>(raw), Device::CUDA);
        (void)t; // goes out of scope
    }

    // raw is still valid — we can still use it
    float val = 1.23f;
    cudaError_t err = cudaMemcpy(raw, &val, sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(cudaFree(raw));

    if (err == cudaSuccess) pass("Non-owning tensor does not free external ptr");
    else                    fail("Non-owning tensor: raw ptr corrupted");
}

static void test_graph_cuda_relu() {
    printf("  Test 12: Graph + ReLUOp end-to-end on CUDA\n");

    // Build graph: single-input, single ReLU node
    Graph graph;
    size_t in_id  = graph.addInput(TensorShape({8}), DataType::Float32);
    auto   out_ids = graph.addNode(std::make_unique<ReLUOp>(), {in_id});
    graph.markOutput(out_ids[0]);
    graph.validate();

    // Create input tensor on CPU, upload to CUDA
    Tensor h_input(TensorShape({8}), DataType::Float32, Device::CPU);
    float* h = static_cast<float*>(h_input.data());
    h[0]= -3.0f; h[1]= 1.0f; h[2]= -0.5f; h[3]=  2.0f;
    h[4]=  0.0f; h[5]=-1.5f; h[6]=  4.0f; h[7]= -2.0f;

    Tensor d_input(TensorShape({8}), DataType::Float32, Device::CUDA);
    h_input.copyTo(d_input);

    // Execute graph on CUDA backend
    auto d_outputs = graph.execute({&d_input}, Backend::CUDA);
    assert(d_outputs.size() == 1);

    // Copy result back to host for verification
    Tensor h_output(TensorShape({8}), DataType::Float32, Device::CPU);
    d_outputs[0]->copyTo(h_output);

    float* out = static_cast<float*>(h_output.data());
    float expected[8];
    for (int i = 0; i < 8; ++i) expected[i] = (h[i] > 0.0f) ? h[i] : 0.0f;

    bool ok = true;
    for (int i = 0; i < 8; ++i) {
        if (fabsf(out[i] - expected[i]) > kTol) { ok = false; break; }
    }
    if (ok) pass("Graph + ReLUOp end-to-end on CUDA");
    else    fail("Graph + ReLUOp end-to-end on CUDA");
}

// --------------------------------------------------------------------- main --

int main() {
    printf("========================================\n");
    printf("ForgeRT Tensor CUDA Memory Tests (Phase 4)\n");
    printf("========================================\n");

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    test_cuda_allocation();
    test_raii_frees_memory();
    test_move_constructor();
    test_move_assignment();
    test_zero();
    test_fill_cuda();
    test_copy_h2d();
    test_copy_d2h();
    test_copy_d2d();
    test_copy_cpu_to_cpu();
    test_non_owning_device_tensor();
    test_graph_cuda_relu();

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    printf("========================================\n");
    return (g_fail == 0) ? 0 : 1;
}

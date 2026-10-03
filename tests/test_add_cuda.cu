/*
 * test_add_cuda.cu  –  CUDA Add operator tests for ForgeRT (Phase 3)
 *
 * Covers:
 *   1. Same-shape add (fast path, 1D)
 *   2. Same-shape add (fast path, 2D)
 *   3. CPU vs CUDA numerical equivalence – same shape
 *   4. Broadcast: [3,4] + [4]   -> [3,4]   (row vector broadcast)
 *   5. Broadcast: [3,4] + [3,1] -> [3,4]   (column vector broadcast)
 *   6. Broadcast: [3,4] + [1,4] -> [3,4]
 *   7. Broadcast: [3,4] + scalar([1,1]) -> [3,4]
 *   8. Large tensor – grid-stride correctness (1M elements)
 *   9. CPU vs CUDA equivalence – broadcast case
 *  10. Error path – CPU tensor rejected by CUDA backend
 */

#include "forgert/operator/add.h"
#include "forgert/tensor/tensor.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <cmath>
#include <string>
#include <vector>

using namespace forgert;

static const float TOL = 1e-5f;

// ------------------------------------------------------------------ helpers --

static void printDevice() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name
              << "  (sm_" << prop.major << prop.minor << ")\n";
}

// Fill a CPU tensor with a simple pattern
static void fillSeq(Tensor& t, float start = 0.0f, float step = 1.0f) {
    float* p = static_cast<float*>(t.data());
    for (size_t i = 0; i < t.numElements(); ++i)
        p[i] = start + step * static_cast<float>(i);
}

// Execute AddOp on CUDA and return result as a CPU tensor
static Tensor cudaAdd(const Tensor& a_cpu, const Tensor& b_cpu,
                       const TensorShape& out_shape) {
    Tensor a_dev(a_cpu.shape(), DataType::Float32, Device::CUDA);
    Tensor b_dev(b_cpu.shape(), DataType::Float32, Device::CUDA);
    Tensor c_dev(out_shape,     DataType::Float32, Device::CUDA);

    a_cpu.copyTo(a_dev);
    b_cpu.copyTo(b_dev);

    AddOp op;
    op.execute(Backend::CUDA, {&a_dev, &b_dev}, {&c_dev});

    Tensor c_host(out_shape, DataType::Float32, Device::CPU);
    c_dev.copyTo(c_host);
    return c_host;
}

// Execute AddOp on CPU
static Tensor cpuAdd(const Tensor& a, const Tensor& b,
                      const TensorShape& out_shape) {
    Tensor c(out_shape, DataType::Float32, Device::CPU);
    AddOp op;
    op.execute(Backend::CPU, {&a, &b}, {&c});
    return c;
}

static int passed = 0, failed = 0;

static void check(const std::string& name, bool ok) {
    if (ok) {
        std::cout << "  \u2713 " << name << "\n";
        ++passed;
    } else {
        std::cout << "  \u2717 FAILED: " << name << "\n";
        ++failed;
    }
}

// ------------------------------------------------------------------ tests ---

void test_same_shape_1d() {
    std::cout << "  Test 1: same-shape 1D add\n";

    TensorShape shape({6});
    Tensor a(shape, DataType::Float32, Device::CPU);
    Tensor b(shape, DataType::Float32, Device::CPU);
    fillSeq(a, 1.0f, 1.0f);   // [1,2,3,4,5,6]
    fillSeq(b, 10.0f, 10.0f); // [10,20,30,40,50,60]

    Tensor c = cudaAdd(a, b, shape);
    const float* cp = static_cast<const float*>(c.data());

    bool ok = true;
    for (int i = 0; i < 6; ++i)
        ok &= std::abs(cp[i] - ((i+1) + (i+1)*10.0f)) < TOL;
    check("same-shape 1D add", ok);
}

void test_same_shape_2d() {
    std::cout << "  Test 2: same-shape 2D add\n";

    TensorShape shape({4, 8});
    Tensor a(shape, DataType::Float32, Device::CPU);
    Tensor b(shape, DataType::Float32, Device::CPU);
    fillSeq(a, 0.0f, 1.0f);
    fillSeq(b, 100.0f, 1.0f);

    Tensor c = cudaAdd(a, b, shape);
    const float* cp = static_cast<const float*>(c.data());

    bool ok = true;
    for (size_t i = 0; i < 32; ++i)
        ok &= std::abs(cp[i] - (static_cast<float>(i) + 100.0f + static_cast<float>(i))) < TOL;
    check("same-shape 2D add", ok);
}

void test_cpu_cuda_equivalence_same_shape() {
    std::cout << "  Test 3: CPU vs CUDA equivalence (same shape)\n";

    TensorShape shape({16, 32});
    Tensor a(shape, DataType::Float32, Device::CPU);
    Tensor b(shape, DataType::Float32, Device::CPU);
    fillSeq(a, -50.0f, 0.5f);
    fillSeq(b,  10.0f, 0.3f);

    Tensor cpu_out = cpuAdd(a, b, shape);
    Tensor gpu_out = cudaAdd(a, b, shape);

    const float* cp = static_cast<const float*>(cpu_out.data());
    const float* gp = static_cast<const float*>(gpu_out.data());

    bool ok = true;
    for (size_t i = 0; i < shape.numElements(); ++i)
        ok &= std::abs(cp[i] - gp[i]) < TOL;
    check("CPU vs CUDA equivalence (same shape)", ok);
}

void test_broadcast_row_vector() {
    std::cout << "  Test 4: broadcast [3,4] + [4] -> [3,4]\n";

    // A: [3,4], B: [4] broadcast to each row
    TensorShape sa({3, 4}), sb({4}), sc({3, 4});

    Tensor a(sa, DataType::Float32, Device::CPU);
    Tensor b(sb, DataType::Float32, Device::CPU);

    float* ap = static_cast<float*>(a.data());
    float* bp = static_cast<float*>(b.data());

    // A rows: [0..3], [4..7], [8..11]
    for (int i = 0; i < 12; ++i) ap[i] = static_cast<float>(i);
    // B: [10, 20, 30, 40]
    bp[0]=10; bp[1]=20; bp[2]=30; bp[3]=40;

    Tensor cpu_out = cpuAdd(a, b, sc);
    Tensor gpu_out = cudaAdd(a, b, sc);

    const float* cp = static_cast<const float*>(cpu_out.data());
    const float* gp = static_cast<const float*>(gpu_out.data());

    bool ok = true;
    for (int i = 0; i < 12; ++i) ok &= std::abs(cp[i] - gp[i]) < TOL;
    // Spot-check element [1][2]: a=6, b=30 -> 36
    ok &= std::abs(gp[1*4+2] - 36.0f) < TOL;
    check("broadcast [3,4] + [4] -> [3,4]", ok);
}

void test_broadcast_col_vector() {
    std::cout << "  Test 5: broadcast [3,4] + [3,1] -> [3,4]\n";

    TensorShape sa({3, 4}), sb({3, 1}), sc({3, 4});
    Tensor a(sa, DataType::Float32, Device::CPU);
    Tensor b(sb, DataType::Float32, Device::CPU);

    float* ap = static_cast<float*>(a.data());
    float* bp = static_cast<float*>(b.data());
    for (int i = 0; i < 12; ++i) ap[i] = static_cast<float>(i);
    bp[0]=100; bp[1]=200; bp[2]=300;  // column vector

    Tensor cpu_out = cpuAdd(a, b, sc);
    Tensor gpu_out = cudaAdd(a, b, sc);

    const float* cp = static_cast<const float*>(cpu_out.data());
    const float* gp = static_cast<const float*>(gpu_out.data());

    bool ok = true;
    for (int i = 0; i < 12; ++i) ok &= std::abs(cp[i] - gp[i]) < TOL;
    // Spot-check: row 2, col 3 -> a=11, b=300 -> 311
    ok &= std::abs(gp[2*4+3] - 311.0f) < TOL;
    check("broadcast [3,4] + [3,1] -> [3,4]", ok);
}

void test_broadcast_row_shape() {
    std::cout << "  Test 6: broadcast [3,4] + [1,4] -> [3,4]\n";

    TensorShape sa({3, 4}), sb({1, 4}), sc({3, 4});
    Tensor a(sa, DataType::Float32, Device::CPU);
    Tensor b(sb, DataType::Float32, Device::CPU);

    float* ap = static_cast<float*>(a.data());
    float* bp = static_cast<float*>(b.data());
    for (int i = 0; i < 12; ++i) ap[i] = static_cast<float>(i);
    bp[0]=1; bp[1]=2; bp[2]=3; bp[3]=4;

    Tensor cpu_out = cpuAdd(a, b, sc);
    Tensor gpu_out = cudaAdd(a, b, sc);

    const float* cp = static_cast<const float*>(cpu_out.data());
    const float* gp = static_cast<const float*>(gpu_out.data());

    bool ok = true;
    for (int i = 0; i < 12; ++i) ok &= std::abs(cp[i] - gp[i]) < TOL;
    check("broadcast [3,4] + [1,4] -> [3,4]", ok);
}

void test_broadcast_scalar() {
    std::cout << "  Test 7: broadcast [3,4] + [1,1] -> [3,4]  (scalar-like)\n";

    TensorShape sa({3, 4}), sb({1, 1}), sc({3, 4});
    Tensor a(sa, DataType::Float32, Device::CPU);
    Tensor b(sb, DataType::Float32, Device::CPU);

    float* ap = static_cast<float*>(a.data());
    float* bp = static_cast<float*>(b.data());
    for (int i = 0; i < 12; ++i) ap[i] = static_cast<float>(i);
    bp[0] = 5.0f;  // scalar value

    Tensor cpu_out = cpuAdd(a, b, sc);
    Tensor gpu_out = cudaAdd(a, b, sc);

    const float* cp = static_cast<const float*>(cpu_out.data());
    const float* gp = static_cast<const float*>(gpu_out.data());

    bool ok = true;
    for (int i = 0; i < 12; ++i) ok &= std::abs(cp[i] - gp[i]) < TOL;
    // Every output = input + 5
    for (int i = 0; i < 12; ++i) ok &= std::abs(gp[i] - (static_cast<float>(i) + 5.0f)) < TOL;
    check("broadcast [3,4] + [1,1] (scalar-like)", ok);
}

void test_large_tensor_grid_stride() {
    std::cout << "  Test 8: large tensor (1M elements, grid-stride)\n";

    const size_t N = 1024 * 1024;
    TensorShape shape({N});
    Tensor a(shape, DataType::Float32, Device::CPU);
    Tensor b(shape, DataType::Float32, Device::CPU);

    float* ap = static_cast<float*>(a.data());
    float* bp = static_cast<float*>(b.data());
    for (size_t i = 0; i < N; ++i) {
        ap[i] = static_cast<float>(i % 1000);
        bp[i] = static_cast<float>(i % 500) * 0.5f;
    }

    Tensor gpu_out = cudaAdd(a, b, shape);
    const float* gp = static_cast<const float*>(gpu_out.data());

    bool ok = true;
    for (size_t i = 0; i < N; ++i) {
        float expected = ap[i] + bp[i];
        if (std::abs(gp[i] - expected) >= TOL) { ok = false; break; }
    }
    check("large tensor grid-stride (1M elements)", ok);
}

void test_cpu_cuda_equivalence_broadcast() {
    std::cout << "  Test 9: CPU vs CUDA equivalence (broadcast [8,16] + [16])\n";

    TensorShape sa({8, 16}), sb({16}), sc({8, 16});
    Tensor a(sa, DataType::Float32, Device::CPU);
    Tensor b(sb, DataType::Float32, Device::CPU);
    fillSeq(a, -10.0f, 0.25f);
    fillSeq(b,   0.0f, 1.0f);

    Tensor cpu_out = cpuAdd(a, b, sc);
    Tensor gpu_out = cudaAdd(a, b, sc);

    const float* cp = static_cast<const float*>(cpu_out.data());
    const float* gp = static_cast<const float*>(gpu_out.data());

    bool ok = true;
    for (size_t i = 0; i < sc.numElements(); ++i)
        ok &= std::abs(cp[i] - gp[i]) < TOL;
    check("CPU vs CUDA equivalence (broadcast [8,16]+[16])", ok);
}

void test_error_cpu_tensor_rejected() {
    std::cout << "  Test 10: CPU tensor rejected by CUDA backend\n";

    TensorShape shape({4});
    Tensor a_cpu(shape, DataType::Float32, Device::CPU);
    Tensor b_cpu(shape, DataType::Float32, Device::CPU);
    Tensor c_dev(shape, DataType::Float32, Device::CUDA);

    AddOp op;
    bool caught = false;
    try {
        op.execute(Backend::CUDA, {&a_cpu, &b_cpu}, {&c_dev});
    } catch (const std::exception&) {
        caught = true;
    }
    check("CPU tensor rejected by CUDA backend", caught);
}

// ------------------------------------------------------------------ main ---

int main() {
    std::cout << "========================================\n";
    std::cout << "ForgeRT CUDA Add Operator Tests\n";
    std::cout << "========================================\n";
    printDevice();

    test_same_shape_1d();
    test_same_shape_2d();
    test_cpu_cuda_equivalence_same_shape();
    test_broadcast_row_vector();
    test_broadcast_col_vector();
    test_broadcast_row_shape();
    test_broadcast_scalar();
    test_large_tensor_grid_stride();
    test_cpu_cuda_equivalence_broadcast();
    test_error_cpu_tensor_rejected();

    std::cout << "========================================\n";
    std::cout << "Results: " << passed << " passed, " << failed << " failed\n";
    std::cout << "========================================\n";
    return failed == 0 ? 0 : 1;
}

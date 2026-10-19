/*
 * test_conv2d_cuda.cu  --  Conv2DOp CUDA correctness tests
 *
 * Strategy: run every test case on CPU first, then on CUDA, compare outputs.
 * This avoids maintaining a separate reference formula; the CPU path is the
 * ground truth (tested independently in test_conv2d.cpp).
 *
 * Coverage:
 *  1. 1x1 conv with bias       (matches CPU exactly)
 *  2. 3x3 / pad=1 / stride=1  (validates padding logic in im2col)
 *  3. 3x3 / stride=2 / no-pad (validates stride in im2col)
 *  4. No-bias path             (nullptr bias device pointer)
 *  5. Larger tensor [1,64,55,55] 1x1 squeeze layer (realistic SqueezeNet size)
 *  6. Larger tensor [1,16,55,55] 3x3 expand layer
 */

#include "forgert/operator/conv2d.h"
#include "forgert/tensor/tensor.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using namespace forgert;

static constexpr float MAX_ABS_DIFF = 1e-3f;  // float32 GEMM accumulation tolerance

static void ok(bool cond, const char* msg) {
    if (!cond) { std::cerr << "FAIL: " << msg << "\n"; std::exit(1); }
    std::cout << "PASS: " << msg << "\n";
}

static void fill_random(Tensor& t, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    float* p = static_cast<float*>(t.data());
    for (size_t i = 0; i < t.numElements(); ++i) p[i] = dist(rng);
}
static void fill_const(Tensor& t, float v) {
    float* p = static_cast<float*>(t.data());
    for (size_t i = 0; i < t.numElements(); ++i) p[i] = v;
}

// Run op on CPU + CUDA and assert max diff < MAX_ABS_DIFF.
static bool compare_cpu_cuda(
    Conv2DOp& op,
    const std::vector<TensorShape>& in_shapes,
    unsigned seed = 42)
{
    // Allocate CPU tensors and fill
    std::vector<Tensor> cpu_in;
    for (const auto& s : in_shapes)
        cpu_in.emplace_back(s, DataType::Float32, Device::CPU);

    fill_random(cpu_in[0], seed);
    fill_random(cpu_in[1], seed + 1);
    if (cpu_in.size() == 3)
        fill_random(cpu_in[2], seed + 2);

    auto out_shapes = op.inferOutputShapes(in_shapes);
    Tensor cpu_out(out_shapes[0], DataType::Float32, Device::CPU);

    std::vector<const Tensor*> cpu_in_ptrs;
    std::vector<Tensor*> cpu_out_ptrs = {&cpu_out};
    for (auto& t : cpu_in) cpu_in_ptrs.push_back(&t);
    op.execute(Backend::CPU, cpu_in_ptrs, cpu_out_ptrs);

    // Allocate CUDA tensors, copy from CPU
    std::vector<Tensor> cuda_in;
    for (const auto& s : in_shapes)
        cuda_in.emplace_back(s, DataType::Float32, Device::CUDA);
    for (size_t i = 0; i < cpu_in.size(); ++i)
        cpu_in[i].copyTo(cuda_in[i]);

    Tensor cuda_out(out_shapes[0], DataType::Float32, Device::CUDA);

    std::vector<const Tensor*> cuda_in_ptrs;
    std::vector<Tensor*> cuda_out_ptrs = {&cuda_out};
    for (auto& t : cuda_in) cuda_in_ptrs.push_back(&t);
    op.execute(Backend::CUDA, cuda_in_ptrs, cuda_out_ptrs);

    // Copy CUDA result to host
    Tensor cuda_out_host(out_shapes[0], DataType::Float32, Device::CPU);
    cuda_out.copyTo(cuda_out_host);

    // Compare
    const float* cpu_d  = static_cast<const float*>(cpu_out.data());
    const float* cuda_d = static_cast<const float*>(cuda_out_host.data());
    const size_t N      = cpu_out.numElements();

    float max_diff = 0.0f;
    for (size_t i = 0; i < N; ++i)
        max_diff = std::max(max_diff, std::fabs(cpu_d[i] - cuda_d[i]));

    if (max_diff > MAX_ABS_DIFF) {
        std::cerr << "  max_diff=" << max_diff << " (limit=" << MAX_ABS_DIFF << ")\n";
        return false;
    }
    return true;
}

static void test_1x1_with_bias() {
    Conv2DParams p; p.kH=1; p.kW=1; p.stride_h=1; p.stride_w=1; p.pad_h=0; p.pad_w=0;
    Conv2DOp op(p);
    ok(compare_cpu_cuda(op, {TensorShape({1,3,8,8}),
                              TensorShape({4,3,1,1}),
                              TensorShape({4})}),
       "CUDA Test 1: 1x1 with bias matches CPU");
}

static void test_3x3_pad1_stride1() {
    Conv2DParams p; p.kH=3; p.kW=3; p.stride_h=1; p.stride_w=1; p.pad_h=1; p.pad_w=1;
    Conv2DOp op(p);
    ok(compare_cpu_cuda(op, {TensorShape({1,4,6,6}),
                              TensorShape({8,4,3,3}),
                              TensorShape({8})}),
       "CUDA Test 2: 3x3/pad=1/stride=1 matches CPU");
}

static void test_3x3_stride2_nopad() {
    Conv2DParams p; p.kH=3; p.kW=3; p.stride_h=2; p.stride_w=2; p.pad_h=0; p.pad_w=0;
    Conv2DOp op(p);
    ok(compare_cpu_cuda(op, {TensorShape({1,3,7,7}),
                              TensorShape({16,3,3,3}),
                              TensorShape({16})}),
       "CUDA Test 3: 3x3/stride=2/no-pad matches CPU");
}

static void test_no_bias() {
    Conv2DParams p; p.kH=1; p.kW=1; p.stride_h=1; p.stride_w=1; p.pad_h=0; p.pad_w=0;
    Conv2DOp op(p);
    ok(compare_cpu_cuda(op, {TensorShape({1,8,4,4}),
                              TensorShape({4,8,1,1})}),
       "CUDA Test 4: no-bias path matches CPU");
}

static void test_realistic_1x1_squeeze() {
    // SqueezeNet fire module squeeze: [1,128,27,27] -> C_out=32
    Conv2DParams p; p.kH=1; p.kW=1; p.stride_h=1; p.stride_w=1; p.pad_h=0; p.pad_w=0;
    Conv2DOp op(p);
    ok(compare_cpu_cuda(op, {TensorShape({1,128,27,27}),
                              TensorShape({32,128,1,1}),
                              TensorShape({32})}, 7),
       "CUDA Test 5: realistic 1x1 squeeze [1,128,27,27] matches CPU");
}

static void test_realistic_3x3_expand() {
    // SqueezeNet fire module expand: [1,32,27,27] -> C_out=128 3x3
    Conv2DParams p; p.kH=3; p.kW=3; p.stride_h=1; p.stride_w=1; p.pad_h=1; p.pad_w=1;
    Conv2DOp op(p);
    ok(compare_cpu_cuda(op, {TensorShape({1,32,27,27}),
                              TensorShape({128,32,3,3}),
                              TensorShape({128})}, 13),
       "CUDA Test 6: realistic 3x3 expand [1,32,27,27] matches CPU");
}

int main() {
    std::cout << "=== Conv2D CUDA Tests ===\n\n";
    test_1x1_with_bias();
    test_3x3_pad1_stride1();
    test_3x3_stride2_nopad();
    test_no_bias();
    test_realistic_1x1_squeeze();
    test_realistic_3x3_expand();
    std::cout << "\n=== All Conv2D CUDA tests PASSED ===\n";
    return 0;
}

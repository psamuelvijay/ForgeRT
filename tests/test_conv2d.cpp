/*
 * test_conv2d.cpp  --  Conv2DOp correctness tests (CPU backend)
 *
 * Coverage:
 *  1. 1x1 conv, no pad, stride 1, with bias    (SqueezeNet 1x1 "squeeze" layers)
 *  2. 3x3 conv, pad=1, stride 1, with bias     (SqueezeNet 3x3 "expand" layers)
 *  3. 3x3 conv, no pad, stride=2, with bias    (SqueezeNet first Conv layer)
 *  4. No-bias variant                           (ensure bias=0 path)
 *  5. Multi-batch (N=2)
 *  6. Known-values reference check             (hand-computed expected output)
 *  7. Zero-padding boundary check              (output at padding boundary == 0-contribution)
 *  8. Input/output shape validation error      (wrong C_in, wrong kernel params)
 *  9. Match against PyTorch-style formula      (H_out = (H+2p-k)/s + 1)
 *
 * Reference: conv2d CPU uses a direct naive loop — matches by construction.
 * CUDA tests are in test_conv2d_cuda.cu.
 */

#include "forgert/operator/conv2d.h"
#include "forgert/tensor/tensor.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace forgert;

static void ok(bool cond, const char* msg) {
    if (!cond) { std::cerr << "FAIL: " << msg << "\n"; std::exit(1); }
    std::cout << "PASS: " << msg << "\n";
}

static bool nearlyEq(float a, float b, float tol = 1e-5f) {
    return std::fabs(a - b) <= tol;
}

// Fill tensor data with a predictable pattern
static void fill_sequential(Tensor& t, float start = 0.1f, float step = 0.1f) {
    float* p = static_cast<float*>(t.data());
    for (size_t i = 0; i < t.numElements(); ++i)
        p[i] = start + step * static_cast<float>(i);
}
static void fill_const(Tensor& t, float v) {
    float* p = static_cast<float*>(t.data());
    for (size_t i = 0; i < t.numElements(); ++i) p[i] = v;
}

// ------------------------------------------------------------------ test 1: 1x1 conv
static void test_1x1_conv_with_bias() {
    // N=1, C_in=3, H=4, W=4 -> C_out=2, 1x1, stride=1, pad=0
    // H_out = W_out = 4
    Conv2DParams p; p.kH=1; p.kW=1; p.stride_h=1; p.stride_w=1; p.pad_h=0; p.pad_w=0;
    Conv2DOp op(p);

    Tensor input(TensorShape({1,3,4,4}), DataType::Float32, Device::CPU);
    Tensor weight(TensorShape({2,3,1,1}), DataType::Float32, Device::CPU);
    Tensor bias(TensorShape({2}), DataType::Float32, Device::CPU);
    // infer output shape
    auto out_shapes = op.inferOutputShapes({input.shape(), weight.shape(), bias.shape()});
    ok(out_shapes[0].dim(0)==1 && out_shapes[0].dim(1)==2 &&
       out_shapes[0].dim(2)==4 && out_shapes[0].dim(3)==4,
       "Test 1: 1x1 output shape [1,2,4,4]");
    Tensor output(out_shapes[0], DataType::Float32, Device::CPU);

    fill_sequential(input,  0.1f, 0.1f);
    fill_sequential(weight, 0.2f, 0.1f);
    static_cast<float*>(bias.data())[0] = 0.5f;
    static_cast<float*>(bias.data())[1] = -0.5f;

    op.execute(Backend::CPU, {&input, &weight, &bias}, {&output});

    // For 1x1 conv: out[n,co,h,w] = sum_ci(in[n,ci,h,w]*w[co,ci,0,0]) + bias[co]
    // Verify one output element manually: out[0,0,0,0]
    const float* in_d = static_cast<const float*>(input.data());
    const float* w_d  = static_cast<const float*>(weight.data());
    float expected = 0.5f; // bias[0]
    for (int ci = 0; ci < 3; ++ci)
        expected += in_d[ci * 16] * w_d[ci]; // H*W=16 stride
    ok(nearlyEq(static_cast<const float*>(output.data())[0], expected),
       "Test 1: 1x1 output[0,0,0,0] matches manual calculation");
}

// ------------------------------------------------------------------ test 2: 3x3 pad=1
static void test_3x3_conv_pad1_stride1() {
    // N=1, C_in=2, H=5, W=5 -> C_out=3, 3x3, stride=1, pad=1
    // H_out = W_out = (5+2-3)/1+1 = 5
    Conv2DParams p; p.kH=3; p.kW=3; p.stride_h=1; p.stride_w=1; p.pad_h=1; p.pad_w=1;
    Conv2DOp op(p);

    Tensor input(TensorShape({1,2,5,5}), DataType::Float32, Device::CPU);
    Tensor weight(TensorShape({3,2,3,3}), DataType::Float32, Device::CPU);
    Tensor bias(TensorShape({3}), DataType::Float32, Device::CPU);

    auto out_shapes = op.inferOutputShapes({input.shape(), weight.shape(), bias.shape()});
    ok(out_shapes[0].dim(1)==3 && out_shapes[0].dim(2)==5 && out_shapes[0].dim(3)==5,
       "Test 2: 3x3/pad=1/stride=1 output shape [1,3,5,5]");

    Tensor output(out_shapes[0], DataType::Float32, Device::CPU);
    fill_const(input, 1.0f);
    fill_const(weight, 1.0f);
    fill_const(bias, 0.0f);

    op.execute(Backend::CPU, {&input, &weight, &bias}, {&output});

    // Input all-1, weight all-1: interior output = C_in * kH * kW = 2*9 = 18
    // Corner output (top-left): only 4 valid positions = C_in * 4 = 8
    const float* out_d = static_cast<const float*>(output.data());
    ok(nearlyEq(out_d[(0*3+0)*5*5 + 0*5 + 0], 8.0f),
       "Test 2: corner output = C_in*4 = 8 (padding zeros contribute 0)");
    ok(nearlyEq(out_d[(0*3+0)*5*5 + 2*5 + 2], 18.0f),
       "Test 2: interior output = C_in*kH*kW = 18");
    ok(nearlyEq(out_d[(0*3+0)*5*5 + 0*5 + 2], 12.0f),
       "Test 2: top-edge output (1 row padding) = C_in*6 = 12");
}

// ------------------------------------------------------------------ test 3: stride=2
static void test_3x3_conv_stride2() {
    // N=1, C_in=3, H=7, W=7 -> C_out=64, 3x3, stride=2, pad=0
    // H_out = W_out = (7+0-3)/2+1 = 3
    Conv2DParams p; p.kH=3; p.kW=3; p.stride_h=2; p.stride_w=2; p.pad_h=0; p.pad_w=0;
    Conv2DOp op(p);

    Tensor input(TensorShape({1,3,7,7}), DataType::Float32, Device::CPU);
    Tensor weight(TensorShape({4,3,3,3}), DataType::Float32, Device::CPU);
    Tensor bias(TensorShape({4}), DataType::Float32, Device::CPU);

    auto out_shapes = op.inferOutputShapes({input.shape(), weight.shape(), bias.shape()});
    ok(out_shapes[0].dim(2)==3 && out_shapes[0].dim(3)==3,
       "Test 3: 3x3/stride=2/no-pad output spatial = 3x3");

    Tensor output(out_shapes[0], DataType::Float32, Device::CPU);
    fill_const(input, 1.0f);
    fill_const(weight, 1.0f);
    fill_const(bias, 1.0f);

    op.execute(Backend::CPU, {&input, &weight, &bias}, {&output});

    // All-1 input+weight, 3x3 kernel, 3 input channels: sum = 3*9 + bias(1) = 28
    const float* out_d = static_cast<const float*>(output.data());
    ok(nearlyEq(out_d[0], 28.0f), "Test 3: stride=2 output[0] = C_in*kH*kW + bias = 28");
    ok(nearlyEq(out_d[1], 28.0f), "Test 3: stride=2 output[1] = 28");
}

// ------------------------------------------------------------------ test 4: no-bias
static void test_no_bias() {
    Conv2DParams p; p.kH=1; p.kW=1; p.stride_h=1; p.stride_w=1; p.pad_h=0; p.pad_w=0;
    Conv2DOp op(p);

    Tensor input(TensorShape({1,2,3,3}), DataType::Float32, Device::CPU);
    Tensor weight(TensorShape({2,2,1,1}), DataType::Float32, Device::CPU);

    auto out_shapes = op.inferOutputShapes({input.shape(), weight.shape()});
    Tensor output(out_shapes[0], DataType::Float32, Device::CPU);
    fill_const(input, 1.0f);
    fill_const(weight, 1.0f);

    op.execute(Backend::CPU, {&input, &weight}, {&output});

    // 1x1 conv, all-1: out = sum_ci(1*1) = C_in = 2, no bias
    const float* out_d = static_cast<const float*>(output.data());
    ok(nearlyEq(out_d[0], 2.0f), "Test 4: no-bias 1x1 output = C_in = 2");
}

// ------------------------------------------------------------------ test 5: batch N=2
static void test_multi_batch() {
    Conv2DParams p; p.kH=1; p.kW=1; p.stride_h=1; p.stride_w=1; p.pad_h=0; p.pad_w=0;
    Conv2DOp op(p);

    Tensor input(TensorShape({2,3,2,2}), DataType::Float32, Device::CPU);
    Tensor weight(TensorShape({2,3,1,1}), DataType::Float32, Device::CPU);
    Tensor bias(TensorShape({2}), DataType::Float32, Device::CPU);

    auto out_shapes = op.inferOutputShapes({input.shape(), weight.shape(), bias.shape()});
    ok(out_shapes[0].dim(0)==2, "Test 5: batch-2 output has N=2");
    Tensor output(out_shapes[0], DataType::Float32, Device::CPU);

    // Fill: batch 0 = 1.0, batch 1 = 2.0
    float* in_d = static_cast<float*>(input.data());
    for (int i = 0; i < 3*2*2; ++i) in_d[i]          = 1.0f;
    for (int i = 0; i < 3*2*2; ++i) in_d[3*2*2 + i]  = 2.0f;
    fill_const(weight, 1.0f);
    fill_const(bias, 0.0f);

    op.execute(Backend::CPU, {&input, &weight, &bias}, {&output});

    const float* out_d = static_cast<const float*>(output.data());
    // batch 0: sum_ci(1*1) = 3; batch 1: sum_ci(2*1) = 6
    ok(nearlyEq(out_d[0], 3.0f), "Test 5: batch-0 output = 3");
    ok(nearlyEq(out_d[2*2*2], 6.0f), "Test 5: batch-1 output = 6");
}

// ------------------------------------------------------------------ test 6: known-values
static void test_known_values() {
    // Minimal: N=1, C_in=1, H=3, W=3, C_out=1, 3x3 kernel, pad=1, stride=1
    // H_out=W_out=3
    Conv2DParams p; p.kH=3; p.kW=3; p.stride_h=1; p.stride_w=1; p.pad_h=1; p.pad_w=1;
    Conv2DOp op(p);

    Tensor input(TensorShape({1,1,3,3}), DataType::Float32, Device::CPU);
    Tensor weight(TensorShape({1,1,3,3}), DataType::Float32, Device::CPU);

    auto out_shapes = op.inferOutputShapes({input.shape(), weight.shape()});
    Tensor output(out_shapes[0], DataType::Float32, Device::CPU);

    // input = 1..9, weight = identity-like (centre=1, rest=0)
    float* in_d = static_cast<float*>(input.data());
    for (int i = 0; i < 9; ++i) in_d[i] = static_cast<float>(i + 1);  // 1..9
    float* w_d = static_cast<float*>(weight.data());
    for (int i = 0; i < 9; ++i) w_d[i] = 0.0f;
    w_d[4] = 1.0f;  // centre tap only

    op.execute(Backend::CPU, {&input, &weight}, {&output});

    // Identity convolution (centre=1) should reproduce input (with zero-padding at boundary)
    const float* out_d = static_cast<const float*>(output.data());
    ok(nearlyEq(out_d[0], 1.0f), "Test 6: identity conv output[0,0,0] = input[0,0] = 1");
    ok(nearlyEq(out_d[4], 5.0f), "Test 6: identity conv output[0,1,1] = input[1,1] = 5");
    ok(nearlyEq(out_d[8], 9.0f), "Test 6: identity conv output[0,2,2] = input[2,2] = 9");
}

// ------------------------------------------------------------------ test 7: error cases
static void test_error_cases() {
    Conv2DOp op({});  // 1x1 default

    // Wrong input ndim
    Tensor bad3d(TensorShape({2,3,4}), DataType::Float32, Device::CPU);
    Tensor w(TensorShape({1,3,1,1}), DataType::Float32, Device::CPU);
    try {
        op.inferOutputShapes({bad3d.shape(), w.shape()});
        ok(false, "Test 7: should throw on 3D input");
    } catch (const std::invalid_argument&) {
        ok(true, "Test 7: throws on 3D input");
    }

    // Mismatched C_in
    Tensor input4d(TensorShape({1,4,5,5}), DataType::Float32, Device::CPU);
    try {
        op.inferOutputShapes({input4d.shape(), w.shape()});
        ok(false, "Test 7: should throw on C_in mismatch");
    } catch (const std::invalid_argument&) {
        ok(true, "Test 7: throws on C_in mismatch");
    }

    // Wrong number of inputs
    try {
        op.inferOutputShapes({w.shape()});
        ok(false, "Test 7: should throw on 1 input");
    } catch (const std::invalid_argument&) {
        ok(true, "Test 7: throws on single-input");
    }
}

// ------------------------------------------------------------------ test 8: output formula
static void test_output_size_formula() {
    // Verify (H+2p-k)/s+1 for several combinations
    auto check = [](int H, int k, int p, int s, int expected_out) {
        Conv2DParams cp; cp.kH=k; cp.kW=k; cp.stride_h=s; cp.stride_w=s;
                        cp.pad_h=p; cp.pad_w=p;
        Conv2DOp op(cp);
        Tensor inp(TensorShape({1,1,(size_t)H,(size_t)H}), DataType::Float32, Device::CPU);
        Tensor wt(TensorShape({1,1,(size_t)k,(size_t)k}), DataType::Float32, Device::CPU);
        auto shapes = op.inferOutputShapes({inp.shape(), wt.shape()});
        return static_cast<int>(shapes[0].dim(2)) == expected_out;
    };
    ok(check(224,3,0,2,111), "Test 8: (224+0-3)/2+1=111");  // SqueezeNet first conv
    ok(check( 55,3,1,1, 55), "Test 8: (55+2-3)/1+1=55");    // pad=1 preserves size
    ok(check( 55,3,0,2, 27), "Test 8: (55+0-3)/2+1=27");    // MaxPool-like
    ok(check( 13,13,0,13, 1), "Test 8: (13+0-13)/13+1=1");  // AveragePool-like
}

int main() {
    std::cout << "=== Conv2D CPU Tests ===\n\n";
    test_1x1_conv_with_bias();
    test_3x3_conv_pad1_stride1();
    test_3x3_conv_stride2();
    test_no_bias();
    test_multi_batch();
    test_known_values();
    test_error_cases();
    test_output_size_formula();
    std::cout << "\n=== All Conv2D CPU tests PASSED ===\n";
    return 0;
}

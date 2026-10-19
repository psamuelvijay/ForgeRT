/*
 * test_maxpool2d.cpp  --  MaxPool2DOp correctness tests (CPU)
 *
 * Coverage:
 *  1. SqueezeNet use-case: [1,64,55,55] 3x3 stride=2 no-pad -> [1,64,27,27]
 *  2. Known-values: 1 channel, known max picks verified by hand
 *  3. Negative values: max correctly selects least-negative value
 *  4. Single-element window: output = input (stride=1, 1x1 kernel)
 *  5. Padding > 0: output size formula and pad-zero boundary check
 *  6. Multi-channel: each channel pooled independently
 *  7. Multi-batch: each batch item pooled independently
 *  8. Output size formula: (H+2p-k)/s+1 for several configs
 *  9. Error: 3-D input throws
 * 10. Error: too many inputs (expects exactly 1)
 */

#include "forgert/operator/maxpool2d.h"
#include "forgert/tensor/tensor.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace forgert;

static void ok(bool cond, const char* msg) {
    if (!cond) { std::cerr << "FAIL: " << msg << "\n"; std::exit(1); }
    std::cout << "PASS: " << msg << "\n";
}

static void fill_seq(Tensor& t, float start = 1.0f) {
    float* p = static_cast<float*>(t.data());
    for (size_t i = 0; i < t.numElements(); ++i) p[i] = start + static_cast<float>(i);
}
static void fill_val(Tensor& t, float v) {
    float* p = static_cast<float*>(t.data());
    for (size_t i = 0; i < t.numElements(); ++i) p[i] = v;
}

// ---- test 1: SqueezeNet shape ----
static void test_squeezenet_shape() {
    MaxPool2DParams p; p.kH=3; p.kW=3; p.stride_h=2; p.stride_w=2; p.pad_h=0; p.pad_w=0;
    MaxPool2DOp op(p);
    auto out = op.inferOutputShapes({TensorShape({1,64,55,55})});
    // (55+0-3)/2+1 = 27
    ok(out[0].dim(0)==1 && out[0].dim(1)==64 &&
       out[0].dim(2)==27 && out[0].dim(3)==27,
       "Test 1: [1,64,55,55] 3x3/s2/p0 -> [1,64,27,27]");
}

// ---- test 2: known-values 1 channel ----
static void test_known_values() {
    // [1,1,4,4] with sequential values 1..16
    // 3x3/s1/p0 -> [1,1,2,2]
    // window [0:3,0:3] max = 11 (row 2 col 2 = index 10 = val 11)
    // window [0:3,1:4] max = 12
    // window [1:4,0:3] max = 15
    // window [1:4,1:4] max = 16
    MaxPool2DParams p; p.kH=3; p.kW=3; p.stride_h=1; p.stride_w=1; p.pad_h=0; p.pad_w=0;
    MaxPool2DOp op(p);
    TensorShape sin({1,1,4,4});
    auto out_s = op.inferOutputShapes({sin});
    ok(out_s[0].dim(2)==2 && out_s[0].dim(3)==2, "Test 2: 4x4/3x3/s1 -> 2x2");

    Tensor input(sin, DataType::Float32, Device::CPU);
    Tensor output(out_s[0], DataType::Float32, Device::CPU);
    fill_seq(input, 1.0f);  // 1..16 row-major

    op.execute(Backend::CPU, {&input}, {&output});

    const float* d = static_cast<const float*>(output.data());
    ok(d[0] == 11.0f, "Test 2: window[0:3,0:3] max = 11");
    ok(d[1] == 12.0f, "Test 2: window[0:3,1:4] max = 12");
    ok(d[2] == 15.0f, "Test 2: window[1:4,0:3] max = 15");
    ok(d[3] == 16.0f, "Test 2: window[1:4,1:4] max = 16");
}

// ---- test 3: negative values ----
static void test_negative_values() {
    // [1,1,3,3] all negative: -5 except centre -1
    MaxPool2DParams p; p.kH=3; p.kW=3; p.stride_h=1; p.stride_w=1; p.pad_h=1; p.pad_w=1;
    MaxPool2DOp op(p);
    TensorShape sin({1,1,3,3});
    auto out_s = op.inferOutputShapes({sin});
    Tensor input(sin, DataType::Float32, Device::CPU);
    Tensor output(out_s[0], DataType::Float32, Device::CPU);

    fill_val(input, -5.0f);
    static_cast<float*>(input.data())[4] = -1.0f;  // centre = -1

    op.execute(Backend::CPU, {&input}, {&output});

    // Centre output window covers all 9 elements including -1 -> max = -1
    const float* d = static_cast<const float*>(output.data());
    // With 3x3 kernel and pad=1 on a 3x3 input, every output position window
    // includes the centre element (-1), so all 9 outputs should equal -1.
    bool all_neg1 = true;
    for (size_t i = 0; i < output.numElements(); ++i)
        if (d[i] != -1.0f) all_neg1 = false;
    ok(all_neg1, "Test 3: all outputs = -1 (centre visible in every 3x3 window with pad=1)");
}

// ---- test 4: 1x1 kernel stride=1 (identity) ----
static void test_1x1_identity() {
    MaxPool2DParams p; p.kH=1; p.kW=1; p.stride_h=1; p.stride_w=1; p.pad_h=0; p.pad_w=0;
    MaxPool2DOp op(p);
    TensorShape sin({1,2,3,3});
    auto out_s = op.inferOutputShapes({sin});
    ok(out_s[0].dim(2)==3 && out_s[0].dim(3)==3, "Test 4: 1x1 kernel output same size");

    Tensor input(sin, DataType::Float32, Device::CPU);
    Tensor output(out_s[0], DataType::Float32, Device::CPU);
    fill_seq(input, 1.0f);
    op.execute(Backend::CPU, {&input}, {&output});

    const float* in_d  = static_cast<const float*>(input.data());
    const float* out_d = static_cast<const float*>(output.data());
    bool same = true;
    for (size_t i = 0; i < input.numElements(); ++i)
        if (in_d[i] != out_d[i]) same = false;
    ok(same, "Test 4: 1x1/s1 output equals input (identity)");
}

// ---- test 5: multi-channel independent ----
static void test_multi_channel() {
    // [1,3,4,4]: channel 0 = 1.0, channel 1 = 2.0, channel 2 = 3.0
    MaxPool2DParams p; p.kH=2; p.kW=2; p.stride_h=2; p.stride_w=2; p.pad_h=0; p.pad_w=0;
    MaxPool2DOp op(p);
    TensorShape sin({1,3,4,4});
    auto out_s = op.inferOutputShapes({sin});
    ok(out_s[0].dim(1)==3 && out_s[0].dim(2)==2, "Test 5: [1,3,4,4] 2x2/s2 -> [1,3,2,2]");

    Tensor input(sin, DataType::Float32, Device::CPU);
    Tensor output(out_s[0], DataType::Float32, Device::CPU);
    float* d = static_cast<float*>(input.data());
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < 16; ++i)
            d[c*16 + i] = static_cast<float>(c + 1);

    op.execute(Backend::CPU, {&input}, {&output});

    const float* od = static_cast<const float*>(output.data());
    bool ch_ok = true;
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < 4; ++i)
            if (od[c*4 + i] != static_cast<float>(c+1)) ch_ok = false;
    ok(ch_ok, "Test 5: each channel max-pooled independently");
}

// ---- test 6: multi-batch ----
static void test_multi_batch() {
    MaxPool2DParams p; p.kH=2; p.kW=2; p.stride_h=2; p.stride_w=2; p.pad_h=0; p.pad_w=0;
    MaxPool2DOp op(p);
    TensorShape sin({2,1,4,4});
    auto out_s = op.inferOutputShapes({sin});
    ok(out_s[0].dim(0)==2, "Test 6: N=2 preserved");

    Tensor input(sin, DataType::Float32, Device::CPU);
    Tensor output(out_s[0], DataType::Float32, Device::CPU);
    float* d = static_cast<float*>(input.data());
    for (int i = 0; i < 16; ++i) d[i]    = 1.0f;   // batch 0 = 1.0
    for (int i = 0; i < 16; ++i) d[16+i] = 9.0f;   // batch 1 = 9.0

    op.execute(Backend::CPU, {&input}, {&output});

    const float* od = static_cast<const float*>(output.data());
    ok(od[0] == 1.0f, "Test 6: batch 0 max = 1.0");
    ok(od[4] == 9.0f, "Test 6: batch 1 max = 9.0");
}

// ---- test 7: output size formula ----
static void test_output_size_formula() {
    auto check = [](int H, int k, int p, int s, int expected) {
        MaxPool2DParams mp; mp.kH=k; mp.kW=k; mp.stride_h=s; mp.stride_w=s; mp.pad_h=p; mp.pad_w=p;
        MaxPool2DOp op(mp);
        auto out = op.inferOutputShapes({TensorShape({1,1,(size_t)H,(size_t)H})});
        return static_cast<int>(out[0].dim(2)) == expected;
    };
    ok(check(55,3,0,2,27), "Test 7: (55+0-3)/2+1=27  (SqueezeNet MaxPool)");
    ok(check(27,3,0,2,13), "Test 7: (27+0-3)/2+1=13");
    ok(check(13,3,0,2, 6), "Test 7: (13+0-3)/2+1=6");
    ok(check( 4,2,0,2, 2), "Test 7: (4+0-2)/2+1=2");
    ok(check( 5,3,1,1, 5), "Test 7: (5+2-3)/1+1=5");
}

// ---- test 8: error cases ----
static void test_errors() {
    MaxPool2DParams p;
    MaxPool2DOp op(p);
    try {
        op.inferOutputShapes({TensorShape({3,4,4})});
        ok(false, "Test 8: should throw on 3-D input");
    } catch (const std::invalid_argument&) {
        ok(true, "Test 8: throws on 3-D input");
    }
    try {
        TensorShape s({1,1,5,5});
        Tensor t(s, DataType::Float32, Device::CPU);
        Tensor o(TensorShape({1,1,2,2}), DataType::Float32, Device::CPU);
        op.execute(Backend::CPU, {&t, &t}, {&o});
        ok(false, "Test 8: should throw on 2 inputs");
    } catch (const std::invalid_argument&) {
        ok(true, "Test 8: throws on 2 inputs");
    }
}

int main() {
    std::cout << "=== MaxPool2DOp CPU Tests ===\n\n";
    test_squeezenet_shape();
    test_known_values();
    test_negative_values();
    test_1x1_identity();
    test_multi_channel();
    test_multi_batch();
    test_output_size_formula();
    test_errors();
    std::cout << "\n=== All MaxPool2DOp tests PASSED ===\n";
    return 0;
}
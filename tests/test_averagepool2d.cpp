/*
 * test_averagepool2d.cpp  --  AveragePool2DOp correctness tests (CPU)
 *
 * Coverage:
 *  1. SqueezeNet shape: [1,1000,13,13] 13x13/s13/p0 -> [1,1000,1,1]
 *  2. Known values: 2x2 average on 4x4 input with sequential data, hand-verified
 *  3. count_include_pad=true vs false with padding=1
 *  4. When pads=0, count_include_pad true/false give identical results
 *  5. Multi-channel independence
 *  6. Multi-batch independence
 *  7. Output size formula: 4 configs
 *  8. Error: 3-D input throws
 *  9. Error: wrong input count throws
 * 10. Error: non-positive kernel/stride throws at construction
 */

#include "forgert/operator/averagepool2d.h"
#include "forgert/tensor/tensor.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

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
    AvgPool2DParams p; p.kH=13; p.kW=13; p.stride_h=13; p.stride_w=13; p.pad_h=0; p.pad_w=0;
    AveragePool2DOp op(p);
    auto out = op.inferOutputShapes({TensorShape({1,1000,13,13})});
    // (13+0-13)/13+1 = 1
    ok(out[0].dim(0)==1 && out[0].dim(1)==1000 &&
       out[0].dim(2)==1 && out[0].dim(3)==1,
       "Test 1: [1,1000,13,13] 13x13/s13/p0 -> [1,1000,1,1]");
}

// ---- test 2: known values 2x2 on 4x4 ----
static void test_known_values() {
    // [1,1,4,4] with values 1..16. 2x2/s2/p0 -> [1,1,2,2]
    // window[0:2,0:2] = (1+2+5+6)/4 = 14/4 = 3.5
    // window[0:2,2:4] = (3+4+7+8)/4 = 22/4 = 5.5
    // window[2:4,0:2] = (9+10+13+14)/4 = 46/4 = 11.5
    // window[2:4,2:4] = (11+12+15+16)/4 = 54/4 = 13.5
    AvgPool2DParams p; p.kH=2; p.kW=2; p.stride_h=2; p.stride_w=2; p.pad_h=0; p.pad_w=0;
    AveragePool2DOp op(p);
    TensorShape sin({1,1,4,4});
    auto out_s = op.inferOutputShapes({sin});
    ok(out_s[0].dim(2)==2 && out_s[0].dim(3)==2, "Test 2: 4x4/2x2/s2 -> 2x2");

    Tensor input(sin, DataType::Float32, Device::CPU);
    Tensor output(out_s[0], DataType::Float32, Device::CPU);
    fill_seq(input, 1.0f);  // 1..16 row-major

    op.execute(Backend::CPU, {&input}, {&output});

    const float* d = static_cast<const float*>(output.data());
    ok(std::fabs(d[0] - 3.5f)  < 1e-5f, "Test 2: window[0:2,0:2] mean = 3.5");
    ok(std::fabs(d[1] - 5.5f)  < 1e-5f, "Test 2: window[0:2,2:4] mean = 5.5");
    ok(std::fabs(d[2] - 11.5f) < 1e-5f, "Test 2: window[2:4,0:2] mean = 11.5");
    ok(std::fabs(d[3] - 13.5f) < 1e-5f, "Test 2: window[2:4,2:4] mean = 13.5");
}

// ---- test 3: count_include_pad with padding > 0 ----
static void test_count_include_pad() {
    // [1,1,3,3] all 1.0. 2x2/s1/p1.
    // count_include_pad=false: top-left window (0,0) covers in-bounds (0,0),(0,1),(1,0),(1,1) = 4 elements -> 4/4 = 1.0
    // count_include_pad=true:  top-left window = 4 in-bounds / 4 kernel area = 1.0 (same here, pads don't reduce count for this case)
    // Use a case where they differ: [1,1,2,2] all 1.0, 2x2/s1/p1 -> [1,1,3,3]
    // Corner (0,0): in-bounds = 1 element (only (0,0)), sum=1. count_include_pad=false -> 1/1=1.0; true -> 1/4=0.25
    {
        AvgPool2DParams p; p.kH=2; p.kW=2; p.stride_h=1; p.stride_w=1; p.pad_h=1; p.pad_w=1;
        p.count_include_pad = false;
        AveragePool2DOp op_false(p);
        TensorShape sin({1,1,2,2});
        auto out_s = op_false.inferOutputShapes({sin});
        Tensor input(sin, DataType::Float32, Device::CPU);
        Tensor output(out_s[0], DataType::Float32, Device::CPU);
        fill_val(input, 1.0f);
        op_false.execute(Backend::CPU, {&input}, {&output});
        const float* d = static_cast<const float*>(output.data());
        // Corner (0,0): only (0,0) in-bounds -> 1/1 = 1.0
        ok(std::fabs(d[0] - 1.0f) < 1e-5f, "Test 3a: count_include_pad=false corner = 1.0 (1 in-bounds element)");
    }
    {
        AvgPool2DParams p; p.kH=2; p.kW=2; p.stride_h=1; p.stride_w=1; p.pad_h=1; p.pad_w=1;
        p.count_include_pad = true;
        AveragePool2DOp op_true(p);
        TensorShape sin({1,1,2,2});
        auto out_s = op_true.inferOutputShapes({sin});
        Tensor input(sin, DataType::Float32, Device::CPU);
        Tensor output(out_s[0], DataType::Float32, Device::CPU);
        fill_val(input, 1.0f);
        op_true.execute(Backend::CPU, {&input}, {&output});
        const float* d = static_cast<const float*>(output.data());
        // Corner (0,0): 1 in-bounds / 4 kernel area = 0.25
        ok(std::fabs(d[0] - 0.25f) < 1e-5f, "Test 3b: count_include_pad=true corner = 0.25 (1/4)");
    }
}

// ---- test 4: pads=0 gives identical results for both modes ----
static void test_pads_zero_identical() {
    AvgPool2DParams pf; pf.kH=2; pf.kW=2; pf.stride_h=2; pf.stride_w=2; pf.count_include_pad=false;
    AvgPool2DParams pt; pt.kH=2; pt.kW=2; pt.stride_h=2; pt.stride_w=2; pt.count_include_pad=true;
    AveragePool2DOp op_f(pf), op_t(pt);
    TensorShape sin({1,2,4,4});
    auto of = op_f.inferOutputShapes({sin});
    auto ot = op_t.inferOutputShapes({sin});
    Tensor input(sin, DataType::Float32, Device::CPU);
    Tensor out_f(of[0], DataType::Float32, Device::CPU);
    Tensor out_t(ot[0], DataType::Float32, Device::CPU);
    fill_seq(input, 1.0f);
    op_f.execute(Backend::CPU, {&input}, {&out_f});
    op_t.execute(Backend::CPU, {&input}, {&out_t});
    const float* df = static_cast<const float*>(out_f.data());
    const float* dt = static_cast<const float*>(out_t.data());
    bool same = true;
    for (size_t i = 0; i < out_f.numElements(); ++i)
        if (std::fabs(df[i] - dt[i]) > 1e-6f) same = false;
    ok(same, "Test 4: pads=0 -> count_include_pad true/false identical");
}

// ---- test 5: multi-channel independence ----
static void test_multi_channel() {
    // [1,3,4,4]: channel 0 = 1.0, channel 1 = 2.0, channel 2 = 3.0
    AvgPool2DParams p; p.kH=2; p.kW=2; p.stride_h=2; p.stride_w=2;
    AveragePool2DOp op(p);
    TensorShape sin({1,3,4,4});
    auto out_s = op.inferOutputShapes({sin});
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
            if (std::fabs(od[c*4 + i] - static_cast<float>(c+1)) > 1e-5f) ch_ok = false;
    ok(ch_ok, "Test 5: each channel averaged independently");
}

// ---- test 6: multi-batch independence ----
static void test_multi_batch() {
    AvgPool2DParams p; p.kH=2; p.kW=2; p.stride_h=2; p.stride_w=2;
    AveragePool2DOp op(p);
    TensorShape sin({2,1,4,4});
    auto out_s = op.inferOutputShapes({sin});
    ok(out_s[0].dim(0)==2, "Test 6: N=2 preserved");
    Tensor input(sin, DataType::Float32, Device::CPU);
    Tensor output(out_s[0], DataType::Float32, Device::CPU);
    float* d = static_cast<float*>(input.data());
    for (int i = 0; i < 16; ++i) d[i]     = 4.0f;  // batch 0 = 4.0
    for (int i = 0; i < 16; ++i) d[16+i]  = 8.0f;  // batch 1 = 8.0
    op.execute(Backend::CPU, {&input}, {&output});
    const float* od = static_cast<const float*>(output.data());
    ok(std::fabs(od[0] - 4.0f) < 1e-5f, "Test 6: batch 0 mean = 4.0");
    ok(std::fabs(od[4] - 8.0f) < 1e-5f, "Test 6: batch 1 mean = 8.0");
}

// ---- test 7: output size formula ----
static void test_output_size_formula() {
    auto check = [](int H, int k, int p, int s, int expected) {
        AvgPool2DParams ap; ap.kH=k; ap.kW=k; ap.stride_h=s; ap.stride_w=s; ap.pad_h=p; ap.pad_w=p;
        AveragePool2DOp op(ap);
        auto out = op.inferOutputShapes({TensorShape({1,1,(size_t)H,(size_t)H})});
        return static_cast<int>(out[0].dim(2)) == expected;
    };
    ok(check(13,13,0,13, 1), "Test 7: (13+0-13)/13+1=1  (SqueezeNet AveragePool)");
    ok(check(55, 3,0, 2,27), "Test 7: (55+0-3)/2+1=27");
    ok(check( 4, 2,0, 2, 2), "Test 7: (4+0-2)/2+1=2");
    ok(check( 5, 3,1, 1, 5), "Test 7: (5+2-3)/1+1=5");
}

// ---- test 8: error - 3-D input ----
static void test_error_3d_input() {
    AvgPool2DParams p; p.kH=2; p.kW=2; p.stride_h=2; p.stride_w=2;
    AveragePool2DOp op(p);
    try {
        op.inferOutputShapes({TensorShape({3,4,4})});
        ok(false, "Test 8: should throw on 3-D input");
    } catch (const std::invalid_argument&) {
        ok(true, "Test 8: throws on 3-D input");
    }
}

// ---- test 9: error - wrong input count ----
static void test_error_wrong_count() {
    AvgPool2DParams p; p.kH=2; p.kW=2; p.stride_h=2; p.stride_w=2;
    AveragePool2DOp op(p);
    try {
        op.inferOutputShapes({TensorShape({1,1,4,4}), TensorShape({1,1,4,4})});
        ok(false, "Test 9: should throw on 2 inputs");
    } catch (const std::invalid_argument&) {
        ok(true, "Test 9: throws on 2 inputs");
    }
}

// ---- test 10: error - non-positive kernel/stride at construction ----
static void test_error_invalid_params() {
    try { AvgPool2DParams p; p.kH=0; AveragePool2DOp op(p); ok(false, "Test 10a: should throw on kH=0"); }
    catch (const std::invalid_argument&) { ok(true, "Test 10a: throws on kH=0"); }
    try { AvgPool2DParams p; p.stride_h=0; AveragePool2DOp op(p); ok(false, "Test 10b: should throw on stride_h=0"); }
    catch (const std::invalid_argument&) { ok(true, "Test 10b: throws on stride_h=0"); }
    try { AvgPool2DParams p; p.pad_h=-1; AveragePool2DOp op(p); ok(false, "Test 10c: should throw on pad_h=-1"); }
    catch (const std::invalid_argument&) { ok(true, "Test 10c: throws on pad_h=-1"); }
}

int main() {
    std::cout << "=== AveragePool2DOp CPU Tests ===\n\n";
    test_squeezenet_shape();
    test_known_values();
    test_count_include_pad();
    test_pads_zero_identical();
    test_multi_channel();
    test_multi_batch();
    test_output_size_formula();
    test_error_3d_input();
    test_error_wrong_count();
    test_error_invalid_params();
    std::cout << "\n=== All AveragePool2DOp tests PASSED ===\n";
    return 0;
}
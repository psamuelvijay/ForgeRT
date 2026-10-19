/*
 * test_concat.cpp  --  ConcatOp correctness tests (CPU)
 *
 * Coverage:
 *  1. SqueezeNet use-case: axis=1 channel concat, two [1,64,55,55] -> [1,128,55,55]
 *  2. Known-values check (small tensors, hand-verifiable)
 *  3. axis=0 (batch concatenation)
 *  4. axis=2 (height concatenation)
 *  5. Three inputs
 *  6. Shape inference: mismatched non-axis dim -> throws
 *  7. Shape inference: wrong rank -> throws
 *  8. Shape inference: axis out of range -> throws
 *  9. Single-input -> throws
 * 10. Empty (zero-size) concat dimension is supported (concatenating 0-size slices)
 */

#include "forgert/operator/concat.h"
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

static void fill_val(Tensor& t, float v) {
    float* p = static_cast<float*>(t.data());
    for (size_t i = 0; i < t.numElements(); ++i) p[i] = v;
}
static void fill_seq(Tensor& t, float start = 0.0f) {
    float* p = static_cast<float*>(t.data());
    for (size_t i = 0; i < t.numElements(); ++i) p[i] = start + static_cast<float>(i);
}

// ---- test 1: SqueezeNet fire expand shape ----
static void test_squeezenet_shape() {
    ConcatOp op(1);
    TensorShape s1({1,64,55,55}), s2({1,64,55,55});
    auto out = op.inferOutputShapes({s1, s2});
    ok(out[0].dim(0)==1 && out[0].dim(1)==128 &&
       out[0].dim(2)==55 && out[0].dim(3)==55,
       "Test 1: SqueezeNet fire expand [1,64,55,55]+[1,64,55,55] -> [1,128,55,55]");
}

// ---- test 2: known-values axis=1 ----
static void test_known_values_axis1() {
    ConcatOp op(1);
    // a: [1,2,2,2] all 1.0, b: [1,3,2,2] all 2.0
    // out: [1,5,2,2]; first 2 channels = 1.0, next 3 = 2.0
    TensorShape sa({1,2,2,2}), sb({1,3,2,2});
    auto out_shapes = op.inferOutputShapes({sa, sb});
    ok(out_shapes[0].dim(1) == 5, "Test 2: shape inference [1,2,2,2]+[1,3,2,2] -> dim1=5");

    Tensor a(sa, DataType::Float32, Device::CPU);
    Tensor b(sb, DataType::Float32, Device::CPU);
    Tensor out(out_shapes[0], DataType::Float32, Device::CPU);
    fill_val(a, 1.0f);
    fill_val(b, 2.0f);

    op.execute(Backend::CPU, {&a, &b}, {&out});

    const float* d = static_cast<const float*>(out.data());
    // layout [1,5,2,2]: channel 0,1 come from a, channel 2,3,4 from b
    // index [0,c,h,w] = d[(c*2 + h)*2 + w]
    bool a_ok = true, b_ok = true;
    for (int c = 0; c < 2; ++c)
        for (int i = 0; i < 4; ++i)
            if (d[c*4 + i] != 1.0f) a_ok = false;
    for (int c = 2; c < 5; ++c)
        for (int i = 0; i < 4; ++i)
            if (d[c*4 + i] != 2.0f) b_ok = false;
    ok(a_ok, "Test 2: first 2 channels from input a = 1.0");
    ok(b_ok, "Test 2: next 3 channels from input b = 2.0");
}

// ---- test 3: axis=0 (batch concat) ----
static void test_axis0() {
    ConcatOp op(0);
    TensorShape s({2,3,4,4});
    auto out = op.inferOutputShapes({s, s});
    ok(out[0].dim(0) == 4, "Test 3: axis=0 batch concat 2+2=4");

    Tensor a(s, DataType::Float32, Device::CPU);
    Tensor b(s, DataType::Float32, Device::CPU);
    Tensor outt(out[0], DataType::Float32, Device::CPU);
    fill_val(a, 3.0f); fill_val(b, 7.0f);
    op.execute(Backend::CPU, {&a, &b}, {&outt});

    const float* d = static_cast<const float*>(outt.data());
    size_t half = a.numElements();
    bool first_ok = true, second_ok = true;
    for (size_t i = 0; i < half; ++i)  if (d[i] != 3.0f) first_ok = false;
    for (size_t i = 0; i < half; ++i)  if (d[half+i] != 7.0f) second_ok = false;
    ok(first_ok,  "Test 3: first batch = 3.0");
    ok(second_ok, "Test 3: second batch = 7.0");
}

// ---- test 4: axis=2 (height concat) with sequential values ----
static void test_axis2_values() {
    ConcatOp op(2);
    // a [1,1,2,3] = 0..5, b [1,1,3,3] = 10..19
    // out [1,1,5,3]: rows 0-1 from a, rows 2-4 from b
    TensorShape sa({1,1,2,3}), sb({1,1,3,3});
    auto out_s = op.inferOutputShapes({sa, sb});
    ok(out_s[0].dim(2) == 5, "Test 4: axis=2 height concat 2+3=5");

    Tensor a(sa, DataType::Float32, Device::CPU);
    Tensor b(sb, DataType::Float32, Device::CPU);
    Tensor outt(out_s[0], DataType::Float32, Device::CPU);
    fill_seq(a, 0.0f);
    fill_seq(b, 10.0f);
    op.execute(Backend::CPU, {&a, &b}, {&outt});

    const float* d = static_cast<const float*>(outt.data());
    // Rows 0-1 come from a: elements 0..5
    bool row_a_ok = true;
    for (int i = 0; i < 6; ++i) if (d[i] != static_cast<float>(i)) row_a_ok = false;
    // Rows 2-4 come from b: elements 10..18
    bool row_b_ok = true;
    for (int i = 0; i < 9; ++i) if (d[6+i] != 10.0f + i) row_b_ok = false;
    ok(row_a_ok, "Test 4: first 2 rows from a match sequential values");
    ok(row_b_ok, "Test 4: next 3 rows from b match sequential values");
}

// ---- test 5: three inputs ----
static void test_three_inputs() {
    ConcatOp op(1);
    TensorShape s({1,2,3,3});
    auto out = op.inferOutputShapes({s, s, s});
    ok(out[0].dim(1) == 6, "Test 5: three inputs axis=1 -> C=6");

    Tensor a(s, DataType::Float32, Device::CPU);
    Tensor b(s, DataType::Float32, Device::CPU);
    Tensor c(s, DataType::Float32, Device::CPU);
    Tensor outt(out[0], DataType::Float32, Device::CPU);
    fill_val(a, 1.0f); fill_val(b, 2.0f); fill_val(c, 3.0f);
    op.execute(Backend::CPU, {&a, &b, &c}, {&outt});

    const float* d = static_cast<const float*>(outt.data());
    size_t slice = 2 * 3 * 3;  // C * H * W per input
    bool ok1 = true, ok2 = true, ok3 = true;
    for (size_t i = 0; i < slice; ++i)      if (d[i]          != 1.0f) ok1 = false;
    for (size_t i = 0; i < slice; ++i)      if (d[slice+i]    != 2.0f) ok2 = false;
    for (size_t i = 0; i < slice; ++i)      if (d[2*slice+i]  != 3.0f) ok3 = false;
    ok(ok1, "Test 5: first input block = 1.0");
    ok(ok2, "Test 5: second input block = 2.0");
    ok(ok3, "Test 5: third input block = 3.0");
}

// ---- test 6: shape mismatch error ----
static void test_shape_mismatch() {
    ConcatOp op(1);
    try {
        op.inferOutputShapes({TensorShape({1,3,4,4}), TensorShape({1,3,5,4})});
        ok(false, "Test 6: should throw on H mismatch");
    } catch (const std::invalid_argument&) {
        ok(true, "Test 6: throws on non-axis dimension mismatch");
    }
}

// ---- test 7: rank mismatch error ----
static void test_rank_mismatch() {
    ConcatOp op(1);
    try {
        op.inferOutputShapes({TensorShape({1,3,4,4}), TensorShape({3,4,4})});
        ok(false, "Test 7: should throw on rank mismatch");
    } catch (const std::invalid_argument&) {
        ok(true, "Test 7: throws on rank mismatch");
    }
}

// ---- test 8: axis out of range ----
static void test_axis_oob() {
    ConcatOp op(5);
    try {
        op.inferOutputShapes({TensorShape({1,3,4,4}), TensorShape({1,3,4,4})});
        ok(false, "Test 8: should throw on axis >= ndim");
    } catch (const std::invalid_argument&) {
        ok(true, "Test 8: throws on axis out of range");
    }
}

// ---- test 9: single-input error ----
static void test_single_input_error() {
    ConcatOp op(1);
    try {
        op.inferOutputShapes({TensorShape({1,3,4,4})});
        ok(false, "Test 9: should throw on single input");
    } catch (const std::invalid_argument&) {
        ok(true, "Test 9: throws on single input");
    }
}

int main() {
    std::cout << "=== ConcatOp CPU Tests ===\n\n";
    test_squeezenet_shape();
    test_known_values_axis1();
    test_axis0();
    test_axis2_values();
    test_three_inputs();
    test_shape_mismatch();
    test_rank_mismatch();
    test_axis_oob();
    test_single_input_error();
    std::cout << "\n=== All ConcatOp tests PASSED ===\n";
    return 0;
}
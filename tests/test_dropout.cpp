/*
 * test_dropout.cpp  --  DropoutOp correctness tests (CPU)
 *
 * Coverage:
 *  1. Byte-for-byte identity on 1-D [5] with sequential values
 *  2. Byte-for-byte identity on 2-D [3,4]
 *  3. Byte-for-byte identity on 4-D [1,3,4,4]
 *  4. Random non-trivial values — no zeroing, no scaling
 *  5. ratio=0.0, 0.5, 0.9 all produce identical output (ratio has no effect)
 *  6. Shape preservation across 1D/2D/4D
 *  7. Error: wrong input count throws
 *  8. Error: wrong output count throws
 */

#include "forgert/operator/dropout.h"
#include "forgert/tensor/tensor.h"

#include <cstring>
#include <iostream>
#include <random>
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

// ---- test 1: 1-D identity ----
static void test_1d_identity() {
    DropoutOp op;
    TensorShape s({5});
    auto out_s = op.inferOutputShapes({s});
    Tensor input(s, DataType::Float32, Device::CPU);
    Tensor output(out_s[0], DataType::Float32, Device::CPU);
    fill_seq(input, 1.0f);
    op.execute(Backend::CPU, {&input}, {&output});
    ok(std::memcmp(input.data(), output.data(), input.numBytes()) == 0,
       "Test 1: 1-D [5] byte-identical identity");
}

// ---- test 2: 2-D identity ----
static void test_2d_identity() {
    DropoutOp op;
    TensorShape s({3,4});
    Tensor input(s, DataType::Float32, Device::CPU);
    auto out_s = op.inferOutputShapes({s});
    Tensor output(out_s[0], DataType::Float32, Device::CPU);
    fill_seq(input, 0.5f);
    op.execute(Backend::CPU, {&input}, {&output});
    ok(std::memcmp(input.data(), output.data(), input.numBytes()) == 0,
       "Test 2: 2-D [3,4] byte-identical identity");
}

// ---- test 3: 4-D identity ----
static void test_4d_identity() {
    DropoutOp op;
    TensorShape s({1,3,4,4});
    Tensor input(s, DataType::Float32, Device::CPU);
    auto out_s = op.inferOutputShapes({s});
    Tensor output(out_s[0], DataType::Float32, Device::CPU);
    fill_seq(input, -2.0f);
    op.execute(Backend::CPU, {&input}, {&output});
    ok(std::memcmp(input.data(), output.data(), input.numBytes()) == 0,
       "Test 3: 4-D [1,3,4,4] byte-identical identity");
}

// ---- test 4: random values, no zeroing or scaling ----
static void test_random_values() {
    DropoutOp op;
    TensorShape s({1,1000});
    Tensor input(s, DataType::Float32, Device::CPU);
    auto out_s = op.inferOutputShapes({s});
    Tensor output(out_s[0], DataType::Float32, Device::CPU);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    float* in = static_cast<float*>(input.data());
    for (size_t i = 0; i < input.numElements(); ++i) in[i] = dist(rng);
    op.execute(Backend::CPU, {&input}, {&output});
    const float* out = static_cast<const float*>(output.data());
    bool all_match = true;
    for (size_t i = 0; i < input.numElements(); ++i)
        if (in[i] != out[i]) all_match = false;  // exact equality — memcpy preserves bits
    ok(all_match, "Test 4: random values preserved exactly (no zeroing/scaling)");
}

// ---- test 5: ratio has no effect ----
static void test_ratio_no_effect() {
    TensorShape s({1,50});
    Tensor input(s, DataType::Float32, Device::CPU);
    fill_seq(input, 0.1f);

    DropoutParams p0; p0.ratio = 0.0f;
    DropoutParams p5; p5.ratio = 0.5f;
    DropoutParams p9; p9.ratio = 0.9f;
    DropoutOp op0(p0), op5(p5), op9(p9);

    auto out_s = op5.inferOutputShapes({s});
    Tensor out0(out_s[0], DataType::Float32, Device::CPU);
    Tensor out5(out_s[0], DataType::Float32, Device::CPU);
    Tensor out9(out_s[0], DataType::Float32, Device::CPU);
    op0.execute(Backend::CPU, {&input}, {&out0});
    op5.execute(Backend::CPU, {&input}, {&out5});
    op9.execute(Backend::CPU, {&input}, {&out9});

    bool same_05 = std::memcmp(out0.data(), out5.data(), out0.numBytes()) == 0;
    bool same_09 = std::memcmp(out0.data(), out9.data(), out0.numBytes()) == 0;
    ok(same_05, "Test 5: ratio=0.0 and ratio=0.5 produce identical output");
    ok(same_09, "Test 5: ratio=0.0 and ratio=0.9 produce identical output");
}

// ---- test 6: shape preservation ----
static void test_shape_preservation() {
    DropoutOp op;
    for (auto& s : std::vector<TensorShape>{TensorShape({5}), TensorShape({3,4}),
                                              TensorShape({1,3,4,4})}) {
        auto out = op.inferOutputShapes({s});
        ok(out[0].ndim() == s.ndim() && out[0].numElements() == s.numElements(),
           "Test 6: shape preserved");
    }
}

// ---- test 7: error - wrong input count ----
static void test_error_wrong_input_count() {
    DropoutOp op;
    try {
        op.inferOutputShapes({TensorShape({5}), TensorShape({5})});
        ok(false, "Test 7: should throw on 2 inputs");
    } catch (const std::invalid_argument&) {
        ok(true, "Test 7: throws on 2 inputs");
    }
}

// ---- test 8: error - wrong output count at execute ----
static void test_error_wrong_output_count() {
    DropoutOp op;
    TensorShape s({5});
    Tensor input(s, DataType::Float32, Device::CPU);
    Tensor dummy(TensorShape({3}), DataType::Float32, Device::CPU);
    try {
        op.execute(Backend::CPU, {&input}, {&dummy});
        ok(false, "Test 8: should throw on shape mismatch");
    } catch (const std::invalid_argument&) {
        ok(true, "Test 8: throws on output shape mismatch");
    }
}

int main() {
    std::cout << "=== DropoutOp CPU Tests ===\n\n";
    test_1d_identity();
    test_2d_identity();
    test_4d_identity();
    test_random_values();
    test_ratio_no_effect();
    test_shape_preservation();
    test_error_wrong_input_count();
    test_error_wrong_output_count();
    std::cout << "\n=== All DropoutOp tests PASSED ===\n";
    return 0;
}
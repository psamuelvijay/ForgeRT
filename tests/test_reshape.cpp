/*
 * test_reshape.cpp  --  ReshapeOp correctness tests (CPU)
 *
 * Coverage:
 *  1. SqueezeNet case: [1,1000,1,1] + shape [0,-1] -> 1000 elements, data preserved
 *  2. Wildcard 0 only: [2,3,4] + [0,2,6] -> 12 elements
 *  3. Wildcard -1 only: [2,3,4] + [2,-1] -> 12 elements
 *  4. Both wildcards: [2,3,4] + [0,-1] -> 12 elements
 *  5. Literal shape: [6] + [2,3] -> 6 elements
 *  6. Data bytes unchanged (memcmp)
 *  7. Error: more than one -1 -> throws
 *  8. Error: 0 at dim i where input has no dim i -> throws
 *  9. Error: element count mismatch -> throws
 * 10. Error: wrong input count -> throws
 * 11. Error: shape tensor not Int32 -> throws (via validateDtype)
 * 12. Error: shape tensor not 1-D -> throws
 */

#include "forgert/operator/reshape.h"
#include "forgert/tensor/tensor.h"
#include "forgert/graph/graph.h"

#include <cstdint>
#include <cstring>
#include <iostream>
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

// Helper: create an Int32 shape tensor from a vector of dims
static Tensor make_shape_tensor(const std::vector<int32_t>& dims) {
    TensorShape s({dims.size()});
    Tensor t(s, DataType::Int32, Device::CPU);
    int32_t* p = static_cast<int32_t*>(t.data());
    for (size_t i = 0; i < dims.size(); ++i) p[i] = dims[i];
    return t;
}

// ---- test 1: SqueezeNet case [1,1000,1,1] + [0,-1] ----
static void test_squeezenet_case() {
    ReshapeOp op;
    TensorShape sd({1,1000,1,1});
    Tensor data(sd, DataType::Float32, Device::CPU);
    fill_seq(data, 1.0f);
    Tensor shape = make_shape_tensor({0, -1});

    // Output pre-allocated with resolved shape [1,1000] (1000 elements)
    TensorShape so({1,1000});
    Tensor output(so, DataType::Float32, Device::CPU);

    op.execute(Backend::CPU, {&data, &shape}, {&output});

    ok(output.numElements() == 1000, "Test 1: output has 1000 elements");
    ok(std::memcmp(data.data(), output.data(), data.numBytes()) == 0,
       "Test 1: data bytes preserved (byte-identical memcpy)");
}

// ---- test 2: wildcard 0 only [2,3,4] + [0,2,6] ----
static void test_wildcard_zero() {
    ReshapeOp op;
    TensorShape sd({2,3,4});
    Tensor data(sd, DataType::Float32, Device::CPU);
    fill_seq(data, 1.0f);
    Tensor shape = make_shape_tensor({0, 2, 6});

    // Resolved: dim0=copy input dim0=2, dim1=2, dim2=6 -> [2,2,6] = 24 elements
    TensorShape so({2,2,6});
    Tensor output(so, DataType::Float32, Device::CPU);

    op.execute(Backend::CPU, {&data, &shape}, {&output});

    ok(output.numElements() == 24, "Test 2: [2,3,4]+[0,2,6] -> 24 elements");
    ok(std::memcmp(data.data(), output.data(), data.numBytes()) == 0,
       "Test 2: data bytes preserved");
}

// ---- test 3: wildcard -1 only [2,3,4] + [2,-1] ----
static void test_wildcard_neg_one() {
    ReshapeOp op;
    TensorShape sd({2,3,4});
    Tensor data(sd, DataType::Float32, Device::CPU);
    fill_seq(data, 1.0f);
    Tensor shape = make_shape_tensor({2, -1});

    // Resolved: dim0=2, dim1=24/2=12 -> [2,12] = 24 elements
    TensorShape so({2,12});
    Tensor output(so, DataType::Float32, Device::CPU);

    op.execute(Backend::CPU, {&data, &shape}, {&output});

    ok(output.numElements() == 24, "Test 3: [2,3,4]+[2,-1] -> 24 elements");
    ok(std::memcmp(data.data(), output.data(), data.numBytes()) == 0,
       "Test 3: data bytes preserved");
}

// ---- test 4: both wildcards [2,3,4] + [0,-1] ----
static void test_both_wildcards() {
    ReshapeOp op;
    TensorShape sd({2,3,4});
    Tensor data(sd, DataType::Float32, Device::CPU);
    fill_seq(data, 1.0f);
    Tensor shape = make_shape_tensor({0, -1});

    // Resolved: dim0=copy input dim0=2, dim1=24/2=12 -> [2,12] = 24 elements
    TensorShape so({2,12});
    Tensor output(so, DataType::Float32, Device::CPU);

    op.execute(Backend::CPU, {&data, &shape}, {&output});

    ok(output.numElements() == 24, "Test 4: [2,3,4]+[0,-1] -> 24 elements");
    ok(std::memcmp(data.data(), output.data(), data.numBytes()) == 0,
       "Test 4: data bytes preserved");
}

// ---- test 5: literal shape [6] + [2,3] ----
static void test_literal_shape() {
    ReshapeOp op;
    TensorShape sd({6});
    Tensor data(sd, DataType::Float32, Device::CPU);
    fill_seq(data, 1.0f);
    Tensor shape = make_shape_tensor({2, 3});

    TensorShape so({2,3});
    Tensor output(so, DataType::Float32, Device::CPU);

    op.execute(Backend::CPU, {&data, &shape}, {&output});

    ok(output.numElements() == 6, "Test 5: [6]+[2,3] -> 6 elements");
    ok(std::memcmp(data.data(), output.data(), data.numBytes()) == 0,
       "Test 5: data bytes preserved");
}

// ---- test 6: data bytes unchanged with random-ish values ----
static void test_data_unchanged() {
    ReshapeOp op;
    TensorShape sd({2,3,4});
    Tensor data(sd, DataType::Float32, Device::CPU);
    float* d = static_cast<float*>(data.data());
    for (size_t i = 0; i < data.numElements(); ++i)
        d[i] = static_cast<float>(i) * 0.7f - 1.3f;
    Tensor shape = make_shape_tensor({-1});

    TensorShape so({24});
    Tensor output(so, DataType::Float32, Device::CPU);

    op.execute(Backend::CPU, {&data, &shape}, {&output});

    bool all_same = true;
    const float* in = static_cast<const float*>(data.data());
    const float* out = static_cast<const float*>(output.data());
    for (size_t i = 0; i < data.numElements(); ++i)
        if (in[i] != out[i]) all_same = false;
    ok(all_same, "Test 6: all data values unchanged after reshape");
}

// ---- test 7: error - more than one -1 ----
static void test_error_multiple_neg_one() {
    ReshapeOp op;
    TensorShape sd({6});
    Tensor data(sd, DataType::Float32, Device::CPU);
    Tensor shape = make_shape_tensor({-1, -1});
    TensorShape so({6});
    Tensor output(so, DataType::Float32, Device::CPU);
    try {
        op.execute(Backend::CPU, {&data, &shape}, {&output});
        ok(false, "Test 7: should throw on multiple -1");
    } catch (const std::invalid_argument&) {
        ok(true, "Test 7: throws on multiple -1");
    }
}

// ---- test 8: error - 0 at dim i where input has no dim i ----
static void test_error_zero_no_match() {
    ReshapeOp op;
    TensorShape sd({5});  // input is 1-D
    Tensor data(sd, DataType::Float32, Device::CPU);
    fill_val(data, 1.0f);
    Tensor shape = make_shape_tensor({0, 5});  // 0 at dim 1, but input is 1-D
    TensorShape so({5});
    Tensor output(so, DataType::Float32, Device::CPU);
    try {
        op.execute(Backend::CPU, {&data, &shape}, {&output});
        ok(false, "Test 8: should throw on 0 with no matching input dim");
    } catch (const std::invalid_argument&) {
        ok(true, "Test 8: throws on 0 at dim beyond input rank");
    }
}

// ---- test 9: error - element count mismatch ----
static void test_error_count_mismatch() {
    ReshapeOp op;
    TensorShape sd({6});
    Tensor data(sd, DataType::Float32, Device::CPU);
    fill_val(data, 1.0f);
    Tensor shape = make_shape_tensor({2, 4});  // product 8 != 6
    TensorShape so({8});
    Tensor output(so, DataType::Float32, Device::CPU);
    try {
        op.execute(Backend::CPU, {&data, &shape}, {&output});
        ok(false, "Test 9: should throw on element count mismatch");
    } catch (const std::invalid_argument&) {
        ok(true, "Test 9: throws on element count mismatch (product 8 != 6)");
    }
}

// ---- test 10: error - wrong input count ----
static void test_error_wrong_count() {
    ReshapeOp op;
    try {
        op.inferOutputShapes({TensorShape({6})});  // only 1 input
        ok(false, "Test 10: should throw on single input");
    } catch (const std::invalid_argument&) {
        ok(true, "Test 10: throws on single input");
    }
}

// ---- test 11: error - shape tensor not Int32 ----
static void test_error_shape_not_int32() {
    ReshapeOp op;
    TensorShape sd({6});
    Tensor data(sd, DataType::Float32, Device::CPU);
    fill_val(data, 1.0f);
    // Create a Float32 shape tensor (wrong dtype)
    TensorShape ss({2});
    Tensor shape(ss, DataType::Float32, Device::CPU);  // Float32, not Int32
    TensorShape so({6});
    Tensor output(so, DataType::Float32, Device::CPU);
    try {
        op.execute(Backend::CPU, {&data, &shape}, {&output});
        ok(false, "Test 11: should throw on Float32 shape tensor");
    } catch (const std::invalid_argument&) {
        ok(true, "Test 11: throws on non-Int32 shape tensor");
    }
}

// ---- test 12: error - shape tensor not 1-D ----
static void test_error_shape_not_1d() {
    ReshapeOp op;
    TensorShape sd({6});
    Tensor data(sd, DataType::Float32, Device::CPU);
    fill_val(data, 1.0f);
    // Create a 2-D Int32 shape tensor
    TensorShape ss({1, 2});
    Tensor shape(ss, DataType::Int32, Device::CPU);
    int32_t* sp = static_cast<int32_t*>(shape.data());
    sp[0] = 2; sp[1] = 3;
    TensorShape so({6});
    Tensor output(so, DataType::Float32, Device::CPU);
    try {
        op.execute(Backend::CPU, {&data, &shape}, {&output});
        ok(false, "Test 12: should throw on 2-D shape tensor");
    } catch (const std::invalid_argument&) {
        ok(true, "Test 12: throws on non-1-D shape tensor");
    }
}

// ======================================================================
// Fixed-target mode (used by the ONNX importer)
// ======================================================================

static bool dimsEq(const TensorShape& s, std::vector<size_t> d) { return s.dimensions() == d; }

// ---- test 13: fixed target resolves exact output shape ----
static void test_fixed_exact_shape() {
    ReshapeOp op(std::vector<int64_t>{0, -1});
    auto out = op.inferOutputShapes({TensorShape({1,1000,1,1})});
    ok(dimsEq(out[0], {1,1000}), "Test 13: fixed [0,-1] on [1,1000,1,1] infers exactly [1,1000]");
    ReshapeOp op2(std::vector<int64_t>{0, 2, 6});
    ok(dimsEq(op2.inferOutputShapes({TensorShape({2,3,4})})[0], {2,2,6}),
       "Test 13: fixed [0,2,6] on [2,3,4] infers exactly [2,2,6]");
    ReshapeOp op3(std::vector<int64_t>{2, -1});
    ok(dimsEq(op3.inferOutputShapes({TensorShape({2,3,4})})[0], {2,12}),
       "Test 13: fixed [2,-1] on [2,3,4] infers exactly [2,12]");
}

// ---- test 14: both modes agree on resolution and data ----
static void test_modes_agree() {
    TensorShape sd({2,3,4});
    Tensor data(sd, DataType::Float32, Device::CPU);
    fill_seq(data, 0.25f);
    ReshapeOp fixed(std::vector<int64_t>{0, -1});
    TensorShape resolved = fixed.inferOutputShapes({sd})[0];
    Tensor out_fixed(resolved, DataType::Float32, Device::CPU);
    fixed.execute(Backend::CPU, {&data}, {&out_fixed});

    ReshapeOp dyn;
    Tensor shape = make_shape_tensor({0, -1});
    Tensor out_dyn(resolved, DataType::Float32, Device::CPU);
    dyn.execute(Backend::CPU, {&data, &shape}, {&out_dyn});

    ok(dimsEq(resolved, {2,12}), "Test 14: fixed mode resolves [0,-1] on [2,3,4] to [2,12]");
    ok(std::memcmp(out_fixed.data(), out_dyn.data(), data.numBytes()) == 0 &&
       std::memcmp(out_fixed.data(), data.data(), data.numBytes()) == 0,
       "Test 14: fixed and shape-tensor modes produce identical bytes");
}

// ---- test 15: Graph allocates the exact resolved shape ----
static void test_fixed_in_graph() {
    Graph g;
    size_t in = g.addInput(TensorShape({1,1000,1,1}), DataType::Float32, "x");
    auto outs = g.addNode(std::make_unique<ReshapeOp>(std::vector<int64_t>{0, -1}), {in}, "flatten");
    g.markOutput(outs[0]);
    g.validate();
    ok(dimsEq(g.getValueShape(outs[0]), {1,1000}), "Test 15: graph value shape is exactly [1,1000]");

    Tensor x(TensorShape({1,1000,1,1}), DataType::Float32, Device::CPU);
    fill_seq(x, -3.0f);
    auto res = g.execute({&x}, Backend::CPU);
    ok(dimsEq(res[0]->shape(), {1,1000}), "Test 15: executed output tensor shape is exactly [1,1000]");
    ok(std::memcmp(res[0]->data(), x.data(), x.numBytes()) == 0, "Test 15: graph output bytes unchanged");
}

// ---- test 16: fixed-mode errors ----
static void test_fixed_errors() {
    auto throws = [](auto fn) { try { fn(); return false; } catch (const std::invalid_argument&) { return true; } };
    ok(throws([]{ ReshapeOp op(std::vector<int64_t>{2, -2}); }), "Test 16: value -2 rejected at construction");
    ok(throws([]{ ReshapeOp op(std::vector<int64_t>{}); }), "Test 16: empty target rejected at construction");
    ok(throws([]{ ReshapeOp op(std::vector<int64_t>{-1, -1}); op.inferOutputShapes({TensorShape({6})}); }),
       "Test 16: two -1 rejected in shape inference");
    ok(throws([]{ ReshapeOp op(std::vector<int64_t>{0, 5}); op.inferOutputShapes({TensorShape({5})}); }),
       "Test 16: 0 beyond input rank rejected");
    ok(throws([]{ ReshapeOp op(std::vector<int64_t>{2, 4}); op.inferOutputShapes({TensorShape({6})}); }),
       "Test 16: element-count mismatch rejected");
    ok(throws([]{ ReshapeOp op(std::vector<int64_t>{4, -1}); op.inferOutputShapes({TensorShape({6})}); }),
       "Test 16: -1 with non-dividing known dims rejected");
    ok(throws([]{ ReshapeOp op(std::vector<int64_t>{0, -1});
                  op.inferOutputShapes({TensorShape({6}), TensorShape({2})}); }),
       "Test 16: fixed mode rejects a second input");
    ok(throws([]{ ReshapeOp op(std::vector<int64_t>{0, -1});
                  Tensor d(TensorShape({2,3}), DataType::Float32, Device::CPU);
                  Tensor o(TensorShape({3,2}), DataType::Float32, Device::CPU);
                  op.executeCPU({&d}, {&o}); }),
       "Test 16: output tensor with wrong exact shape rejected");
    ReshapeOp src(std::vector<int64_t>{0, -1});
    auto c = src.clone();
    ok(dimsEq(c->inferOutputShapes({TensorShape({1,1000,1,1})})[0], {1,1000}),
       "Test 16: clone preserves the fixed target");
}

int main() {
    std::cout << "=== ReshapeOp CPU Tests ===\n\n";
    test_squeezenet_case();
    test_wildcard_zero();
    test_wildcard_neg_one();
    test_both_wildcards();
    test_literal_shape();
    test_data_unchanged();
    test_error_multiple_neg_one();
    test_error_zero_no_match();
    test_error_count_mismatch();
    test_error_wrong_count();
    test_error_shape_not_int32();
    test_error_shape_not_1d();
    test_fixed_exact_shape();
    test_modes_agree();
    test_fixed_in_graph();
    test_fixed_errors();
    std::cout << "\n=== All ReshapeOp tests PASSED ===\n";
    return 0;
}
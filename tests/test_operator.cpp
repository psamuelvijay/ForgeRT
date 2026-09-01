#include "forgert/operator/add.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

using namespace forgert;

void test_add_operator_basic() {
    std::cout << "Testing AddOp basic operation..." << std::endl;

    // Create two 1D tensors: [1, 2, 3] + [4, 5, 6] = [5, 7, 9]
    TensorShape shape({3});
    Tensor a(shape, DataType::Float32, Device::CPU);
    Tensor b(shape, DataType::Float32, Device::CPU);
    Tensor c(shape, DataType::Float32, Device::CPU);

    float* a_data = static_cast<float*>(a.data());
    float* b_data = static_cast<float*>(b.data());
    
    a_data[0] = 1.0f; a_data[1] = 2.0f; a_data[2] = 3.0f;
    b_data[0] = 4.0f; b_data[1] = 5.0f; b_data[2] = 6.0f;

    AddOp add_op;
    assert(add_op.name() == "Add");
    assert(add_op.supportsBackend(Backend::CPU));
    assert(!add_op.supportsBackend(Backend::CUDA));

    add_op.execute(Backend::CPU, {&a, &b}, {&c});

    float* c_data = static_cast<float*>(c.data());
    assert(std::abs(c_data[0] - 5.0f) < 1e-6f);
    assert(std::abs(c_data[1] - 7.0f) < 1e-6f);
    assert(std::abs(c_data[2] - 9.0f) < 1e-6f);

    std::cout << "  ✓ Basic addition passed" << std::endl;
}

void test_add_operator_2d() {
    std::cout << "Testing AddOp 2D operation..." << std::endl;

    // Create two 2x3 tensors
    TensorShape shape({2, 3});
    Tensor a(shape, DataType::Float32, Device::CPU);
    Tensor b(shape, DataType::Float32, Device::CPU);
    Tensor c(shape, DataType::Float32, Device::CPU);

    float* a_data = static_cast<float*>(a.data());
    float* b_data = static_cast<float*>(b.data());

    // Initialize: A = [[1, 2, 3], [4, 5, 6]]
    for (size_t i = 0; i < 6; ++i) {
        a_data[i] = static_cast<float>(i + 1);
        b_data[i] = static_cast<float>(i + 1) * 2.0f;
    }

    AddOp add_op;
    add_op.execute(Backend::CPU, {&a, &b}, {&c});

    float* c_data = static_cast<float*>(c.data());
    
    // C = [[3, 6, 9], [12, 15, 18]]
    for (size_t i = 0; i < 6; ++i) {
        float expected = static_cast<float>(i + 1) * 3.0f;
        assert(std::abs(c_data[i] - expected) < 1e-6f);
    }

    std::cout << "  ✓ 2D addition passed" << std::endl;
}

void test_add_operator_broadcast_scalar() {
    std::cout << "Testing AddOp broadcasting (scalar)..." << std::endl;

    // [3, 4] + [4] = [3, 4]
    TensorShape shape_a({3, 4});
    TensorShape shape_b({4});
    TensorShape shape_c({3, 4});

    Tensor a(shape_a, DataType::Float32, Device::CPU);
    Tensor b(shape_b, DataType::Float32, Device::CPU);
    Tensor c(shape_c, DataType::Float32, Device::CPU);

    float* a_data = static_cast<float*>(a.data());
    float* b_data = static_cast<float*>(b.data());

    // A = [[1, 2, 3, 4],
    //      [5, 6, 7, 8],
    //      [9, 10, 11, 12]]
    for (size_t i = 0; i < 12; ++i) {
        a_data[i] = static_cast<float>(i + 1);
    }

    // B = [10, 20, 30, 40]
    b_data[0] = 10.0f;
    b_data[1] = 20.0f;
    b_data[2] = 30.0f;
    b_data[3] = 40.0f;

    AddOp add_op;
    add_op.execute(Backend::CPU, {&a, &b}, {&c});

    float* c_data = static_cast<float*>(c.data());

    // Expected: [[11, 22, 33, 44],
    //            [15, 26, 37, 48],
    //            [19, 30, 41, 52]]
    float expected[] = {11, 22, 33, 44, 15, 26, 37, 48, 19, 30, 41, 52};
    for (size_t i = 0; i < 12; ++i) {
        assert(std::abs(c_data[i] - expected[i]) < 1e-6f);
    }

    std::cout << "  ✓ Broadcasting (scalar) passed" << std::endl;
}

void test_add_operator_broadcast_column() {
    std::cout << "Testing AddOp broadcasting (column)..." << std::endl;

    // [3, 4] + [3, 1] = [3, 4]
    TensorShape shape_a({3, 4});
    TensorShape shape_b({3, 1});
    TensorShape shape_c({3, 4});

    Tensor a(shape_a, DataType::Float32, Device::CPU);
    Tensor b(shape_b, DataType::Float32, Device::CPU);
    Tensor c(shape_c, DataType::Float32, Device::CPU);

    float* a_data = static_cast<float*>(a.data());
    float* b_data = static_cast<float*>(b.data());

    // A = [[1, 2, 3, 4],
    //      [5, 6, 7, 8],
    //      [9, 10, 11, 12]]
    for (size_t i = 0; i < 12; ++i) {
        a_data[i] = static_cast<float>(i + 1);
    }

    // B = [[100],
    //      [200],
    //      [300]]
    b_data[0] = 100.0f;
    b_data[1] = 200.0f;
    b_data[2] = 300.0f;

    AddOp add_op;
    add_op.execute(Backend::CPU, {&a, &b}, {&c});

    float* c_data = static_cast<float*>(c.data());

    // Expected: [[101, 102, 103, 104],
    //            [205, 206, 207, 208],
    //            [309, 310, 311, 312]]
    float expected[] = {101, 102, 103, 104, 205, 206, 207, 208, 309, 310, 311, 312};
    for (size_t i = 0; i < 12; ++i) {
        assert(std::abs(c_data[i] - expected[i]) < 1e-6f);
    }

    std::cout << "  ✓ Broadcasting (column) passed" << std::endl;
}

void test_add_operator_shape_inference() {
    std::cout << "Testing AddOp shape inference..." << std::endl;

    AddOp add_op;

    // Case 1: Same shapes
    {
        TensorShape shape_a({3, 4});
        TensorShape shape_b({3, 4});
        auto output_shapes = add_op.inferOutputShapes({shape_a, shape_b});
        assert(output_shapes.size() == 1);
        assert(output_shapes[0].ndim() == 2);
        assert(output_shapes[0].dim(0) == 3);
        assert(output_shapes[0].dim(1) == 4);
    }

    // Case 2: Broadcasting [3, 4] + [4]
    {
        TensorShape shape_a({3, 4});
        TensorShape shape_b({4});
        auto output_shapes = add_op.inferOutputShapes({shape_a, shape_b});
        assert(output_shapes.size() == 1);
        assert(output_shapes[0].ndim() == 2);
        assert(output_shapes[0].dim(0) == 3);
        assert(output_shapes[0].dim(1) == 4);
    }

    // Case 3: Broadcasting [3, 4] + [3, 1]
    {
        TensorShape shape_a({3, 4});
        TensorShape shape_b({3, 1});
        auto output_shapes = add_op.inferOutputShapes({shape_a, shape_b});
        assert(output_shapes.size() == 1);
        assert(output_shapes[0].ndim() == 2);
        assert(output_shapes[0].dim(0) == 3);
        assert(output_shapes[0].dim(1) == 4);
    }

    // Case 4: Incompatible shapes should throw
    {
        TensorShape shape_a({3, 4});
        TensorShape shape_b({2, 4});
        bool threw = false;
        try {
            add_op.inferOutputShapes({shape_a, shape_b});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    std::cout << "  ✓ Shape inference passed" << std::endl;
}

void test_operator_validation() {
    std::cout << "Testing Operator validation..." << std::endl;

    AddOp add_op;
    TensorShape shape({3});
    Tensor a(shape, DataType::Float32, Device::CPU);
    Tensor b(shape, DataType::Float32, Device::CPU);
    Tensor c(shape, DataType::Float32, Device::CPU);

    // Test wrong input count
    {
        bool threw = false;
        try {
            add_op.execute(Backend::CPU, {&a}, {&c});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    // Test wrong output count
    {
        bool threw = false;
        try {
            add_op.execute(Backend::CPU, {&a, &b}, {&c, &c});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    // Test unsupported backend
    {
        bool threw = false;
        try {
            add_op.execute(Backend::CUDA, {&a, &b}, {&c});
        } catch (const std::runtime_error&) {
            threw = true;
        }
        assert(threw);
    }

    std::cout << "  ✓ Validation passed" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "ForgeRT Operator Tests (Phase 1)" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        test_add_operator_basic();
        test_add_operator_2d();
        test_add_operator_broadcast_scalar();
        test_add_operator_broadcast_column();
        test_add_operator_shape_inference();
        test_operator_validation();

        std::cout << "========================================" << std::endl;
        std::cout << "All tests PASSED ✓" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Test FAILED: " << e.what() << std::endl;
        return 1;
    }
}

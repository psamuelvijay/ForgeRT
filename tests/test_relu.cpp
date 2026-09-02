#include "forgert/operator/relu.h"
#include "forgert/tensor/tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace forgert;

void test_relu_positive() {
    std::cout << "Testing ReLU with positive values..." << std::endl;
    
    TensorShape shape({4});
    Tensor input(shape, DataType::Float32, Device::CPU);
    Tensor output(shape, DataType::Float32, Device::CPU);

    float* in_data = static_cast<float*>(input.data());
    in_data[0] = 1.0f;
    in_data[1] = 2.5f;
    in_data[2] = 0.1f;
    in_data[3] = 100.0f;

    ReLUOp op;
    op.execute(Backend::CPU, {&input}, {&output});

    const float* out_data = static_cast<const float*>(output.data());
    assert(std::abs(out_data[0] - 1.0f) < 1e-6f);
    assert(std::abs(out_data[1] - 2.5f) < 1e-6f);
    assert(std::abs(out_data[2] - 0.1f) < 1e-6f);
    assert(std::abs(out_data[3] - 100.0f) < 1e-6f);

    std::cout << "  ✓ Positive values passed" << std::endl;
}

void test_relu_negative() {
    std::cout << "Testing ReLU with negative values..." << std::endl;
    
    TensorShape shape({4});
    Tensor input(shape, DataType::Float32, Device::CPU);
    Tensor output(shape, DataType::Float32, Device::CPU);

    float* in_data = static_cast<float*>(input.data());
    in_data[0] = -1.0f;
    in_data[1] = -2.5f;
    in_data[2] = -0.1f;
    in_data[3] = -100.0f;

    ReLUOp op;
    op.execute(Backend::CPU, {&input}, {&output});

    const float* out_data = static_cast<const float*>(output.data());
    assert(std::abs(out_data[0] - 0.0f) < 1e-6f);
    assert(std::abs(out_data[1] - 0.0f) < 1e-6f);
    assert(std::abs(out_data[2] - 0.0f) < 1e-6f);
    assert(std::abs(out_data[3] - 0.0f) < 1e-6f);

    std::cout << "  ✓ Negative values passed" << std::endl;
}

void test_relu_zero() {
    std::cout << "Testing ReLU with zero..." << std::endl;
    
    TensorShape shape({3});
    Tensor input(shape, DataType::Float32, Device::CPU);
    Tensor output(shape, DataType::Float32, Device::CPU);

    float* in_data = static_cast<float*>(input.data());
    in_data[0] = 0.0f;
    in_data[1] = 0.0f;
    in_data[2] = 0.0f;

    ReLUOp op;
    op.execute(Backend::CPU, {&input}, {&output});

    const float* out_data = static_cast<const float*>(output.data());
    assert(std::abs(out_data[0] - 0.0f) < 1e-6f);
    assert(std::abs(out_data[1] - 0.0f) < 1e-6f);
    assert(std::abs(out_data[2] - 0.0f) < 1e-6f);

    std::cout << "  ✓ Zero values passed" << std::endl;
}

void test_relu_mixed() {
    std::cout << "Testing ReLU with mixed values..." << std::endl;
    
    TensorShape shape({6});
    Tensor input(shape, DataType::Float32, Device::CPU);
    Tensor output(shape, DataType::Float32, Device::CPU);

    float* in_data = static_cast<float*>(input.data());
    in_data[0] = -5.0f;
    in_data[1] = 3.2f;
    in_data[2] = 0.0f;
    in_data[3] = -0.01f;
    in_data[4] = 42.0f;
    in_data[5] = -999.9f;

    ReLUOp op;
    op.execute(Backend::CPU, {&input}, {&output});

    const float* out_data = static_cast<const float*>(output.data());
    assert(std::abs(out_data[0] - 0.0f) < 1e-6f);    // -5.0 -> 0.0
    assert(std::abs(out_data[1] - 3.2f) < 1e-6f);    // 3.2 -> 3.2
    assert(std::abs(out_data[2] - 0.0f) < 1e-6f);    // 0.0 -> 0.0
    assert(std::abs(out_data[3] - 0.0f) < 1e-6f);    // -0.01 -> 0.0
    assert(std::abs(out_data[4] - 42.0f) < 1e-6f);   // 42.0 -> 42.0
    assert(std::abs(out_data[5] - 0.0f) < 1e-6f);    // -999.9 -> 0.0

    std::cout << "  ✓ Mixed values passed" << std::endl;
}

void test_relu_2d_shape() {
    std::cout << "Testing ReLU with 2D shape..." << std::endl;
    
    TensorShape shape({2, 3});
    Tensor input(shape, DataType::Float32, Device::CPU);
    Tensor output(shape, DataType::Float32, Device::CPU);

    float* in_data = static_cast<float*>(input.data());
    in_data[0] = 1.0f;  in_data[1] = -2.0f; in_data[2] = 3.0f;
    in_data[3] = -4.0f; in_data[4] = 5.0f;  in_data[5] = -6.0f;

    ReLUOp op;
    op.execute(Backend::CPU, {&input}, {&output});

    const float* out_data = static_cast<const float*>(output.data());
    assert(std::abs(out_data[0] - 1.0f) < 1e-6f);
    assert(std::abs(out_data[1] - 0.0f) < 1e-6f);
    assert(std::abs(out_data[2] - 3.0f) < 1e-6f);
    assert(std::abs(out_data[3] - 0.0f) < 1e-6f);
    assert(std::abs(out_data[4] - 5.0f) < 1e-6f);
    assert(std::abs(out_data[5] - 0.0f) < 1e-6f);

    std::cout << "  ✓ 2D shape preservation passed" << std::endl;
}

void test_relu_shape_inference() {
    std::cout << "Testing ReLU shape inference..." << std::endl;
    
    ReLUOp op;
    
    // Test various shapes
    TensorShape shape1({5});
    TensorShape shape2({3, 4});
    TensorShape shape3({2, 3, 4});
    
    auto out1 = op.inferOutputShapes({shape1});
    auto out2 = op.inferOutputShapes({shape2});
    auto out3 = op.inferOutputShapes({shape3});
    
    assert(out1.size() == 1);
    assert(out2.size() == 1);
    assert(out3.size() == 1);
    
    assert(out1[0].ndim() == 1 && out1[0].dim(0) == 5);
    assert(out2[0].ndim() == 2 && out2[0].dim(0) == 3 && out2[0].dim(1) == 4);
    assert(out3[0].ndim() == 3 && out3[0].dim(0) == 2 && out3[0].dim(1) == 3 && out3[0].dim(2) == 4);

    std::cout << "  ✓ Shape inference passed" << std::endl;
}

void test_relu_invalid_input_count() {
    std::cout << "Testing ReLU invalid input count..." << std::endl;
    
    ReLUOp op;
    TensorShape shape({3});
    
    bool caught = false;
    try {
        // Try to infer with 0 inputs
        op.inferOutputShapes({});
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    assert(caught);
    
    caught = false;
    try {
        // Try to infer with 2 inputs
        op.inferOutputShapes({shape, shape});
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    assert(caught);

    std::cout << "  ✓ Invalid input count detection passed" << std::endl;
}

void test_relu_backend_support() {
    std::cout << "Testing ReLU backend support..." << std::endl;
    
    ReLUOp op;
    
    // CPU should be supported
    assert(op.supportsBackend(Backend::CPU));
    
    // CUDA not yet implemented in Phase 2
    assert(!op.supportsBackend(Backend::CUDA));

    std::cout << "  ✓ Backend support passed" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "ForgeRT ReLU Operator Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    test_relu_positive();
    test_relu_negative();
    test_relu_zero();
    test_relu_mixed();
    test_relu_2d_shape();
    test_relu_shape_inference();
    test_relu_invalid_input_count();
    test_relu_backend_support();

    std::cout << "========================================" << std::endl;
    std::cout << "All tests PASSED ✓" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}

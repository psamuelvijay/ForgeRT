#include "forgert/operator/layernorm.h"
#include "forgert/tensor/tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace forgert;

const float TOLERANCE = 1e-4f;

void test_layernorm_simple_1d() {
    std::cout << "Testing LayerNorm with simple 1D input..." << std::endl;
    
    // Input: [1, 2, 3, 4, 5]
    // Mean = 3.0
    // Variance = ((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 5
    //          = (4 + 1 + 0 + 1 + 4) / 5 = 2.0
    // Std = sqrt(2.0) ≈ 1.4142
    // Output = [(1-3)/1.4142, (2-3)/1.4142, (3-3)/1.4142, (4-3)/1.4142, (5-3)/1.4142]
    //        ≈ [-1.4142, -0.7071, 0.0, 0.7071, 1.4142]
    
    TensorShape shape({5});
    Tensor input(shape, DataType::Float32, Device::CPU);
    Tensor output(shape, DataType::Float32, Device::CPU);

    float* in_data = static_cast<float*>(input.data());
    in_data[0] = 1.0f;
    in_data[1] = 2.0f;
    in_data[2] = 3.0f;
    in_data[3] = 4.0f;
    in_data[4] = 5.0f;

    LayerNormOp op;
    op.execute(Backend::CPU, {&input}, {&output});

    const float* out_data = static_cast<const float*>(output.data());
    
    // Check approximate values
    assert(std::abs(out_data[0] - (-1.4142f)) < TOLERANCE);
    assert(std::abs(out_data[1] - (-0.7071f)) < TOLERANCE);
    assert(std::abs(out_data[2] - 0.0f) < TOLERANCE);
    assert(std::abs(out_data[3] - 0.7071f) < TOLERANCE);
    assert(std::abs(out_data[4] - 1.4142f) < TOLERANCE);
    
    // Verify mean ≈ 0
    float sum = 0.0f;
    for (int i = 0; i < 5; ++i) {
        sum += out_data[i];
    }
    float mean = sum / 5.0f;
    assert(std::abs(mean) < TOLERANCE);
    
    // Verify variance ≈ 1
    float var_sum = 0.0f;
    for (int i = 0; i < 5; ++i) {
        var_sum += out_data[i] * out_data[i];
    }
    float variance = var_sum / 5.0f;
    assert(std::abs(variance - 1.0f) < TOLERANCE);

    std::cout << "  ✓ Simple 1D passed" << std::endl;
}

void test_layernorm_zero_mean() {
    std::cout << "Testing LayerNorm with zero-mean input..." << std::endl;
    
    // Input: [-2, -1, 0, 1, 2]
    // Mean = 0.0
    // Variance = (4 + 1 + 0 + 1 + 4) / 5 = 2.0
    
    TensorShape shape({5});
    Tensor input(shape, DataType::Float32, Device::CPU);
    Tensor output(shape, DataType::Float32, Device::CPU);

    float* in_data = static_cast<float*>(input.data());
    in_data[0] = -2.0f;
    in_data[1] = -1.0f;
    in_data[2] = 0.0f;
    in_data[3] = 1.0f;
    in_data[4] = 2.0f;

    LayerNormOp op;
    op.execute(Backend::CPU, {&input}, {&output});

    const float* out_data = static_cast<const float*>(output.data());
    
    // Verify mean ≈ 0
    float sum = 0.0f;
    for (int i = 0; i < 5; ++i) {
        sum += out_data[i];
    }
    float mean = sum / 5.0f;
    assert(std::abs(mean) < TOLERANCE);
    
    // Verify variance ≈ 1
    float var_sum = 0.0f;
    for (int i = 0; i < 5; ++i) {
        var_sum += out_data[i] * out_data[i];
    }
    float variance = var_sum / 5.0f;
    assert(std::abs(variance - 1.0f) < TOLERANCE);

    std::cout << "  ✓ Zero-mean input passed" << std::endl;
}

void test_layernorm_constant_input() {
    std::cout << "Testing LayerNorm with constant input..." << std::endl;
    
    // Input: [5, 5, 5, 5]
    // Mean = 5.0
    // Variance = 0.0
    // With epsilon, output should be [0, 0, 0, 0]
    
    TensorShape shape({4});
    Tensor input(shape, DataType::Float32, Device::CPU);
    Tensor output(shape, DataType::Float32, Device::CPU);

    float* in_data = static_cast<float*>(input.data());
    in_data[0] = 5.0f;
    in_data[1] = 5.0f;
    in_data[2] = 5.0f;
    in_data[3] = 5.0f;

    LayerNormOp op(1e-5f);
    op.execute(Backend::CPU, {&input}, {&output});

    const float* out_data = static_cast<const float*>(output.data());
    
    // All outputs should be 0 (constant - mean = 0)
    for (int i = 0; i < 4; ++i) {
        assert(std::abs(out_data[i]) < TOLERANCE);
    }

    std::cout << "  ✓ Constant input passed" << std::endl;
}

void test_layernorm_2d_batch() {
    std::cout << "Testing LayerNorm with 2D batch..." << std::endl;
    
    // Input: [[1, 2, 3, 4],
    //         [5, 6, 7, 8]]
    // Each row should be normalized independently
    
    TensorShape shape({2, 4});
    Tensor input(shape, DataType::Float32, Device::CPU);
    Tensor output(shape, DataType::Float32, Device::CPU);

    float* in_data = static_cast<float*>(input.data());
    // Row 0: [1, 2, 3, 4]
    in_data[0] = 1.0f; in_data[1] = 2.0f; in_data[2] = 3.0f; in_data[3] = 4.0f;
    // Row 1: [5, 6, 7, 8]
    in_data[4] = 5.0f; in_data[5] = 6.0f; in_data[6] = 7.0f; in_data[7] = 8.0f;

    LayerNormOp op;
    op.execute(Backend::CPU, {&input}, {&output});

    const float* out_data = static_cast<const float*>(output.data());
    
    // Check each row independently
    for (int row = 0; row < 2; ++row) {
        float sum = 0.0f;
        for (int i = 0; i < 4; ++i) {
            sum += out_data[row * 4 + i];
        }
        float mean = sum / 4.0f;
        assert(std::abs(mean) < TOLERANCE);
        
        float var_sum = 0.0f;
        for (int i = 0; i < 4; ++i) {
            var_sum += out_data[row * 4 + i] * out_data[row * 4 + i];
        }
        float variance = var_sum / 4.0f;
        assert(std::abs(variance - 1.0f) < TOLERANCE);
    }

    std::cout << "  ✓ 2D batch passed" << std::endl;
}

void test_layernorm_large_values() {
    std::cout << "Testing LayerNorm with large values..." << std::endl;
    
    // Input: [1000, 1001, 1002, 1003]
    // Should produce same pattern as [0, 1, 2, 3]
    
    TensorShape shape({4});
    Tensor input(shape, DataType::Float32, Device::CPU);
    Tensor output(shape, DataType::Float32, Device::CPU);

    float* in_data = static_cast<float*>(input.data());
    in_data[0] = 1000.0f;
    in_data[1] = 1001.0f;
    in_data[2] = 1002.0f;
    in_data[3] = 1003.0f;

    LayerNormOp op;
    op.execute(Backend::CPU, {&input}, {&output});

    const float* out_data = static_cast<const float*>(output.data());
    
    // Verify no NaN or inf
    for (int i = 0; i < 4; ++i) {
        assert(!std::isnan(out_data[i]));
        assert(!std::isinf(out_data[i]));
    }
    
    // Verify mean ≈ 0 and variance ≈ 1
    float sum = 0.0f;
    for (int i = 0; i < 4; ++i) {
        sum += out_data[i];
    }
    float mean = sum / 4.0f;
    assert(std::abs(mean) < TOLERANCE);
    
    float var_sum = 0.0f;
    for (int i = 0; i < 4; ++i) {
        var_sum += out_data[i] * out_data[i];
    }
    float variance = var_sum / 4.0f;
    assert(std::abs(variance - 1.0f) < TOLERANCE);

    std::cout << "  ✓ Large values passed" << std::endl;
}

void test_layernorm_shape_preservation() {
    std::cout << "Testing LayerNorm shape preservation..." << std::endl;
    
    TensorShape shape({10, 512});
    Tensor input(shape, DataType::Float32, Device::CPU);
    Tensor output(shape, DataType::Float32, Device::CPU);

    float* in_data = static_cast<float*>(input.data());
    for (size_t i = 0; i < 10 * 512; ++i) {
        in_data[i] = static_cast<float>(i % 100) - 50.0f;
    }

    LayerNormOp op;
    op.execute(Backend::CPU, {&input}, {&output});

    // Verify output shape matches input
    assert(output.shape().ndim() == 2);
    assert(output.shape().dim(0) == 10);
    assert(output.shape().dim(1) == 512);

    std::cout << "  ✓ Shape preservation passed" << std::endl;
}

void test_layernorm_shape_inference() {
    std::cout << "Testing LayerNorm shape inference..." << std::endl;
    
    LayerNormOp op;
    
    // Test various shapes
    auto out1 = op.inferOutputShapes({TensorShape({128})});
    assert(out1.size() == 1);
    assert(out1[0].ndim() == 1 && out1[0].dim(0) == 128);
    
    auto out2 = op.inferOutputShapes({TensorShape({10, 512})});
    assert(out2.size() == 1);
    assert(out2[0].ndim() == 2 && out2[0].dim(0) == 10 && out2[0].dim(1) == 512);
    
    auto out3 = op.inferOutputShapes({TensorShape({2, 3, 768})});
    assert(out3.size() == 1);
    assert(out3[0].ndim() == 3);

    std::cout << "  ✓ Shape inference passed" << std::endl;
}

void test_layernorm_invalid_input_count() {
    std::cout << "Testing LayerNorm invalid input count..." << std::endl;
    
    LayerNormOp op;
    TensorShape shape({10});
    
    bool caught = false;
    try {
        op.inferOutputShapes({});
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    assert(caught);
    
    caught = false;
    try {
        op.inferOutputShapes({shape, shape});
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    assert(caught);

    std::cout << "  ✓ Invalid input count detection passed" << std::endl;
}

void test_layernorm_backend_support() {
    std::cout << "Testing LayerNorm backend support..." << std::endl;
    
    LayerNormOp op;
    
    // CPU should be supported
    assert(op.supportsBackend(Backend::CPU));
    
    // CUDA not yet implemented in Phase 2
    assert(!op.supportsBackend(Backend::CUDA));

    std::cout << "  ✓ Backend support passed" << std::endl;
}

void test_layernorm_epsilon_validation() {
    std::cout << "Testing LayerNorm epsilon validation..." << std::endl;
    
    // Valid epsilon
    LayerNormOp op1(1e-5f);
    LayerNormOp op2(1e-8f);
    
    // Invalid epsilon (negative)
    bool caught = false;
    try {
        LayerNormOp op3(-1e-5f);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    assert(caught);
    
    // Invalid epsilon (zero)
    caught = false;
    try {
        LayerNormOp op4(0.0f);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    assert(caught);

    std::cout << "  ✓ Epsilon validation passed" << std::endl;
}

void test_layernorm_3d_tensor() {
    std::cout << "Testing LayerNorm with 3D tensor..." << std::endl;
    
    // Shape: [2, 3, 4] - layernorm along last dimension (size 4)
    // Should produce 6 independent normalizations (2*3 = 6 groups)
    
    TensorShape shape({2, 3, 4});
    Tensor input(shape, DataType::Float32, Device::CPU);
    Tensor output(shape, DataType::Float32, Device::CPU);

    float* in_data = static_cast<float*>(input.data());
    for (int i = 0; i < 24; ++i) {
        in_data[i] = static_cast<float>(i);
    }

    LayerNormOp op;
    op.execute(Backend::CPU, {&input}, {&output});

    const float* out_data = static_cast<const float*>(output.data());
    
    // Check that each group of 4 is normalized
    for (int group = 0; group < 6; ++group) {
        float sum = 0.0f;
        for (int i = 0; i < 4; ++i) {
            sum += out_data[group * 4 + i];
        }
        float mean = sum / 4.0f;
        assert(std::abs(mean) < TOLERANCE);
        
        float var_sum = 0.0f;
        for (int i = 0; i < 4; ++i) {
            var_sum += out_data[group * 4 + i] * out_data[group * 4 + i];
        }
        float variance = var_sum / 4.0f;
        assert(std::abs(variance - 1.0f) < TOLERANCE);
    }

    std::cout << "  ✓ 3D tensor passed" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "ForgeRT LayerNorm Operator Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    test_layernorm_simple_1d();
    test_layernorm_zero_mean();
    test_layernorm_constant_input();
    test_layernorm_2d_batch();
    test_layernorm_large_values();
    test_layernorm_shape_preservation();
    test_layernorm_shape_inference();
    test_layernorm_invalid_input_count();
    test_layernorm_backend_support();
    test_layernorm_epsilon_validation();
    test_layernorm_3d_tensor();

    std::cout << "========================================" << std::endl;
    std::cout << "All tests PASSED ✓" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}

#include "forgert/operator/softmax.h"
#include "forgert/tensor/tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace forgert;

const float TOLERANCE = 1e-5f;

void test_softmax_simple_1d() {
    std::cout << "Testing Softmax with simple 1D input..." << std::endl;
    
    // Input: [1, 2, 3]
    // max = 3
    // exp(1-3) = exp(-2) ≈ 0.1353
    // exp(2-3) = exp(-1) ≈ 0.3679
    // exp(3-3) = exp(0) = 1.0
    // sum ≈ 1.5032
    // softmax ≈ [0.0900, 0.2447, 0.6652]
    
    TensorShape shape({3});
    Tensor input(shape, DataType::Float32, Device::CPU);
    Tensor output(shape, DataType::Float32, Device::CPU);

    float* in_data = static_cast<float*>(input.data());
    in_data[0] = 1.0f;
    in_data[1] = 2.0f;
    in_data[2] = 3.0f;

    SoftmaxOp op;
    op.execute(Backend::CPU, {&input}, {&output});

    const float* out_data = static_cast<const float*>(output.data());
    
    // Check approximate values
    assert(std::abs(out_data[0] - 0.09003057f) < TOLERANCE);
    assert(std::abs(out_data[1] - 0.24472847f) < TOLERANCE);
    assert(std::abs(out_data[2] - 0.66524096f) < TOLERANCE);
    
    // Check sum = 1
    float sum = out_data[0] + out_data[1] + out_data[2];
    assert(std::abs(sum - 1.0f) < TOLERANCE);

    std::cout << "  ✓ Simple 1D passed" << std::endl;
}

void test_softmax_uniform() {
    std::cout << "Testing Softmax with uniform input..." << std::endl;
    
    // Input: [5, 5, 5, 5]
    // All equal -> uniform distribution [0.25, 0.25, 0.25, 0.25]
    
    TensorShape shape({4});
    Tensor input(shape, DataType::Float32, Device::CPU);
    Tensor output(shape, DataType::Float32, Device::CPU);

    float* in_data = static_cast<float*>(input.data());
    in_data[0] = 5.0f;
    in_data[1] = 5.0f;
    in_data[2] = 5.0f;
    in_data[3] = 5.0f;

    SoftmaxOp op;
    op.execute(Backend::CPU, {&input}, {&output});

    const float* out_data = static_cast<const float*>(output.data());
    
    for (int i = 0; i < 4; ++i) {
        assert(std::abs(out_data[i] - 0.25f) < TOLERANCE);
    }
    
    // Check sum = 1
    float sum = 0.0f;
    for (int i = 0; i < 4; ++i) {
        sum += out_data[i];
    }
    assert(std::abs(sum - 1.0f) < TOLERANCE);

    std::cout << "  ✓ Uniform input passed" << std::endl;
}

void test_softmax_negative_values() {
    std::cout << "Testing Softmax with negative values..." << std::endl;
    
    // Input: [-1, -2, -3]
    // max = -1
    // exp(-1-(-1)) = exp(0) = 1.0
    // exp(-2-(-1)) = exp(-1) ≈ 0.3679
    // exp(-3-(-1)) = exp(-2) ≈ 0.1353
    // sum ≈ 1.5032
    // softmax ≈ [0.6652, 0.2447, 0.0900]
    
    TensorShape shape({3});
    Tensor input(shape, DataType::Float32, Device::CPU);
    Tensor output(shape, DataType::Float32, Device::CPU);

    float* in_data = static_cast<float*>(input.data());
    in_data[0] = -1.0f;
    in_data[1] = -2.0f;
    in_data[2] = -3.0f;

    SoftmaxOp op;
    op.execute(Backend::CPU, {&input}, {&output});

    const float* out_data = static_cast<const float*>(output.data());
    
    assert(std::abs(out_data[0] - 0.66524096f) < TOLERANCE);
    assert(std::abs(out_data[1] - 0.24472847f) < TOLERANCE);
    assert(std::abs(out_data[2] - 0.09003057f) < TOLERANCE);
    
    // Check sum = 1
    float sum = out_data[0] + out_data[1] + out_data[2];
    assert(std::abs(sum - 1.0f) < TOLERANCE);

    std::cout << "  ✓ Negative values passed" << std::endl;
}

void test_softmax_large_values() {
    std::cout << "Testing Softmax numerical stability with large values..." << std::endl;
    
    // Input: [1000, 1001, 1002]
    // Without max subtraction, exp(1000) would overflow
    // With max subtraction (max = 1002):
    //   exp(1000-1002) = exp(-2) ≈ 0.1353
    //   exp(1001-1002) = exp(-1) ≈ 0.3679
    //   exp(1002-1002) = exp(0) = 1.0
    // Result should be same as [1, 2, 3] case
    
    TensorShape shape({3});
    Tensor input(shape, DataType::Float32, Device::CPU);
    Tensor output(shape, DataType::Float32, Device::CPU);

    float* in_data = static_cast<float*>(input.data());
    in_data[0] = 1000.0f;
    in_data[1] = 1001.0f;
    in_data[2] = 1002.0f;

    SoftmaxOp op;
    op.execute(Backend::CPU, {&input}, {&output});

    const float* out_data = static_cast<const float*>(output.data());
    
    // Should match the [1, 2, 3] case due to shift invariance
    assert(std::abs(out_data[0] - 0.09003057f) < TOLERANCE);
    assert(std::abs(out_data[1] - 0.24472847f) < TOLERANCE);
    assert(std::abs(out_data[2] - 0.66524096f) < TOLERANCE);
    
    // Check sum = 1
    float sum = out_data[0] + out_data[1] + out_data[2];
    assert(std::abs(sum - 1.0f) < TOLERANCE);
    
    // Verify no NaN or inf
    for (int i = 0; i < 3; ++i) {
        assert(!std::isnan(out_data[i]));
        assert(!std::isinf(out_data[i]));
    }

    std::cout << "  ✓ Large values (numerical stability) passed" << std::endl;
}

void test_softmax_2d_batch() {
    std::cout << "Testing Softmax with 2D batch (typical classification case)..." << std::endl;
    
    // Input: [[1, 2, 3],
    //         [4, 5, 6]]
    // Each row should be softmax'd independently
    
    TensorShape shape({2, 3});
    Tensor input(shape, DataType::Float32, Device::CPU);
    Tensor output(shape, DataType::Float32, Device::CPU);

    float* in_data = static_cast<float*>(input.data());
    // Row 0: [1, 2, 3]
    in_data[0] = 1.0f; in_data[1] = 2.0f; in_data[2] = 3.0f;
    // Row 1: [4, 5, 6]
    in_data[3] = 4.0f; in_data[4] = 5.0f; in_data[5] = 6.0f;

    SoftmaxOp op;
    op.execute(Backend::CPU, {&input}, {&output});

    const float* out_data = static_cast<const float*>(output.data());
    
    // Row 0: softmax([1, 2, 3])
    assert(std::abs(out_data[0] - 0.09003057f) < TOLERANCE);
    assert(std::abs(out_data[1] - 0.24472847f) < TOLERANCE);
    assert(std::abs(out_data[2] - 0.66524096f) < TOLERANCE);
    
    // Row 1: softmax([4, 5, 6]) - same pattern due to shift invariance
    assert(std::abs(out_data[3] - 0.09003057f) < TOLERANCE);
    assert(std::abs(out_data[4] - 0.24472847f) < TOLERANCE);
    assert(std::abs(out_data[5] - 0.66524096f) < TOLERANCE);
    
    // Check each row sums to 1
    float sum_row0 = out_data[0] + out_data[1] + out_data[2];
    float sum_row1 = out_data[3] + out_data[4] + out_data[5];
    assert(std::abs(sum_row0 - 1.0f) < TOLERANCE);
    assert(std::abs(sum_row1 - 1.0f) < TOLERANCE);

    std::cout << "  ✓ 2D batch passed" << std::endl;
}

void test_softmax_shape_preservation() {
    std::cout << "Testing Softmax shape preservation..." << std::endl;
    
    TensorShape shape({5});
    Tensor input(shape, DataType::Float32, Device::CPU);
    Tensor output(shape, DataType::Float32, Device::CPU);

    float* in_data = static_cast<float*>(input.data());
    for (int i = 0; i < 5; ++i) {
        in_data[i] = static_cast<float>(i);
    }

    SoftmaxOp op;
    op.execute(Backend::CPU, {&input}, {&output});

    // Verify output shape matches input
    assert(output.shape().ndim() == 1);
    assert(output.shape().dim(0) == 5);
    assert(output.numElements() == 5);

    std::cout << "  ✓ Shape preservation passed" << std::endl;
}

void test_softmax_shape_inference() {
    std::cout << "Testing Softmax shape inference..." << std::endl;
    
    SoftmaxOp op;
    
    // Test various shapes
    auto out1 = op.inferOutputShapes({TensorShape({10})});
    assert(out1.size() == 1);
    assert(out1[0].ndim() == 1 && out1[0].dim(0) == 10);
    
    auto out2 = op.inferOutputShapes({TensorShape({3, 4})});
    assert(out2.size() == 1);
    assert(out2[0].ndim() == 2 && out2[0].dim(0) == 3 && out2[0].dim(1) == 4);
    
    auto out3 = op.inferOutputShapes({TensorShape({2, 3, 4})});
    assert(out3.size() == 1);
    assert(out3[0].ndim() == 3 && out3[0].dim(0) == 2 && out3[0].dim(1) == 3 && out3[0].dim(2) == 4);

    std::cout << "  ✓ Shape inference passed" << std::endl;
}

void test_softmax_invalid_input_count() {
    std::cout << "Testing Softmax invalid input count..." << std::endl;
    
    SoftmaxOp op;
    TensorShape shape({3});
    
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

void test_softmax_backend_support() {
    std::cout << "Testing Softmax backend support..." << std::endl;
    
    SoftmaxOp op;
    
    // CPU should be supported
    assert(op.supportsBackend(Backend::CPU));
    
    // CUDA not yet implemented in Phase 2
    assert(!op.supportsBackend(Backend::CUDA));

    std::cout << "  ✓ Backend support passed" << std::endl;
}

void test_softmax_probability_sum() {
    std::cout << "Testing Softmax probability sum equals 1..." << std::endl;
    
    // Test with various input sizes
    for (size_t size : {2, 5, 10, 100}) {
        TensorShape shape({size});
        Tensor input(shape, DataType::Float32, Device::CPU);
        Tensor output(shape, DataType::Float32, Device::CPU);

        float* in_data = static_cast<float*>(input.data());
        // Fill with arbitrary values
        for (size_t i = 0; i < size; ++i) {
            in_data[i] = static_cast<float>(i) * 0.5f - 10.0f;
        }

        SoftmaxOp op;
        op.execute(Backend::CPU, {&input}, {&output});

        const float* out_data = static_cast<const float*>(output.data());
        
        // Sum all probabilities
        float sum = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            sum += out_data[i];
            // Each probability should be in [0, 1]
            assert(out_data[i] >= 0.0f && out_data[i] <= 1.0f);
        }
        
        assert(std::abs(sum - 1.0f) < TOLERANCE);
    }

    std::cout << "  ✓ Probability sum passed" << std::endl;
}

void test_softmax_3d_tensor() {
    std::cout << "Testing Softmax with 3D tensor..." << std::endl;
    
    // Shape: [2, 2, 3] - softmax along last dimension (size 3)
    // Should produce 4 independent softmax operations (2*2 = 4 groups)
    
    TensorShape shape({2, 2, 3});
    Tensor input(shape, DataType::Float32, Device::CPU);
    Tensor output(shape, DataType::Float32, Device::CPU);

    float* in_data = static_cast<float*>(input.data());
    // Fill with values [0, 1, 2, ..., 11]
    for (int i = 0; i < 12; ++i) {
        in_data[i] = static_cast<float>(i);
    }

    SoftmaxOp op;
    op.execute(Backend::CPU, {&input}, {&output});

    const float* out_data = static_cast<const float*>(output.data());
    
    // Check that each group of 3 sums to 1
    for (int group = 0; group < 4; ++group) {
        float sum = 0.0f;
        for (int i = 0; i < 3; ++i) {
            sum += out_data[group * 3 + i];
        }
        assert(std::abs(sum - 1.0f) < TOLERANCE);
    }

    std::cout << "  ✓ 3D tensor passed" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "ForgeRT Softmax Operator Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    test_softmax_simple_1d();
    test_softmax_uniform();
    test_softmax_negative_values();
    test_softmax_large_values();
    test_softmax_2d_batch();
    test_softmax_shape_preservation();
    test_softmax_shape_inference();
    test_softmax_invalid_input_count();
    test_softmax_backend_support();
    test_softmax_probability_sum();
    test_softmax_3d_tensor();

    std::cout << "========================================" << std::endl;
    std::cout << "All tests PASSED ✓" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}

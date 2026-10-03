#include "forgert/operator/layernorm.h"
#include "forgert/tensor/tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <random>
#include <algorithm>

using namespace forgert;

const float TOLERANCE = 1e-4f;

void test_layernorm_cuda_simple_1d() {
    std::cout << "Testing CUDA LayerNorm with simple 1D input..." << std::endl;
    
    // Input: [1, 2, 3, 4, 5]
    // Expected output should match CPU implementation
    
    TensorShape shape({5});
    Tensor input_cpu(shape, DataType::Float32, Device::CPU);
    Tensor output_cpu(shape, DataType::Float32, Device::CPU);
    Tensor input_cuda(shape, DataType::Float32, Device::CUDA);
    Tensor output_cuda(shape, DataType::Float32, Device::CUDA);

    float* in_cpu = static_cast<float*>(input_cpu.data());
    in_cpu[0] = 1.0f; in_cpu[1] = 2.0f; in_cpu[2] = 3.0f; in_cpu[3] = 4.0f; in_cpu[4] = 5.0f;
    
    // Copy to CUDA
    input_cpu.copyTo(input_cuda);

    LayerNormOp op;
    
    // Execute both CPU and CUDA
    op.execute(Backend::CPU, {&input_cpu}, {&output_cpu});
    op.execute(Backend::CUDA, {&input_cuda}, {&output_cuda});

    // Copy CUDA result back to CPU for comparison
    Tensor output_cuda_host(shape, DataType::Float32, Device::CPU);
    output_cuda.copyTo(output_cuda_host);

    const float* cpu_data = static_cast<const float*>(output_cpu.data());
    const float* cuda_data = static_cast<const float*>(output_cuda_host.data());
    
    // Compare CPU vs CUDA results
    for (int i = 0; i < 5; ++i) {
        assert(std::abs(cpu_data[i] - cuda_data[i]) < TOLERANCE);
    }
    
    // Verify CUDA normalization properties
    float sum = 0.0f;
    for (int i = 0; i < 5; ++i) {
        sum += cuda_data[i];
    }
    float mean = sum / 5.0f;
    assert(std::abs(mean) < TOLERANCE);
    
    float var_sum = 0.0f;
    for (int i = 0; i < 5; ++i) {
        var_sum += cuda_data[i] * cuda_data[i];
    }
    float variance = var_sum / 5.0f;
    assert(std::abs(variance - 1.0f) < TOLERANCE);

    std::cout << "  ✓ CUDA simple 1D passed" << std::endl;
}

void test_layernorm_cuda_2d_batch() {
    std::cout << "Testing CUDA LayerNorm with 2D batch..." << std::endl;
    
    // Input: [[1, 2, 3, 4],
    //         [5, 6, 7, 8]]
    // Each row should be normalized independently
    
    TensorShape shape({2, 4});
    Tensor input_cpu(shape, DataType::Float32, Device::CPU);
    Tensor output_cpu(shape, DataType::Float32, Device::CPU);
    Tensor input_cuda(shape, DataType::Float32, Device::CUDA);
    Tensor output_cuda(shape, DataType::Float32, Device::CUDA);

    float* in_cpu = static_cast<float*>(input_cpu.data());
    // Row 0: [1, 2, 3, 4]
    in_cpu[0] = 1.0f; in_cpu[1] = 2.0f; in_cpu[2] = 3.0f; in_cpu[3] = 4.0f;
    // Row 1: [5, 6, 7, 8]
    in_cpu[4] = 5.0f; in_cpu[5] = 6.0f; in_cpu[6] = 7.0f; in_cpu[7] = 8.0f;
    
    input_cpu.copyTo(input_cuda);

    LayerNormOp op;
    
    op.execute(Backend::CPU, {&input_cpu}, {&output_cpu});
    op.execute(Backend::CUDA, {&input_cuda}, {&output_cuda});

    Tensor output_cuda_host(shape, DataType::Float32, Device::CPU);
    output_cuda.copyTo(output_cuda_host);

    const float* cpu_data = static_cast<const float*>(output_cpu.data());
    const float* cuda_data = static_cast<const float*>(output_cuda_host.data());
    
    // Compare CPU vs CUDA results
    for (int i = 0; i < 8; ++i) {
        assert(std::abs(cpu_data[i] - cuda_data[i]) < TOLERANCE);
    }
    
    // Check each row independently
    for (int row = 0; row < 2; ++row) {
        float sum = 0.0f;
        for (int i = 0; i < 4; ++i) {
            sum += cuda_data[row * 4 + i];
        }
        float mean = sum / 4.0f;
        assert(std::abs(mean) < TOLERANCE);
        
        float var_sum = 0.0f;
        for (int i = 0; i < 4; ++i) {
            var_sum += cuda_data[row * 4 + i] * cuda_data[row * 4 + i];
        }
        float variance = var_sum / 4.0f;
        assert(std::abs(variance - 1.0f) < TOLERANCE);
    }

    std::cout << "  ✓ CUDA 2D batch passed" << std::endl;
}

void test_layernorm_cuda_large_tensor() {
    std::cout << "Testing CUDA LayerNorm with large tensor..." << std::endl;
    
    // Large tensor: [128, 768] (typical transformer hidden size)
    TensorShape shape({128, 768});
    Tensor input_cpu(shape, DataType::Float32, Device::CPU);
    Tensor output_cpu(shape, DataType::Float32, Device::CPU);
    Tensor input_cuda(shape, DataType::Float32, Device::CUDA);
    Tensor output_cuda(shape, DataType::Float32, Device::CUDA);

    // Fill with random data
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    float* in_cpu = static_cast<float*>(input_cpu.data());
    for (size_t i = 0; i < 128 * 768; ++i) {
        in_cpu[i] = dist(gen);
    }
    
    input_cpu.copyTo(input_cuda);

    LayerNormOp op(1e-5f);
    
    op.execute(Backend::CPU, {&input_cpu}, {&output_cpu});
    op.execute(Backend::CUDA, {&input_cuda}, {&output_cuda});

    Tensor output_cuda_host(shape, DataType::Float32, Device::CPU);
    output_cuda.copyTo(output_cuda_host);

    const float* cpu_data = static_cast<const float*>(output_cpu.data());
    const float* cuda_data = static_cast<const float*>(output_cuda_host.data());
    
    // Compare CPU vs CUDA results with relaxed tolerance for large computation
    float max_diff = 0.0f;
    for (size_t i = 0; i < 128 * 768; ++i) {
        float diff = std::abs(cpu_data[i] - cuda_data[i]);
        max_diff = std::max(max_diff, diff);
        assert(diff < TOLERANCE * 2.0f);  // Relaxed tolerance for large tensors
    }
    
    std::cout << "    Max difference: " << max_diff << std::endl;

    std::cout << "  ✓ CUDA large tensor passed" << std::endl;
}

void test_layernorm_cuda_constant_input() {
    std::cout << "Testing CUDA LayerNorm with constant input..." << std::endl;
    
    // Input: [7, 7, 7, 7, 7]
    // All outputs should be 0 due to zero variance
    
    TensorShape shape({5});
    Tensor input_cpu(shape, DataType::Float32, Device::CPU);
    Tensor output_cpu(shape, DataType::Float32, Device::CPU);
    Tensor input_cuda(shape, DataType::Float32, Device::CUDA);
    Tensor output_cuda(shape, DataType::Float32, Device::CUDA);

    float* in_cpu = static_cast<float*>(input_cpu.data());
    for (int i = 0; i < 5; ++i) {
        in_cpu[i] = 7.0f;
    }
    
    input_cpu.copyTo(input_cuda);

    LayerNormOp op(1e-5f);
    
    op.execute(Backend::CPU, {&input_cpu}, {&output_cpu});
    op.execute(Backend::CUDA, {&input_cuda}, {&output_cuda});

    Tensor output_cuda_host(shape, DataType::Float32, Device::CPU);
    output_cuda.copyTo(output_cuda_host);

    const float* cpu_data = static_cast<const float*>(output_cpu.data());
    const float* cuda_data = static_cast<const float*>(output_cuda_host.data());
    
    // Both should produce zeros
    for (int i = 0; i < 5; ++i) {
        assert(std::abs(cpu_data[i]) < TOLERANCE);
        assert(std::abs(cuda_data[i]) < TOLERANCE);
        assert(std::abs(cpu_data[i] - cuda_data[i]) < TOLERANCE);
    }

    std::cout << "  ✓ CUDA constant input passed" << std::endl;
}

void test_layernorm_cuda_3d_tensor() {
    std::cout << "Testing CUDA LayerNorm with 3D tensor..." << std::endl;
    
    // Shape: [4, 6, 8] - layernorm along last dimension (size 8)
    TensorShape shape({4, 6, 8});
    Tensor input_cpu(shape, DataType::Float32, Device::CPU);
    Tensor output_cpu(shape, DataType::Float32, Device::CPU);
    Tensor input_cuda(shape, DataType::Float32, Device::CUDA);
    Tensor output_cuda(shape, DataType::Float32, Device::CUDA);

    // Fill with sequential values
    float* in_cpu = static_cast<float*>(input_cpu.data());
    for (int i = 0; i < 4 * 6 * 8; ++i) {
        in_cpu[i] = static_cast<float>(i % 100) - 50.0f;  // Values from -50 to 49
    }
    
    input_cpu.copyTo(input_cuda);

    LayerNormOp op;
    
    op.execute(Backend::CPU, {&input_cpu}, {&output_cpu});
    op.execute(Backend::CUDA, {&input_cuda}, {&output_cuda});

    Tensor output_cuda_host(shape, DataType::Float32, Device::CPU);
    output_cuda.copyTo(output_cuda_host);

    const float* cpu_data = static_cast<const float*>(output_cpu.data());
    const float* cuda_data = static_cast<const float*>(output_cuda_host.data());
    
    // Compare CPU vs CUDA results
    for (size_t i = 0; i < 4 * 6 * 8; ++i) {
        assert(std::abs(cpu_data[i] - cuda_data[i]) < TOLERANCE);
    }
    
    // Check that each group of 8 is normalized
    for (int group = 0; group < 24; ++group) {  // 4*6 = 24 groups
        float sum = 0.0f;
        for (int i = 0; i < 8; ++i) {
            sum += cuda_data[group * 8 + i];
        }
        float mean = sum / 8.0f;
        assert(std::abs(mean) < TOLERANCE);
        
        float var_sum = 0.0f;
        for (int i = 0; i < 8; ++i) {
            var_sum += cuda_data[group * 8 + i] * cuda_data[group * 8 + i];
        }
        float variance = var_sum / 8.0f;
        assert(std::abs(variance - 1.0f) < TOLERANCE);
    }

    std::cout << "  ✓ CUDA 3D tensor passed" << std::endl;
}

void test_layernorm_cuda_numerical_stability() {
    std::cout << "Testing CUDA LayerNorm numerical stability..." << std::endl;
    
    // Test with very large and very small values
    TensorShape shape({4});
    Tensor input_cuda(shape, DataType::Float32, Device::CUDA);
    Tensor output_cuda(shape, DataType::Float32, Device::CUDA);
    
    Tensor input_host(shape, DataType::Float32, Device::CPU);
    float* in_data = static_cast<float*>(input_host.data());
    
    // Large values that could cause overflow in naive implementation
    in_data[0] = 1e6f;
    in_data[1] = 1e6f + 1.0f;
    in_data[2] = 1e6f + 2.0f;
    in_data[3] = 1e6f + 3.0f;
    
    input_host.copyTo(input_cuda);

    LayerNormOp op(1e-5f);
    op.execute(Backend::CUDA, {&input_cuda}, {&output_cuda});

    Tensor output_host(shape, DataType::Float32, Device::CPU);
    output_cuda.copyTo(output_host);
    
    const float* out_data = static_cast<const float*>(output_host.data());
    
    // Verify no NaN or inf
    for (int i = 0; i < 4; ++i) {
        assert(!std::isnan(out_data[i]));
        assert(!std::isinf(out_data[i]));
    }
    
    // Verify normalization properties
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

    std::cout << "  ✓ CUDA numerical stability passed" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "ForgeRT CUDA LayerNorm Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    test_layernorm_cuda_simple_1d();
    test_layernorm_cuda_2d_batch();
    test_layernorm_cuda_large_tensor();
    test_layernorm_cuda_constant_input();
    test_layernorm_cuda_3d_tensor();
    test_layernorm_cuda_numerical_stability();

    std::cout << "========================================" << std::endl;
    std::cout << "All CUDA LayerNorm tests PASSED ✓" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
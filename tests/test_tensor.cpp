#include "forgert/tensor/tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace forgert;

void test_shape_creation() {
    std::cout << "Testing TensorShape creation..." << std::endl;
    
    // Test 2D shape
    TensorShape shape2d({2, 3});
    assert(shape2d.ndim() == 2);
    assert(shape2d.dim(0) == 2);
    assert(shape2d.dim(1) == 3);
    assert(shape2d.numElements() == 6);
    
    // Row-major strides for 2x3: [3, 1]
    assert(shape2d.stride(0) == 3);
    assert(shape2d.stride(1) == 1);
    
    std::cout << "  " << shape2d.toString() << std::endl;
    std::cout << "  ✓ 2D shape creation passed" << std::endl;
}

void test_shape_flat_indexing() {
    std::cout << "Testing TensorShape flat indexing..." << std::endl;
    
    TensorShape shape({2, 3});
    
    // For row-major 2x3:
    // [0,0] -> 0, [0,1] -> 1, [0,2] -> 2
    // [1,0] -> 3, [1,1] -> 4, [1,2] -> 5
    assert(shape.flatIndex({0, 0}) == 0);
    assert(shape.flatIndex({0, 1}) == 1);
    assert(shape.flatIndex({0, 2}) == 2);
    assert(shape.flatIndex({1, 0}) == 3);
    assert(shape.flatIndex({1, 1}) == 4);
    assert(shape.flatIndex({1, 2}) == 5);
    
    std::cout << "  ✓ Flat indexing passed" << std::endl;
}

void test_tensor_allocation() {
    std::cout << "Testing Tensor CPU allocation..." << std::endl;
    
    TensorShape shape({10, 20});
    Tensor tensor(shape, DataType::Float32, Device::CPU);
    
    assert(tensor.shape().numElements() == 200);
    assert(tensor.dtype() == DataType::Float32);
    assert(tensor.device() == Device::CPU);
    assert(tensor.numBytes() == 200 * 4); // 200 float32 = 800 bytes
    assert(tensor.data() != nullptr);
    
    std::cout << "  Allocated " << tensor.numBytes() << " bytes for " 
              << tensor.numElements() << " float32 elements" << std::endl;
    std::cout << "  ✓ Tensor allocation passed" << std::endl;
}

void test_tensor_zero() {
    std::cout << "Testing Tensor::zero()..." << std::endl;
    
    TensorShape shape({5, 4});
    Tensor tensor(shape, DataType::Float32, Device::CPU);
    tensor.zero();
    
    float* data = static_cast<float*>(tensor.data());
    for (size_t i = 0; i < tensor.numElements(); ++i) {
        assert(data[i] == 0.0f);
    }
    
    std::cout << "  ✓ Zero initialization passed" << std::endl;
}

void test_tensor_fill() {
    std::cout << "Testing Tensor::fill()..." << std::endl;
    
    TensorShape shape({3, 3});
    Tensor tensor(shape, DataType::Float32, Device::CPU);
    tensor.fill(3.14f);
    
    float* data = static_cast<float*>(tensor.data());
    for (size_t i = 0; i < tensor.numElements(); ++i) {
        assert(std::abs(data[i] - 3.14f) < 1e-6f);
    }
    
    std::cout << "  ✓ Fill operation passed" << std::endl;
}

void test_tensor_move() {
    std::cout << "Testing Tensor move semantics..." << std::endl;
    
    TensorShape shape({5, 5});
    Tensor t1(shape, DataType::Float32, Device::CPU);
    t1.fill(42.0f);
    
    void* original_ptr = t1.data();
    
    // Move construct
    Tensor t2(std::move(t1));
    assert(t2.data() == original_ptr);
    assert(t2.numElements() == 25);
    assert(t1.data() == nullptr); // Moved from
    
    float* data = static_cast<float*>(t2.data());
    assert(std::abs(data[0] - 42.0f) < 1e-6f);
    
    std::cout << "  ✓ Move semantics passed" << std::endl;
}

void test_dtype_functions() {
    std::cout << "Testing DataType functions..." << std::endl;
    
    assert(sizeOf(DataType::Float32) == 4);
    assert(sizeOf(DataType::Int32) == 4);
    assert(toString(DataType::Float32) == "float32");
    assert(toString(DataType::Int32) == "int32");
    
    std::cout << "  ✓ DataType functions passed" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "ForgeRT Tensor Tests (Phase 1)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        test_dtype_functions();
        test_shape_creation();
        test_shape_flat_indexing();
        test_tensor_allocation();
        test_tensor_zero();
        test_tensor_fill();
        test_tensor_move();
        
        std::cout << "========================================" << std::endl;
        std::cout << "All tests PASSED ✓" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Test FAILED: " << e.what() << std::endl;
        return 1;
    }
}

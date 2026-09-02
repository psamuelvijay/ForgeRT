#include "forgert/operator/matmul.h"
#include "forgert/tensor/tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace forgert;

void test_matmul_2x3_3x2() {
    std::cout << "Testing MatMul [2,3] @ [3,2]..." << std::endl;
    
    // A = [[1, 2, 3],
    //      [4, 5, 6]]
    // B = [[7,  8],
    //      [9, 10],
    //      [11, 12]]
    // C = A @ B = [[58,  64],
    //              [139, 154]]
    
    TensorShape shape_a({2, 3});
    TensorShape shape_b({3, 2});
    TensorShape shape_c({2, 2});
    
    Tensor a(shape_a, DataType::Float32, Device::CPU);
    Tensor b(shape_b, DataType::Float32, Device::CPU);
    Tensor c(shape_c, DataType::Float32, Device::CPU);

    float* a_data = static_cast<float*>(a.data());
    a_data[0] = 1.0f; a_data[1] = 2.0f; a_data[2] = 3.0f;
    a_data[3] = 4.0f; a_data[4] = 5.0f; a_data[5] = 6.0f;

    float* b_data = static_cast<float*>(b.data());
    b_data[0] = 7.0f;  b_data[1] = 8.0f;
    b_data[2] = 9.0f;  b_data[3] = 10.0f;
    b_data[4] = 11.0f; b_data[5] = 12.0f;

    MatMulOp op;
    op.execute(Backend::CPU, {&a, &b}, {&c});

    const float* c_data = static_cast<const float*>(c.data());
    
    // C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    assert(std::abs(c_data[0] - 58.0f) < 1e-5f);
    // C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    assert(std::abs(c_data[1] - 64.0f) < 1e-5f);
    // C[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    assert(std::abs(c_data[2] - 139.0f) < 1e-5f);
    // C[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    assert(std::abs(c_data[3] - 154.0f) < 1e-5f);

    std::cout << "  ✓ [2,3] @ [3,2] passed" << std::endl;
}

void test_matmul_1x5_5x1() {
    std::cout << "Testing MatMul [1,5] @ [5,1] (dot product)..." << std::endl;
    
    // A = [[1, 2, 3, 4, 5]]
    // B = [[1],
    //      [2],
    //      [3],
    //      [4],
    //      [5]]
    // C = A @ B = [[55]]  (1*1 + 2*2 + 3*3 + 4*4 + 5*5 = 1 + 4 + 9 + 16 + 25 = 55)
    
    TensorShape shape_a({1, 5});
    TensorShape shape_b({5, 1});
    TensorShape shape_c({1, 1});
    
    Tensor a(shape_a, DataType::Float32, Device::CPU);
    Tensor b(shape_b, DataType::Float32, Device::CPU);
    Tensor c(shape_c, DataType::Float32, Device::CPU);

    float* a_data = static_cast<float*>(a.data());
    for (int i = 0; i < 5; ++i) {
        a_data[i] = static_cast<float>(i + 1);
    }

    float* b_data = static_cast<float*>(b.data());
    for (int i = 0; i < 5; ++i) {
        b_data[i] = static_cast<float>(i + 1);
    }

    MatMulOp op;
    op.execute(Backend::CPU, {&a, &b}, {&c});

    const float* c_data = static_cast<const float*>(c.data());
    assert(std::abs(c_data[0] - 55.0f) < 1e-5f);

    std::cout << "  ✓ [1,5] @ [5,1] passed" << std::endl;
}

void test_matmul_3x1_1x4() {
    std::cout << "Testing MatMul [3,1] @ [1,4] (outer product)..." << std::endl;
    
    // A = [[2],
    //      [3],
    //      [4]]
    // B = [[1, 2, 3, 4]]
    // C = A @ B = [[2,  4,  6,  8],
    //              [3,  6,  9, 12],
    //              [4,  8, 12, 16]]
    
    TensorShape shape_a({3, 1});
    TensorShape shape_b({1, 4});
    TensorShape shape_c({3, 4});
    
    Tensor a(shape_a, DataType::Float32, Device::CPU);
    Tensor b(shape_b, DataType::Float32, Device::CPU);
    Tensor c(shape_c, DataType::Float32, Device::CPU);

    float* a_data = static_cast<float*>(a.data());
    a_data[0] = 2.0f;
    a_data[1] = 3.0f;
    a_data[2] = 4.0f;

    float* b_data = static_cast<float*>(b.data());
    b_data[0] = 1.0f; b_data[1] = 2.0f; b_data[2] = 3.0f; b_data[3] = 4.0f;

    MatMulOp op;
    op.execute(Backend::CPU, {&a, &b}, {&c});

    const float* c_data = static_cast<const float*>(c.data());
    
    // Row 0: [2*1, 2*2, 2*3, 2*4]
    assert(std::abs(c_data[0] - 2.0f) < 1e-5f);
    assert(std::abs(c_data[1] - 4.0f) < 1e-5f);
    assert(std::abs(c_data[2] - 6.0f) < 1e-5f);
    assert(std::abs(c_data[3] - 8.0f) < 1e-5f);
    
    // Row 1: [3*1, 3*2, 3*3, 3*4]
    assert(std::abs(c_data[4] - 3.0f) < 1e-5f);
    assert(std::abs(c_data[5] - 6.0f) < 1e-5f);
    assert(std::abs(c_data[6] - 9.0f) < 1e-5f);
    assert(std::abs(c_data[7] - 12.0f) < 1e-5f);
    
    // Row 2: [4*1, 4*2, 4*3, 4*4]
    assert(std::abs(c_data[8] - 4.0f) < 1e-5f);
    assert(std::abs(c_data[9] - 8.0f) < 1e-5f);
    assert(std::abs(c_data[10] - 12.0f) < 1e-5f);
    assert(std::abs(c_data[11] - 16.0f) < 1e-5f);

    std::cout << "  ✓ [3,1] @ [1,4] passed" << std::endl;
}

void test_matmul_identity() {
    std::cout << "Testing MatMul with identity matrix..." << std::endl;
    
    // A = [[1, 2],
    //      [3, 4]]
    // I = [[1, 0],
    //      [0, 1]]
    // C = A @ I = A
    
    TensorShape shape({2, 2});
    
    Tensor a(shape, DataType::Float32, Device::CPU);
    Tensor identity(shape, DataType::Float32, Device::CPU);
    Tensor c(shape, DataType::Float32, Device::CPU);

    float* a_data = static_cast<float*>(a.data());
    a_data[0] = 1.0f; a_data[1] = 2.0f;
    a_data[2] = 3.0f; a_data[3] = 4.0f;

    float* i_data = static_cast<float*>(identity.data());
    i_data[0] = 1.0f; i_data[1] = 0.0f;
    i_data[2] = 0.0f; i_data[3] = 1.0f;

    MatMulOp op;
    op.execute(Backend::CPU, {&a, &identity}, {&c});

    const float* c_data = static_cast<const float*>(c.data());
    assert(std::abs(c_data[0] - 1.0f) < 1e-5f);
    assert(std::abs(c_data[1] - 2.0f) < 1e-5f);
    assert(std::abs(c_data[2] - 3.0f) < 1e-5f);
    assert(std::abs(c_data[3] - 4.0f) < 1e-5f);

    std::cout << "  ✓ Identity matrix passed" << std::endl;
}

void test_matmul_shape_inference() {
    std::cout << "Testing MatMul shape inference..." << std::endl;
    
    MatMulOp op;
    
    // Valid cases
    auto out1 = op.inferOutputShapes({TensorShape({2, 3}), TensorShape({3, 4})});
    assert(out1.size() == 1);
    assert(out1[0].ndim() == 2 && out1[0].dim(0) == 2 && out1[0].dim(1) == 4);
    
    auto out2 = op.inferOutputShapes({TensorShape({5, 7}), TensorShape({7, 3})});
    assert(out2.size() == 1);
    assert(out2[0].ndim() == 2 && out2[0].dim(0) == 5 && out2[0].dim(1) == 3);
    
    auto out3 = op.inferOutputShapes({TensorShape({1, 10}), TensorShape({10, 1})});
    assert(out3.size() == 1);
    assert(out3[0].ndim() == 2 && out3[0].dim(0) == 1 && out3[0].dim(1) == 1);

    std::cout << "  ✓ Shape inference passed" << std::endl;
}

void test_matmul_invalid_shapes() {
    std::cout << "Testing MatMul invalid shape detection..." << std::endl;
    
    MatMulOp op;
    
    // Mismatched inner dimensions: [2,3] @ [4,5]
    bool caught = false;
    try {
        op.inferOutputShapes({TensorShape({2, 3}), TensorShape({4, 5})});
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    assert(caught);
    
    // Non-2D tensor: [3] @ [3,4]
    caught = false;
    try {
        op.inferOutputShapes({TensorShape({3}), TensorShape({3, 4})});
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    assert(caught);
    
    // Non-2D tensor: [2,3] @ [3]
    caught = false;
    try {
        op.inferOutputShapes({TensorShape({2, 3}), TensorShape({3})});
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    assert(caught);

    std::cout << "  ✓ Invalid shape detection passed" << std::endl;
}

void test_matmul_backend_support() {
    std::cout << "Testing MatMul backend support..." << std::endl;
    
    MatMulOp op;
    
    // CPU should be supported
    assert(op.supportsBackend(Backend::CPU));
    
    // CUDA not yet implemented in Phase 2
    assert(!op.supportsBackend(Backend::CUDA));

    std::cout << "  ✓ Backend support passed" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "ForgeRT MatMul Operator Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    test_matmul_2x3_3x2();
    test_matmul_1x5_5x1();
    test_matmul_3x1_1x4();
    test_matmul_identity();
    test_matmul_shape_inference();
    test_matmul_invalid_shapes();
    test_matmul_backend_support();

    std::cout << "========================================" << std::endl;
    std::cout << "All tests PASSED ✓" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}

#include "forgert/graph/graph.h"
#include "forgert/operator/matmul.h"
#include "forgert/operator/relu.h"
#include "forgert/operator/add.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace forgert;

void test_graph_matmul_simple() {
    std::cout << "Testing Graph with MatMul..." << std::endl;
    
    // Graph: C = A @ B
    // A: [2,3], B: [3,2], C: [2,2]
    
    Graph graph;
    auto input_a = graph.addInput(TensorShape({2, 3}), DataType::Float32, "A");
    auto input_b = graph.addInput(TensorShape({3, 2}), DataType::Float32, "B");
    
    auto outputs = graph.addNode(
        std::make_unique<MatMulOp>(),
        {input_a, input_b},
        "MatMul");
    
    graph.markOutput(outputs[0]);
    graph.validate();
    
    // Create input tensors
    Tensor a(TensorShape({2, 3}), DataType::Float32, Device::CPU);
    Tensor b(TensorShape({3, 2}), DataType::Float32, Device::CPU);
    
    float* a_data = static_cast<float*>(a.data());
    a_data[0] = 1.0f; a_data[1] = 2.0f; a_data[2] = 3.0f;
    a_data[3] = 4.0f; a_data[4] = 5.0f; a_data[5] = 6.0f;
    
    float* b_data = static_cast<float*>(b.data());
    b_data[0] = 7.0f;  b_data[1] = 8.0f;
    b_data[2] = 9.0f;  b_data[3] = 10.0f;
    b_data[4] = 11.0f; b_data[5] = 12.0f;
    
    // Execute
    auto results = graph.execute({&a, &b}, Backend::CPU);
    
    // Validate
    assert(results.size() == 1);
    const float* c_data = static_cast<const float*>(results[0]->data());
    
    assert(std::abs(c_data[0] - 58.0f) < 1e-5f);
    assert(std::abs(c_data[1] - 64.0f) < 1e-5f);
    assert(std::abs(c_data[2] - 139.0f) < 1e-5f);
    assert(std::abs(c_data[3] - 154.0f) < 1e-5f);
    
    std::cout << "  ✓ Graph MatMul passed" << std::endl;
}

void test_graph_matmul_relu() {
    std::cout << "Testing Graph with MatMul + ReLU..." << std::endl;
    
    // Graph: D = ReLU(A @ B)
    
    Graph graph;
    auto input_a = graph.addInput(TensorShape({2, 2}), DataType::Float32, "A");
    auto input_b = graph.addInput(TensorShape({2, 2}), DataType::Float32, "B");
    
    auto matmul_out = graph.addNode(
        std::make_unique<MatMulOp>(),
        {input_a, input_b},
        "MatMul");
    
    auto relu_out = graph.addNode(
        std::make_unique<ReLUOp>(),
        {matmul_out[0]},
        "ReLU");
    
    graph.markOutput(relu_out[0]);
    graph.validate();
    
    // Create input tensors
    Tensor a(TensorShape({2, 2}), DataType::Float32, Device::CPU);
    Tensor b(TensorShape({2, 2}), DataType::Float32, Device::CPU);
    
    float* a_data = static_cast<float*>(a.data());
    a_data[0] = 1.0f;  a_data[1] = -2.0f;
    a_data[2] = 3.0f;  a_data[3] = 4.0f;
    
    float* b_data = static_cast<float*>(b.data());
    b_data[0] = -1.0f; b_data[1] = 2.0f;
    b_data[2] = 3.0f;  b_data[3] = -4.0f;
    
    // Execute
    // MatMul result:
    // C[0,0] = 1*(-1) + (-2)*3 = -1 - 6 = -7
    // C[0,1] = 1*2 + (-2)*(-4) = 2 + 8 = 10
    // C[1,0] = 3*(-1) + 4*3 = -3 + 12 = 9
    // C[1,1] = 3*2 + 4*(-4) = 6 - 16 = -10
    // After ReLU: [0, 10, 9, 0]
    
    auto results = graph.execute({&a, &b}, Backend::CPU);
    
    // Validate
    assert(results.size() == 1);
    const float* d_data = static_cast<const float*>(results[0]->data());
    
    assert(std::abs(d_data[0] - 0.0f) < 1e-5f);   // max(0, -7) = 0
    assert(std::abs(d_data[1] - 10.0f) < 1e-5f);  // max(0, 10) = 10
    assert(std::abs(d_data[2] - 9.0f) < 1e-5f);   // max(0, 9) = 9
    assert(std::abs(d_data[3] - 0.0f) < 1e-5f);   // max(0, -10) = 0
    
    std::cout << "  ✓ Graph MatMul + ReLU passed" << std::endl;
}

void test_graph_matmul_chain() {
    std::cout << "Testing Graph with chained MatMul..." << std::endl;
    
    // Graph: D = (A @ B) @ C
    // A: [2,3], B: [3,2], C: [2,1]
    // D: [2,1]
    
    Graph graph;
    auto input_a = graph.addInput(TensorShape({2, 3}), DataType::Float32, "A");
    auto input_b = graph.addInput(TensorShape({3, 2}), DataType::Float32, "B");
    auto input_c = graph.addInput(TensorShape({2, 1}), DataType::Float32, "C");
    
    auto matmul1_out = graph.addNode(
        std::make_unique<MatMulOp>(),
        {input_a, input_b},
        "MatMul1");
    
    auto matmul2_out = graph.addNode(
        std::make_unique<MatMulOp>(),
        {matmul1_out[0], input_c},
        "MatMul2");
    
    graph.markOutput(matmul2_out[0]);
    graph.validate();
    
    // Create input tensors
    Tensor a(TensorShape({2, 3}), DataType::Float32, Device::CPU);
    Tensor b(TensorShape({3, 2}), DataType::Float32, Device::CPU);
    Tensor c(TensorShape({2, 1}), DataType::Float32, Device::CPU);
    
    // Set simple values
    float* a_data = static_cast<float*>(a.data());
    for (int i = 0; i < 6; ++i) a_data[i] = 1.0f;
    
    float* b_data = static_cast<float*>(b.data());
    for (int i = 0; i < 6; ++i) b_data[i] = 1.0f;
    
    float* c_data = static_cast<float*>(c.data());
    c_data[0] = 2.0f;
    c_data[1] = 3.0f;
    
    // Execute
    // A @ B: [2,3] @ [3,2] = [[3, 3], [3, 3]]
    // (A @ B) @ C: [[3, 3], [3, 3]] @ [[2], [3]] = [[15], [15]]
    
    auto results = graph.execute({&a, &b, &c}, Backend::CPU);
    
    // Validate
    assert(results.size() == 1);
    const float* d_data = static_cast<const float*>(results[0]->data());
    
    assert(std::abs(d_data[0] - 15.0f) < 1e-5f);
    assert(std::abs(d_data[1] - 15.0f) < 1e-5f);
    
    std::cout << "  ✓ Graph chained MatMul passed" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "ForgeRT Graph Integration Tests (MatMul)" << std::endl;
    std::cout << "========================================" << std::endl;

    test_graph_matmul_simple();
    test_graph_matmul_relu();
    test_graph_matmul_chain();

    std::cout << "========================================" << std::endl;
    std::cout << "All tests PASSED ✓" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}

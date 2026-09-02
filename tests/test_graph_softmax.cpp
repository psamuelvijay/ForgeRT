#include "forgert/graph/graph.h"
#include "forgert/operator/matmul.h"
#include "forgert/operator/add.h"
#include "forgert/operator/relu.h"
#include "forgert/operator/softmax.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace forgert;

const float TOLERANCE = 1e-5f;

void test_graph_softmax_simple() {
    std::cout << "Testing Graph with Softmax..." << std::endl;
    
    // Graph: Output = Softmax(Input)
    
    Graph graph;
    auto input = graph.addInput(TensorShape({3}), DataType::Float32, "Input");
    
    auto output = graph.addNode(
        std::make_unique<SoftmaxOp>(),
        {input},
        "Softmax");
    
    graph.markOutput(output[0]);
    graph.validate();
    
    // Create input tensor: [1, 2, 3]
    Tensor input_tensor(TensorShape({3}), DataType::Float32, Device::CPU);
    float* in_data = static_cast<float*>(input_tensor.data());
    in_data[0] = 1.0f;
    in_data[1] = 2.0f;
    in_data[2] = 3.0f;
    
    // Execute
    auto results = graph.execute({&input_tensor}, Backend::CPU);
    
    // Validate
    assert(results.size() == 1);
    const float* out_data = static_cast<const float*>(results[0]->data());
    
    assert(std::abs(out_data[0] - 0.09003057f) < TOLERANCE);
    assert(std::abs(out_data[1] - 0.24472847f) < TOLERANCE);
    assert(std::abs(out_data[2] - 0.66524096f) < TOLERANCE);
    
    float sum = out_data[0] + out_data[1] + out_data[2];
    assert(std::abs(sum - 1.0f) < TOLERANCE);
    
    std::cout << "  ✓ Graph Softmax passed" << std::endl;
}

void test_graph_classification_pipeline() {
    std::cout << "Testing Graph with classification pipeline (MatMul -> Add -> ReLU -> Softmax)..." << std::endl;
    
    // Simulates a simple neural network layer:
    // Input: [1, 3] (1 sample, 3 features)
    // Weights: [3, 4] (3 features -> 4 classes)
    // Bias: [4]
    // 
    // Pipeline:
    //   logits = input @ weights      # [1, 3] @ [3, 4] -> [1, 4]
    //   logits = logits + bias        # [1, 4] + [4] -> [1, 4]
    //   logits = ReLU(logits)         # [1, 4] -> [1, 4]
    //   probs = Softmax(logits)       # [1, 4] -> [1, 4]
    
    Graph graph;
    auto input = graph.addInput(TensorShape({1, 3}), DataType::Float32, "Input");
    auto weights = graph.addInput(TensorShape({3, 4}), DataType::Float32, "Weights");
    auto bias = graph.addInput(TensorShape({4}), DataType::Float32, "Bias");
    
    // MatMul: input @ weights
    auto matmul_out = graph.addNode(
        std::make_unique<MatMulOp>(),
        {input, weights},
        "MatMul");
    
    // Add: matmul_out + bias
    auto add_out = graph.addNode(
        std::make_unique<AddOp>(),
        {matmul_out[0], bias},
        "Add");
    
    // ReLU: ReLU(add_out)
    auto relu_out = graph.addNode(
        std::make_unique<ReLUOp>(),
        {add_out[0]},
        "ReLU");
    
    // Softmax: Softmax(relu_out)
    auto softmax_out = graph.addNode(
        std::make_unique<SoftmaxOp>(),
        {relu_out[0]},
        "Softmax");
    
    graph.markOutput(softmax_out[0]);
    graph.validate();
    
    // Create input tensors
    Tensor input_tensor(TensorShape({1, 3}), DataType::Float32, Device::CPU);
    Tensor weights_tensor(TensorShape({3, 4}), DataType::Float32, Device::CPU);
    Tensor bias_tensor(TensorShape({4}), DataType::Float32, Device::CPU);
    
    // Input: [[1, 2, 3]]
    float* input_data = static_cast<float*>(input_tensor.data());
    input_data[0] = 1.0f;
    input_data[1] = 2.0f;
    input_data[2] = 3.0f;
    
    // Weights: [[1, 0, 0, 0],
    //           [0, 1, 0, 0],
    //           [0, 0, 1, 0]]
    // This creates logits = [1, 2, 3, 0]
    float* weights_data = static_cast<float*>(weights_tensor.data());
    for (int i = 0; i < 12; ++i) weights_data[i] = 0.0f;
    weights_data[0] = 1.0f;  // [0,0]
    weights_data[5] = 1.0f;  // [1,1]
    weights_data[10] = 1.0f; // [2,2]
    
    // Bias: [0, 0, 0, 1]
    // After add: [1, 2, 3, 1]
    float* bias_data = static_cast<float*>(bias_tensor.data());
    bias_data[0] = 0.0f;
    bias_data[1] = 0.0f;
    bias_data[2] = 0.0f;
    bias_data[3] = 1.0f;
    
    // Execute
    // After MatMul: [[1, 2, 3, 0]]
    // After Add: [[1, 2, 3, 1]]
    // After ReLU: [[1, 2, 3, 1]] (all positive, no change)
    // After Softmax: probabilities that sum to 1
    
    auto results = graph.execute({&input_tensor, &weights_tensor, &bias_tensor}, Backend::CPU);
    
    // Validate
    assert(results.size() == 1);
    assert(results[0]->shape().ndim() == 2);
    assert(results[0]->shape().dim(0) == 1);
    assert(results[0]->shape().dim(1) == 4);
    
    const float* out_data = static_cast<const float*>(results[0]->data());
    
    // Verify it's a valid probability distribution
    float sum = 0.0f;
    for (int i = 0; i < 4; ++i) {
        assert(out_data[i] >= 0.0f && out_data[i] <= 1.0f);
        sum += out_data[i];
    }
    assert(std::abs(sum - 1.0f) < TOLERANCE);
    
    // The maximum probability should be at index 2 (value 3 is largest)
    int max_idx = 0;
    float max_prob = out_data[0];
    for (int i = 1; i < 4; ++i) {
        if (out_data[i] > max_prob) {
            max_prob = out_data[i];
            max_idx = i;
        }
    }
    assert(max_idx == 2);
    
    std::cout << "  ✓ Classification pipeline passed" << std::endl;
}

void test_graph_multi_sample_classification() {
    std::cout << "Testing Graph with multi-sample classification..." << std::endl;
    
    // Multiple samples in batch
    // Input: [2, 3] (2 samples, 3 features each)
    // Weights: [3, 4]
    // Bias: [4]
    // Output: [2, 4] (2 samples, 4 class probabilities each)
    
    Graph graph;
    auto input = graph.addInput(TensorShape({2, 3}), DataType::Float32, "Input");
    auto weights = graph.addInput(TensorShape({3, 4}), DataType::Float32, "Weights");
    auto bias = graph.addInput(TensorShape({4}), DataType::Float32, "Bias");
    
    auto matmul_out = graph.addNode(
        std::make_unique<MatMulOp>(),
        {input, weights},
        "MatMul");
    
    auto add_out = graph.addNode(
        std::make_unique<AddOp>(),
        {matmul_out[0], bias},
        "Add");
    
    auto softmax_out = graph.addNode(
        std::make_unique<SoftmaxOp>(),
        {add_out[0]},
        "Softmax");
    
    graph.markOutput(softmax_out[0]);
    graph.validate();
    
    // Create inputs
    Tensor input_tensor(TensorShape({2, 3}), DataType::Float32, Device::CPU);
    Tensor weights_tensor(TensorShape({3, 4}), DataType::Float32, Device::CPU);
    Tensor bias_tensor(TensorShape({4}), DataType::Float32, Device::CPU);
    
    float* input_data = static_cast<float*>(input_tensor.data());
    // Sample 0: [1, 0, 0]
    input_data[0] = 1.0f; input_data[1] = 0.0f; input_data[2] = 0.0f;
    // Sample 1: [0, 0, 1]
    input_data[3] = 0.0f; input_data[4] = 0.0f; input_data[5] = 1.0f;
    
    // Simple identity-like weights
    float* weights_data = static_cast<float*>(weights_tensor.data());
    for (int i = 0; i < 12; ++i) weights_data[i] = 0.0f;
    weights_data[0] = 1.0f;  // [0,0]
    weights_data[10] = 1.0f; // [2,2]
    
    // Zero bias
    float* bias_data = static_cast<float*>(bias_tensor.data());
    for (int i = 0; i < 4; ++i) bias_data[i] = 0.0f;
    
    // Execute
    auto results = graph.execute({&input_tensor, &weights_tensor, &bias_tensor}, Backend::CPU);
    
    // Validate shape
    assert(results.size() == 1);
    assert(results[0]->shape().ndim() == 2);
    assert(results[0]->shape().dim(0) == 2);
    assert(results[0]->shape().dim(1) == 4);
    
    const float* out_data = static_cast<const float*>(results[0]->data());
    
    // Check each sample independently
    for (int sample = 0; sample < 2; ++sample) {
        float sum = 0.0f;
        for (int i = 0; i < 4; ++i) {
            float prob = out_data[sample * 4 + i];
            assert(prob >= 0.0f && prob <= 1.0f);
            sum += prob;
        }
        assert(std::abs(sum - 1.0f) < TOLERANCE);
    }
    
    std::cout << "  ✓ Multi-sample classification passed" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "ForgeRT Graph Integration Tests (Softmax)" << std::endl;
    std::cout << "========================================" << std::endl;

    test_graph_softmax_simple();
    test_graph_classification_pipeline();
    test_graph_multi_sample_classification();

    std::cout << "========================================" << std::endl;
    std::cout << "All tests PASSED ✓" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}

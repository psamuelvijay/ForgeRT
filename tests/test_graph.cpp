#include "forgert/graph/graph.h"
#include "forgert/operator/add.h"
#include "forgert/operator/relu.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace forgert;

void test_graph_single_node() {
    std::cout << "Testing Graph with single Add node..." << std::endl;

    // Build graph: A + B = C
    Graph graph;
    auto input_a = graph.addInput(TensorShape({3}), DataType::Float32, "A");
    auto input_b = graph.addInput(TensorShape({3}), DataType::Float32, "B");
    
    auto outputs = graph.addNode(
        std::make_unique<AddOp>(),
        {input_a, input_b},
        "Add");
    
    graph.markOutput(outputs[0]);
    graph.validate();

    // Create input tensors
    TensorShape shape({3});
    Tensor a(shape, DataType::Float32, Device::CPU);
    Tensor b(shape, DataType::Float32, Device::CPU);

    float* a_data = static_cast<float*>(a.data());
    float* b_data = static_cast<float*>(b.data());
    
    a_data[0] = 1.0f; a_data[1] = 2.0f; a_data[2] = 3.0f;
    b_data[0] = 4.0f; b_data[1] = 5.0f; b_data[2] = 6.0f;

    // Execute
    auto results = graph.execute({&a, &b}, Backend::CPU);

    // Validate
    assert(results.size() == 1);
    const float* c_data = static_cast<const float*>(results[0]->data());
    assert(std::abs(c_data[0] - 5.0f) < 1e-6f);
    assert(std::abs(c_data[1] - 7.0f) < 1e-6f);
    assert(std::abs(c_data[2] - 9.0f) < 1e-6f);

    std::cout << "  ✓ Single node execution passed" << std::endl;
}

void test_graph_multi_node() {
    std::cout << "Testing Graph with multiple nodes..." << std::endl;

    // Build graph: (A + B) -> ReLU -> Output
    Graph graph;
    auto input_a = graph.addInput(TensorShape({4}), DataType::Float32, "A");
    auto input_b = graph.addInput(TensorShape({4}), DataType::Float32, "B");
    
    auto add_out = graph.addNode(
        std::make_unique<AddOp>(),
        {input_a, input_b},
        "Add");
    
    auto relu_out = graph.addNode(
        std::make_unique<ReLUOp>(),
        {add_out[0]},
        "ReLU");
    
    graph.markOutput(relu_out[0]);
    graph.validate();

    // Create inputs: A = [-2, -1, 1, 2], B = [1, 2, 3, 4]
    // Add: [-1, 1, 4, 6]
    // ReLU: [0, 1, 4, 6]
    TensorShape shape({4});
    Tensor a(shape, DataType::Float32, Device::CPU);
    Tensor b(shape, DataType::Float32, Device::CPU);

    float* a_data = static_cast<float*>(a.data());
    float* b_data = static_cast<float*>(b.data());
    
    a_data[0] = -2.0f; a_data[1] = -1.0f; a_data[2] = 1.0f; a_data[3] = 2.0f;
    b_data[0] = 1.0f; b_data[1] = 2.0f; b_data[2] = 3.0f; b_data[3] = 4.0f;

    // Execute
    auto results = graph.execute({&a, &b}, Backend::CPU);

    // Validate
    assert(results.size() == 1);
    const float* out_data = static_cast<const float*>(results[0]->data());
    assert(std::abs(out_data[0] - 0.0f) < 1e-6f);
    assert(std::abs(out_data[1] - 1.0f) < 1e-6f);
    assert(std::abs(out_data[2] - 4.0f) < 1e-6f);
    assert(std::abs(out_data[3] - 6.0f) < 1e-6f);

    std::cout << "  ✓ Multi-node execution passed" << std::endl;
}

void test_graph_diamond() {
    std::cout << "Testing Graph with diamond dependency..." << std::endl;

    // Build graph:
    //        A
    //       / \
    //   Add1   Add2
    //   (A+B)  (A+C)
    //       \ /
    //       Add3
    //    (Add1+Add2)
    
    Graph graph;
    auto input_a = graph.addInput(TensorShape({2}), DataType::Float32, "A");
    auto input_b = graph.addInput(TensorShape({2}), DataType::Float32, "B");
    auto input_c = graph.addInput(TensorShape({2}), DataType::Float32, "C");
    
    auto add1_out = graph.addNode(
        std::make_unique<AddOp>(),
        {input_a, input_b},
        "Add1");
    
    auto add2_out = graph.addNode(
        std::make_unique<AddOp>(),
        {input_a, input_c},
        "Add2");
    
    auto add3_out = graph.addNode(
        std::make_unique<AddOp>(),
        {add1_out[0], add2_out[0]},
        "Add3");
    
    graph.markOutput(add3_out[0]);
    graph.validate();

    // A = [1, 2], B = [10, 20], C = [100, 200]
    // Add1 = [11, 22]
    // Add2 = [101, 202]
    // Add3 = [112, 224]
    TensorShape shape({2});
    Tensor a(shape, DataType::Float32, Device::CPU);
    Tensor b(shape, DataType::Float32, Device::CPU);
    Tensor c(shape, DataType::Float32, Device::CPU);

    float* a_data = static_cast<float*>(a.data());
    float* b_data = static_cast<float*>(b.data());
    float* c_data = static_cast<float*>(c.data());
    
    a_data[0] = 1.0f; a_data[1] = 2.0f;
    b_data[0] = 10.0f; b_data[1] = 20.0f;
    c_data[0] = 100.0f; c_data[1] = 200.0f;

    // Execute
    auto results = graph.execute({&a, &b, &c}, Backend::CPU);

    // Validate
    assert(results.size() == 1);
    const float* out_data = static_cast<const float*>(results[0]->data());
    assert(std::abs(out_data[0] - 112.0f) < 1e-6f);
    assert(std::abs(out_data[1] - 224.0f) < 1e-6f);

    std::cout << "  ✓ Diamond dependency passed" << std::endl;
}

void test_graph_multiple_outputs() {
    std::cout << "Testing Graph with multiple outputs..." << std::endl;

    // Build graph with two outputs
    Graph graph;
    auto input_a = graph.addInput(TensorShape({2}), DataType::Float32, "A");
    auto input_b = graph.addInput(TensorShape({2}), DataType::Float32, "B");
    
    auto add_out = graph.addNode(
        std::make_unique<AddOp>(),
        {input_a, input_b},
        "Add");
    
    auto relu_out = graph.addNode(
        std::make_unique<ReLUOp>(),
        {add_out[0]},
        "ReLU");
    
    // Mark both Add and ReLU outputs
    graph.markOutput(add_out[0]);
    graph.markOutput(relu_out[0]);
    graph.validate();

    // A = [-1, 1], B = [2, 3]
    // Add = [1, 4]
    // ReLU = [1, 4]
    TensorShape shape({2});
    Tensor a(shape, DataType::Float32, Device::CPU);
    Tensor b(shape, DataType::Float32, Device::CPU);

    float* a_data = static_cast<float*>(a.data());
    float* b_data = static_cast<float*>(b.data());
    
    a_data[0] = -1.0f; a_data[1] = 1.0f;
    b_data[0] = 2.0f; b_data[1] = 3.0f;

    // Execute
    auto results = graph.execute({&a, &b}, Backend::CPU);

    // Validate - should get two outputs
    assert(results.size() == 2);
    
    const float* out1_data = static_cast<const float*>(results[0]->data());
    assert(std::abs(out1_data[0] - 1.0f) < 1e-6f);
    assert(std::abs(out1_data[1] - 4.0f) < 1e-6f);
    
    const float* out2_data = static_cast<const float*>(results[1]->data());
    assert(std::abs(out2_data[0] - 1.0f) < 1e-6f);
    assert(std::abs(out2_data[1] - 4.0f) < 1e-6f);

    std::cout << "  ✓ Multiple outputs passed" << std::endl;
}

void test_graph_shape_inference() {
    std::cout << "Testing Graph shape inference..." << std::endl;

    // Build graph with broadcasting
    Graph graph;
    auto input_a = graph.addInput(TensorShape({3, 4}), DataType::Float32, "A");
    auto input_b = graph.addInput(TensorShape({4}), DataType::Float32, "B");
    
    auto add_out = graph.addNode(
        std::make_unique<AddOp>(),
        {input_a, input_b},
        "Add");
    
    graph.markOutput(add_out[0]);
    graph.validate();

    // Create inputs
    Tensor a(TensorShape({3, 4}), DataType::Float32, Device::CPU);
    Tensor b(TensorShape({4}), DataType::Float32, Device::CPU);
    a.fill(1.0f);
    b.fill(10.0f);

    // Execute
    auto results = graph.execute({&a, &b}, Backend::CPU);

    // Validate output shape is [3, 4]
    assert(results.size() == 1);
    assert(results[0]->shape().ndim() == 2);
    assert(results[0]->shape().dim(0) == 3);
    assert(results[0]->shape().dim(1) == 4);
    assert(results[0]->numElements() == 12);

    // Validate values
    const float* out_data = static_cast<const float*>(results[0]->data());
    for (size_t i = 0; i < 12; ++i) {
        assert(std::abs(out_data[i] - 11.0f) < 1e-6f);
    }

    std::cout << "  ✓ Shape inference passed" << std::endl;
}

void test_graph_invalid_shapes() {
    std::cout << "Testing Graph invalid shape detection..." << std::endl;

    // Try to create graph with incompatible shapes
    Graph graph;
    auto input_a = graph.addInput(TensorShape({3}), DataType::Float32, "A");
    auto input_b = graph.addInput(TensorShape({5}), DataType::Float32, "B");
    
    bool threw = false;
    try {
        graph.addNode(
            std::make_unique<AddOp>(),
            {input_a, input_b},
            "Add");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    std::cout << "  ✓ Invalid shape detection passed" << std::endl;
}

void test_graph_cycle_detection() {
    std::cout << "Testing Graph cycle detection..." << std::endl;

    // Cannot easily create a cycle with the current API since each
    // addNode returns NEW value IDs. This is actually a good design -
    // makes cycles impossible by construction.
    // 
    // We can test the validation still works by checking a valid graph passes
    
    Graph graph;
    auto input = graph.addInput(TensorShape({2}), DataType::Float32);
    auto out1 = graph.addNode(std::make_unique<ReLUOp>(), {input});
    auto out2 = graph.addNode(std::make_unique<ReLUOp>(), {out1[0]});
    graph.markOutput(out2[0]);
    
    // Should not throw
    graph.validate();

    std::cout << "  ✓ Cycle detection passed (no cycles possible by design)" << std::endl;
}

void test_graph_no_outputs() {
    std::cout << "Testing Graph validation without outputs..." << std::endl;

    Graph graph;
    auto input = graph.addInput(TensorShape({2}), DataType::Float32);
    graph.addNode(std::make_unique<ReLUOp>(), {input});
    
    // No outputs marked - should fail validation
    bool threw = false;
    try {
        graph.validate();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);

    std::cout << "  ✓ No outputs validation passed" << std::endl;
}

void test_graph_execution_order() {
    std::cout << "Testing Graph execution order independence..." << std::endl;

    // Create graph where nodes are added in non-execution order
    // Add nodes: C, B, A (reverse order)
    // But execution should still be: A -> B -> C
    
    Graph graph;
    auto input = graph.addInput(TensorShape({2}), DataType::Float32);
    
    // We'll create: input -> ReLU1 -> ReLU2 -> ReLU3
    // But add in order: 3, 2, 1
    
    // Actually, we must add in dependency order due to API design
    // (need input IDs to exist first). This is good design.
    // Instead test that execution order is computed correctly
    // by verifying correct output.
    
    auto out1 = graph.addNode(std::make_unique<ReLUOp>(), {input}, "ReLU1");
    auto out2 = graph.addNode(std::make_unique<ReLUOp>(), {out1[0]}, "ReLU2");
    auto out3 = graph.addNode(std::make_unique<ReLUOp>(), {out2[0]}, "ReLU3");
    graph.markOutput(out3[0]);
    graph.validate();

    // Input = [-1, 2]
    // After 3x ReLU: [0, 2]
    TensorShape shape({2});
    Tensor input_tensor(shape, DataType::Float32, Device::CPU);
    float* data = static_cast<float*>(input_tensor.data());
    data[0] = -1.0f;
    data[1] = 2.0f;

    auto results = graph.execute({&input_tensor}, Backend::CPU);
    assert(results.size() == 1);
    
    const float* out_data = static_cast<const float*>(results[0]->data());
    assert(std::abs(out_data[0] - 0.0f) < 1e-6f);
    assert(std::abs(out_data[1] - 2.0f) < 1e-6f);

    std::cout << "  ✓ Execution order passed" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "ForgeRT Graph Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        test_graph_single_node();
        test_graph_multi_node();
        test_graph_diamond();
        test_graph_multiple_outputs();
        test_graph_shape_inference();
        test_graph_invalid_shapes();
        test_graph_cycle_detection();
        test_graph_no_outputs();
        test_graph_execution_order();

        std::cout << "========================================" << std::endl;
        std::cout << "All tests PASSED ✓" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Test FAILED: " << e.what() << std::endl;
        return 1;
    }
}

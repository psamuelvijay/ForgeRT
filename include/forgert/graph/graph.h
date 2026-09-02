#pragma once

#include "forgert/operator/operator.h"
#include "forgert/tensor/tensor.h"
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace forgert {

/**
 * @brief A node in the computation graph
 * 
 * Represents a single operation in the graph.
 * Has input value IDs and output value IDs.
 */
struct GraphNode {
    std::unique_ptr<Operator> op;           // The operator this node executes
    std::vector<size_t> input_ids;          // IDs of input values
    std::vector<size_t> output_ids;         // IDs of output values
    std::string name;                       // Optional name for debugging

    GraphNode(std::unique_ptr<Operator> op_,
              std::vector<size_t> inputs,
              std::vector<size_t> outputs,
              std::string name_ = "")
        : op(std::move(op_))
        , input_ids(std::move(inputs))
        , output_ids(std::move(outputs))
        , name(std::move(name_)) {}
};

/**
 * @brief Computation graph for neural network inference
 * 
 * Represents a DAG of operations on tensors.
 * 
 * Design:
 * - Values are identified by integer IDs
 * - Graph inputs are values 0..N-1
 * - Each node produces new values
 * - Graph outputs are specified value IDs
 * - Nodes store operators (unique ownership)
 * - Execution allocates and owns intermediate tensors
 * 
 * Usage:
 *   1. Create graph
 *   2. Add nodes with operators
 *   3. Mark outputs
 *   4. Validate
 *   5. Execute with inputs
 */
class Graph {
public:
    Graph() : next_value_id_(0) {}

    // Disable copy, enable move
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    Graph(Graph&&) = default;
    Graph& operator=(Graph&&) = default;

    /**
     * @brief Add a graph input
     * 
     * Graph inputs are external tensors provided at execution time.
     * Each input gets a unique value ID.
     * 
     * @param shape Shape of the input tensor
     * @param dtype Data type of the input
     * @param name Optional name for debugging
     * @return Value ID for this input
     */
    size_t addInput(const TensorShape& shape, DataType dtype, const std::string& name = "") {
        size_t id = next_value_id_++;
        input_ids_.push_back(id);
        value_shapes_[id] = shape;
        value_dtypes_[id] = dtype;
        if (!name.empty()) {
            value_names_[id] = name;
        }
        return id;
    }

    /**
     * @brief Add a node to the graph
     * 
     * The node takes input values and produces output values.
     * Output shapes are inferred from the operator.
     * 
     * @param op Operator to execute (graph takes ownership)
     * @param input_ids IDs of input values
     * @param name Optional name for debugging
     * @return Vector of output value IDs
     */
    std::vector<size_t> addNode(
        std::unique_ptr<Operator> op,
        const std::vector<size_t>& input_ids,
        const std::string& name = "") {
        
        // Validate inputs exist
        for (size_t id : input_ids) {
            if (value_shapes_.find(id) == value_shapes_.end()) {
                throw std::invalid_argument(
                    "Graph::addNode: input value " + std::to_string(id) + " does not exist");
            }
        }

        // Infer output shapes
        std::vector<TensorShape> input_shapes;
        for (size_t id : input_ids) {
            input_shapes.push_back(value_shapes_[id]);
        }

        std::vector<TensorShape> output_shapes;
        try {
            output_shapes = op->inferOutputShapes(input_shapes);
        } catch (const std::exception& e) {
            throw std::invalid_argument(
                "Graph::addNode: shape inference failed for " + op->name() + ": " + e.what());
        }

        // Allocate value IDs for outputs
        std::vector<size_t> output_ids;
        for (size_t i = 0; i < output_shapes.size(); ++i) {
            size_t id = next_value_id_++;
            output_ids.push_back(id);
            value_shapes_[id] = output_shapes[i];
            
            // Assume same dtype as first input for now
            // More sophisticated handling can be added later
            if (!input_ids.empty()) {
                value_dtypes_[id] = value_dtypes_[input_ids[0]];
            }
        }

        // Create node
        nodes_.emplace_back(std::make_unique<GraphNode>(
            std::move(op), input_ids, output_ids, name));

        return output_ids;
    }

    /**
     * @brief Mark a value as a graph output
     * 
     * @param value_id ID of the value to output
     */
    void markOutput(size_t value_id) {
        if (value_shapes_.find(value_id) == value_shapes_.end()) {
            throw std::invalid_argument(
                "Graph::markOutput: value " + std::to_string(value_id) + " does not exist");
        }
        output_ids_.push_back(value_id);
    }

    /**
     * @brief Validate the graph structure
     * 
     * Checks:
     * - All values referenced are defined
     * - No cycles
     * - At least one output marked
     * 
     * @throws std::runtime_error if graph is invalid
     */
    void validate() {
        if (output_ids_.empty()) {
            throw std::runtime_error("Graph::validate: no outputs marked");
        }

        // Check for cycles using topological sort
        try {
            computeExecutionOrder();
        } catch (const std::runtime_error& e) {
            throw std::runtime_error(std::string("Graph::validate: ") + e.what());
        }
    }

    /**
     * @brief Execute the graph with given inputs
     * 
     * @param inputs Input tensors (must match addInput order and shapes)
     * @param backend Backend to use for execution (CPU or CUDA)
     * @return Output tensors (in markOutput order)
     */
    std::vector<std::unique_ptr<Tensor>> execute(
        const std::vector<const Tensor*>& inputs,
        Backend backend = Backend::CPU) {
        
        // Validate input count
        if (inputs.size() != input_ids_.size()) {
            throw std::invalid_argument(
                "Graph::execute: expected " + std::to_string(input_ids_.size()) +
                " inputs, got " + std::to_string(inputs.size()));
        }

        // Validate input shapes
        for (size_t i = 0; i < inputs.size(); ++i) {
            size_t value_id = input_ids_[i];
            if (inputs[i]->shape().numElements() != value_shapes_[value_id].numElements()) {
                throw std::invalid_argument(
                    "Graph::execute: input " + std::to_string(i) + " shape mismatch");
            }
        }

        // Compute execution order
        auto exec_order = computeExecutionOrder();

        // Value storage: maps value ID to tensor
        std::unordered_map<size_t, Tensor*> value_storage;

        // Map inputs
        for (size_t i = 0; i < inputs.size(); ++i) {
            value_storage[input_ids_[i]] = const_cast<Tensor*>(inputs[i]);
        }

        // Storage for intermediate and output tensors
        std::vector<std::unique_ptr<Tensor>> owned_tensors;

        // Execute nodes in order
        for (size_t node_idx : exec_order) {
            const auto& node = nodes_[node_idx];

            // Gather input tensors
            std::vector<const Tensor*> node_inputs;
            for (size_t input_id : node->input_ids) {
                if (value_storage.find(input_id) == value_storage.end()) {
                    throw std::runtime_error(
                        "Graph::execute: value " + std::to_string(input_id) + " not computed");
                }
                node_inputs.push_back(value_storage[input_id]);
            }

            // Allocate output tensors
            std::vector<Tensor*> node_outputs;
            for (size_t output_id : node->output_ids) {
                const auto& shape = value_shapes_[output_id];
                const auto& dtype = value_dtypes_[output_id];
                
                Device device = (backend == Backend::CPU) ? Device::CPU : Device::CUDA;
                owned_tensors.emplace_back(std::make_unique<Tensor>(shape, dtype, device));
                
                Tensor* output_ptr = owned_tensors.back().get();
                value_storage[output_id] = output_ptr;
                node_outputs.push_back(output_ptr);
            }

            // Execute operator
            node->op->execute(backend, node_inputs, node_outputs);
        }

        // Gather outputs
        std::vector<std::unique_ptr<Tensor>> outputs;
        for (size_t output_id : output_ids_) {
            // Find the tensor in owned_tensors and move it out
            Tensor* output_ptr = value_storage[output_id];
            
            // Find and move from owned_tensors
            for (auto it = owned_tensors.begin(); it != owned_tensors.end(); ++it) {
                if (it->get() == output_ptr) {
                    outputs.push_back(std::move(*it));
                    owned_tensors.erase(it);
                    break;
                }
            }
        }

        return outputs;
    }

    // Getters for introspection
    size_t numInputs() const { return input_ids_.size(); }
    size_t numOutputs() const { return output_ids_.size(); }
    size_t numNodes() const { return nodes_.size(); }
    const std::vector<size_t>& getInputIds() const { return input_ids_; }
    const std::vector<size_t>& getOutputIds() const { return output_ids_; }

private:
    /**
     * @brief Compute topological execution order
     * 
     * Uses Kahn's algorithm for topological sort.
     * Detects cycles.
     * 
     * @return Vector of node indices in execution order
     * @throws std::runtime_error if graph has cycles
     */
    std::vector<size_t> computeExecutionOrder() const {
        size_t n = nodes_.size();
        
        // Build dependency graph: value_id -> nodes that produce it
        std::unordered_map<size_t, size_t> value_producer;
        for (size_t i = 0; i < n; ++i) {
            for (size_t output_id : nodes_[i]->output_ids) {
                value_producer[output_id] = i;
            }
        }

        // Build adjacency list: node -> dependent nodes
        std::vector<std::vector<size_t>> adj(n);
        std::vector<size_t> in_degree(n, 0);

        for (size_t i = 0; i < n; ++i) {
            for (size_t input_id : nodes_[i]->input_ids) {
                // If this input is produced by another node, add dependency
                auto it = value_producer.find(input_id);
                if (it != value_producer.end()) {
                    size_t producer = it->second;
                    adj[producer].push_back(i);
                    in_degree[i]++;
                }
            }
        }

        // Kahn's algorithm
        std::vector<size_t> order;
        std::vector<size_t> queue;

        // Start with nodes that have no dependencies
        for (size_t i = 0; i < n; ++i) {
            if (in_degree[i] == 0) {
                queue.push_back(i);
            }
        }

        while (!queue.empty()) {
            size_t node = queue.back();
            queue.pop_back();
            order.push_back(node);

            for (size_t dependent : adj[node]) {
                in_degree[dependent]--;
                if (in_degree[dependent] == 0) {
                    queue.push_back(dependent);
                }
            }
        }

        if (order.size() != n) {
            throw std::runtime_error("cycle detected in graph");
        }

        return order;
    }

    std::vector<std::unique_ptr<GraphNode>> nodes_;
    std::vector<size_t> input_ids_;
    std::vector<size_t> output_ids_;
    
    size_t next_value_id_;
    std::unordered_map<size_t, TensorShape> value_shapes_;
    std::unordered_map<size_t, DataType> value_dtypes_;
    std::unordered_map<size_t, std::string> value_names_;
};

} // namespace forgert

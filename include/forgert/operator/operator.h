#pragma once

#include "forgert/tensor/tensor.h"
#include <vector>
#include <string>
#include <memory>

namespace forgert {

/**
 * @brief Backend type for operator execution
 */
enum class Backend {
    CPU,
    CUDA
};

/**
 * @brief Abstract base class for all operators
 * 
 * An operator transforms input tensors into output tensors.
 * Each operator can have multiple backend implementations (CPU, CUDA).
 * 
 * Design principles:
 * - Operators are stateless functions (parameters can be stored separately)
 * - Clear input/output contracts via shape inference
 * - Backend selection is explicit (scheduler decides, not the operator)
 * - Operators validate their inputs
 */
class Operator {
public:
    virtual ~Operator() = default;

    /**
     * @brief Get the operator name for debugging/profiling
     */
    virtual std::string name() const = 0;

    /**
     * @brief Infer output shapes from input shapes
     * 
     * This allows graph construction without allocating tensors.
     * 
     * @param input_shapes Shapes of input tensors
     * @return Shapes of output tensors
     * @throws std::invalid_argument if inputs are incompatible
     */
    virtual std::vector<TensorShape> inferOutputShapes(
        const std::vector<TensorShape>& input_shapes) const = 0;

    /**
     * @brief Check if this operator supports a given backend
     * 
     * @param backend Backend to check
     * @return true if supported, false otherwise
     */
    virtual bool supportsBackend(Backend backend) const = 0;

    /**
     * @brief Execute the operator on CPU
     * 
     * @param inputs Input tensors (must be on CPU)
     * @param outputs Output tensors (must be on CPU, pre-allocated)
     * @throws std::runtime_error if not supported or inputs invalid
     */
    virtual void executeCPU(
        const std::vector<const Tensor*>& inputs,
        const std::vector<Tensor*>& outputs) = 0;

    /**
     * @brief Execute the operator on CUDA
     * 
     * Phase 3: CUDA implementation
     * For now, all operators should throw std::runtime_error
     * 
     * @param inputs Input tensors (must be on CUDA device)
     * @param outputs Output tensors (must be on CUDA device, pre-allocated)
     * @throws std::runtime_error if not supported or inputs invalid
     */
    virtual void executeCUDA(
        const std::vector<const Tensor*>& inputs,
        const std::vector<Tensor*>& outputs) {
        (void)inputs;
        (void)outputs;
        throw std::runtime_error(name() + ": CUDA execution not implemented (Phase 3)");
    }

    /**
     * @brief Execute the operator on the appropriate backend
     * 
     * Convenience method that dispatches to executeCPU or executeCUDA.
     * 
     * @param backend Backend to use
     * @param inputs Input tensors
     * @param outputs Output tensors (pre-allocated)
     */
    void execute(
        Backend backend,
        const std::vector<const Tensor*>& inputs,
        const std::vector<Tensor*>& outputs) {
        
        if (!supportsBackend(backend)) {
            throw std::runtime_error(name() + ": backend not supported");
        }

        // Validate inputs and outputs match expected shapes
        auto expected_shapes = inferOutputShapes(getShapes(inputs));
        if (expected_shapes.size() != outputs.size()) {
            throw std::invalid_argument(
                name() + ": output count mismatch (expected " +
                std::to_string(expected_shapes.size()) + ", got " +
                std::to_string(outputs.size()) + ")");
        }

        for (size_t i = 0; i < outputs.size(); ++i) {
            if (outputs[i]->shape().numElements() != expected_shapes[i].numElements()) {
                throw std::invalid_argument(
                    name() + ": output " + std::to_string(i) +
                    " shape mismatch (expected " +
                    std::to_string(expected_shapes[i].numElements()) +
                    " elements, got " +
                    std::to_string(outputs[i]->shape().numElements()) + ")");
            }
        }

        switch (backend) {
            case Backend::CPU:
                executeCPU(inputs, outputs);
                break;
            case Backend::CUDA:
                executeCUDA(inputs, outputs);
                break;
        }
    }

protected:
    /**
     * @brief Helper: extract shapes from tensors
     */
    static std::vector<TensorShape> getShapes(const std::vector<const Tensor*>& tensors) {
        std::vector<TensorShape> shapes;
        shapes.reserve(tensors.size());
        for (const auto* t : tensors) {
            shapes.push_back(t->shape());
        }
        return shapes;
    }

    /**
     * @brief Helper: validate tensor count
     */
    static void validateInputCount(
        const std::string& op_name,
        size_t expected,
        size_t actual) {
        if (actual != expected) {
            throw std::invalid_argument(
                op_name + ": expected " + std::to_string(expected) +
                " inputs, got " + std::to_string(actual));
        }
    }

    /**
     * @brief Helper: validate tensor device
     */
    static void validateDevice(
        const std::string& op_name,
        const Tensor* tensor,
        Device expected) {
        if (tensor->device() != expected) {
            throw std::invalid_argument(
                op_name + ": tensor on wrong device");
        }
    }

    /**
     * @brief Helper: validate tensor dtype
     */
    static void validateDtype(
        const std::string& op_name,
        const Tensor* tensor,
        DataType expected) {
        if (tensor->dtype() != expected) {
            throw std::invalid_argument(
                op_name + ": expected dtype " + toString(expected) +
                ", got " + toString(tensor->dtype()));
        }
    }
};

} // namespace forgert

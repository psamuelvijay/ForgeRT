#pragma once

#include "forgert/operator/operator.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace forgert {

/**
 * @brief Reshape operator with ONNX 0 / -1 wildcard semantics (CPU, float32).
 *
 * Wildcard rules (ONNX Reshape, allowzero = 0):
 *   - `0`  at dim i copies the input's dim i (error if the input has no dim i)
 *   - `-1` at dim i is inferred from the remaining elements (at most one -1;
 *          the known dims must divide the element count exactly)
 *   - other values must be positive; any other negative value is an error
 *   - without -1, the product of the resolved dims must equal the element count
 *
 * Two modes:
 *
 *   1. Fixed target: `ReshapeOp(std::vector<int64_t> target)`.
 *      One input (data). The target shape is part of the operator, so
 *      inferOutputShapes() resolves the exact output shape and Graph allocates
 *      the output correctly (e.g. [1,1000,1,1] + [0,-1] -> exactly [1,1000]).
 *      This is the mode the ONNX importer uses; the ONNX shape tensor must be
 *      a constant initializer.
 *
 *   2. Shape tensor: `ReshapeOp()`.
 *      Two inputs: data (Float32) and a 1-D Int32 shape tensor read at execute
 *      time. Because shape inference only sees shapes, not values,
 *      inferOutputShapes() returns a flat shape with the matching element
 *      count; callers that need the exact shape must pre-allocate the output.
 *
 * Both modes share resolveShape() and copy the data bytes unchanged.
 * CPU-only as a Phase 8 scope decision; a CUDA backend is not precluded.
 */
class ReshapeOp : public Operator {
public:
    /// Shape-tensor mode (two inputs).
    ReshapeOp() = default;

    /// Fixed-target mode (one input). Values are validated against the input
    /// shape during shape inference; values below -1 are rejected immediately.
    explicit ReshapeOp(std::vector<int64_t> target)
        : has_target_(true), target_(std::move(target)) {
        if (target_.empty())
            throw std::invalid_argument("ReshapeOp: target shape must have at least one dim");
        for (size_t i = 0; i < target_.size(); ++i) {
            if (target_[i] < -1)
                throw std::invalid_argument(
                    "ReshapeOp: target value " + std::to_string(target_[i]) +
                    " at dim " + std::to_string(i) +
                    " is invalid (must be 0, -1, or positive)");
        }
    }

    std::string name() const override { return "Reshape"; }

    std::unique_ptr<Operator> clone() const override {
        return has_target_ ? std::make_unique<ReshapeOp>(target_)
                           : std::make_unique<ReshapeOp>();
    }

    bool supportsBackend(Backend backend) const override {
        return backend == Backend::CPU;
    }

    bool hasFixedTarget() const { return has_target_; }
    const std::vector<int64_t>& target() const { return target_; }

    /**
     * @brief Resolve ONNX wildcards in `dims` against `input`.
     * @throws std::invalid_argument on any rule violation.
     */
    static TensorShape resolveShape(const TensorShape& input,
                                    const int64_t* dims, size_t rank) {
        if (rank == 0)
            throw std::invalid_argument("ReshapeOp: target shape must have at least one dim");

        const size_t input_elems = input.numElements();
        const size_t input_ndim  = input.ndim();
        std::vector<size_t> resolved(rank, 0);
        long long unknown_idx = -1;
        size_t running_product = 1;

        for (size_t i = 0; i < rank; ++i) {
            const int64_t d = dims[i];
            if (d == 0) {
                if (i >= input_ndim)
                    throw std::invalid_argument(
                        "ReshapeOp: wildcard 0 at dim " + std::to_string(i) +
                        " but input has only " + std::to_string(input_ndim) +
                        " dimensions");
                resolved[i] = input.dim(i);
                running_product *= resolved[i];
            } else if (d == -1) {
                if (unknown_idx >= 0)
                    throw std::invalid_argument(
                        "ReshapeOp: at most one -1 wildcard allowed (found second at dim " +
                        std::to_string(i) + ")");
                unknown_idx = static_cast<long long>(i);
            } else if (d > 0) {
                resolved[i] = static_cast<size_t>(d);
                running_product *= resolved[i];
            } else {
                throw std::invalid_argument(
                    "ReshapeOp: shape value " + std::to_string(d) + " at dim " +
                    std::to_string(i) + " is invalid (must be 0, -1, or positive)");
            }
        }

        if (unknown_idx >= 0) {
            if (running_product == 0 || input_elems % running_product != 0)
                throw std::invalid_argument(
                    "ReshapeOp: cannot resolve -1 wildcard: input has " +
                    std::to_string(input_elems) + " elements but known dims product is " +
                    std::to_string(running_product));
            resolved[static_cast<size_t>(unknown_idx)] = input_elems / running_product;
        } else if (running_product != input_elems) {
            throw std::invalid_argument(
                "ReshapeOp: element count mismatch: shape product is " +
                std::to_string(running_product) + " but input has " +
                std::to_string(input_elems) + " elements");
        }
        return TensorShape(resolved);
    }

    // ------------------------------------------------------------------
    // Shape inference
    // ------------------------------------------------------------------
    std::vector<TensorShape> inferOutputShapes(
        const std::vector<TensorShape>& input_shapes) const override
    {
        if (has_target_) {
            validateInputCount("ReshapeOp (fixed target)", 1, input_shapes.size());
            return { resolveShape(input_shapes[0], target_.data(), target_.size()) };
        }

        validateInputCount("ReshapeOp", 2, input_shapes.size());
        if (input_shapes[1].ndim() != 1)
            throw std::invalid_argument(
                "ReshapeOp: shape tensor must be 1-D, got " +
                std::to_string(input_shapes[1].ndim()) + "-D");
        // Values unavailable here: return a flat shape with matching count.
        return { TensorShape({input_shapes[0].numElements()}) };
    }

    // ------------------------------------------------------------------
    // CPU execution
    // ------------------------------------------------------------------
    void executeCPU(
        const std::vector<const Tensor*>& inputs,
        const std::vector<Tensor*>&       outputs) override
    {
        validateInputCount(has_target_ ? "ReshapeOp (fixed target)" : "ReshapeOp",
                           has_target_ ? 1 : 2, inputs.size());
        validateInputCount("ReshapeOp outputs", 1, outputs.size());

        const Tensor* data = inputs[0];
        Tensor*       output = outputs[0];
        validateDevice("ReshapeOp", data,   Device::CPU);
        validateDevice("ReshapeOp", output, Device::CPU);
        validateDtype ("ReshapeOp", data,   DataType::Float32);
        validateDtype ("ReshapeOp", output, DataType::Float32);

        if (has_target_) {
            const TensorShape expected =
                resolveShape(data->shape(), target_.data(), target_.size());
            if (output->shape().dimensions() != expected.dimensions())
                throw std::invalid_argument(
                    "ReshapeOp: output tensor shape does not match the resolved target shape");
        } else {
            const Tensor* shape_tensor = inputs[1];
            validateDevice("ReshapeOp", shape_tensor, Device::CPU);
            validateDtype ("ReshapeOp", shape_tensor, DataType::Int32);
            if (shape_tensor->shape().ndim() != 1)
                throw std::invalid_argument(
                    "ReshapeOp: shape tensor must be 1-D, got " +
                    std::to_string(shape_tensor->shape().ndim()) + "-D");
            const size_t rank = shape_tensor->shape().dim(0);
            const int32_t* raw = static_cast<const int32_t*>(shape_tensor->data());
            std::vector<int64_t> dims(raw, raw + rank);
            (void)resolveShape(data->shape(), dims.data(), rank);
        }

        if (output->numElements() != data->numElements())
            throw std::invalid_argument(
                "ReshapeOp: output has " + std::to_string(output->numElements()) +
                " elements but input has " + std::to_string(data->numElements()));

        std::memcpy(output->data(), data->data(), data->numBytes());
    }

private:
    bool has_target_ = false;
    std::vector<int64_t> target_;
};

} // namespace forgert
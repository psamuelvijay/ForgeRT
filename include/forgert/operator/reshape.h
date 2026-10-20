#pragma once

#include "forgert/operator/operator.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace forgert {

/**
 * @brief Reshape operator — ONNX-compatible with 0 and -1 wildcard semantics.
 *
 * The target shape is provided as the second input tensor (Int32, 1-D).
 * ONNX wildcard rules:
 *   - `0` at dim i: copy the input's dim i (input must have dim i)
 *   - `-1` at dim i: infer this dim from the remaining elements
 *   - At most one `-1` per shape (ONNX constraint)
 *   - Resolved product must equal input.numElements()
 *
 * SqueezeNet 1.1 Reshape: input [1,1000,1,1] + shape [0,-1] -> [1,1000]
 *
 * Architecture note: ForgeRT's Tensor owns its memory and shape immutably.
 * The output tensor is pre-allocated by the caller with the resolved shape
 * (the importer/graph builder resolves wildcards using the shape tensor's
 * data, which is available at graph-construction time for static models).
 * inferOutputShapes returns a 1-D shape {input.numElements()} so the
 * operator.h execute() validation (which checks numElements, not exact
 * shape) passes. executeCPU then validates wildcard semantics and does
 * a byte-identical memcpy — the data layout is unchanged by reshape.
 *
 * CPU implementation: memcpy. No CUDA backend — a CUDA kernel would add
 * 20 us launch overhead for zero compute benefit. A zero-copy view mode
 * (avoiding the memcpy) is a future optimization requiring Tensor API
 * changes; deferred. A CUDA backend is not precluded for future phases
 * but is out of scope for Phase 8.
 *
 * Inputs:  [0] data tensor  (Float32, any shape)
 *          [1] shape tensor (Int32, 1-D)
 * Outputs: [0] output tensor (Float32, resolved shape, same numElements as input)
 */
class ReshapeOp : public Operator {
public:
    std::string name() const override { return "Reshape"; }

    std::unique_ptr<Operator> clone() const override {
        return std::make_unique<ReshapeOp>();
    }

    bool supportsBackend(Backend backend) const override {
        return backend == Backend::CPU;
    }

    // ------------------------------------------------------------------
    // Shape inference
    //
    // The shape tensor's VALUES are needed to compute the exact output
    // shape, but inferOutputShapes only receives shapes (not data). We
    // validate input types and return a 1-D shape with the correct
    // element count so the operator.h execute() validation passes. The
    // caller pre-allocates the output with the resolved shape; executeCPU
    // re-validates wildcard semantics against the actual shape data.
    // ------------------------------------------------------------------
    std::vector<TensorShape> inferOutputShapes(
        const std::vector<TensorShape>& input_shapes) const override
    {
        validateInputCount("ReshapeOp", 2, input_shapes.size());

        const auto& data_shape  = input_shapes[0];
        const auto& target_shape = input_shapes[1];

        // Shape tensor must be 1-D Int32 (we can only check rank here; the
        // dtype check against Int32 happens in executeCPU where we have
        // the actual tensor).
        if (target_shape.ndim() != 1)
            throw std::invalid_argument(
                "ReshapeOp: shape tensor must be 1-D, got " +
                std::to_string(target_shape.ndim()) + "-D");

        // Return a 1-D shape with matching element count. The caller is
        // expected to pre-allocate the output with the resolved shape;
        // this return value satisfies the numElements validation in
        // operator.h::execute().
        return { TensorShape({data_shape.numElements()}) };
    }

    // ------------------------------------------------------------------
    // CPU execution — resolve wildcards, validate, memcpy
    // ------------------------------------------------------------------
    void executeCPU(
        const std::vector<const Tensor*>& inputs,
        const std::vector<Tensor*>&       outputs) override
    {
        validateInputCount("ReshapeOp", 2, inputs.size());
        validateInputCount("ReshapeOp outputs", 1, outputs.size());

        const Tensor* data_tensor  = inputs[0];
        const Tensor* shape_tensor = inputs[1];
        Tensor*       output       = outputs[0];

        validateDevice("ReshapeOp", data_tensor,  Device::CPU);
        validateDevice("ReshapeOp", shape_tensor, Device::CPU);
        validateDevice("ReshapeOp", output,       Device::CPU);

        validateDtype("ReshapeOp", data_tensor,  DataType::Float32);
        validateDtype("ReshapeOp", shape_tensor, DataType::Int32);
        validateDtype("ReshapeOp", output,       DataType::Float32);

        if (shape_tensor->shape().ndim() != 1)
            throw std::invalid_argument(
                "ReshapeOp: shape tensor must be 1-D, got " +
                std::to_string(shape_tensor->shape().ndim()) + "-D");

        const int32_t* shape_data = static_cast<const int32_t*>(shape_tensor->data());
        const size_t target_rank = shape_tensor->shape().dim(0);
        const size_t input_elems  = data_tensor->numElements();
        const size_t input_ndim   = data_tensor->shape().ndim();

        // Resolve wildcards and validate
        std::vector<size_t> resolved(target_rank, 0);
        int unknown_idx = -1;
        size_t running_product = 1;

        for (size_t i = 0; i < target_rank; ++i) {
            int32_t d = shape_data[i];
            if (d == 0) {
                // Copy input dim i — input must have dim i
                if (i >= input_ndim)
                    throw std::invalid_argument(
                        "ReshapeOp: wildcard 0 at dim " + std::to_string(i) +
                        " but input has only " + std::to_string(input_ndim) +
                        " dimensions");
                resolved[i] = data_tensor->shape().dim(i);
                running_product *= resolved[i];
            } else if (d == -1) {
                // At most one -1
                if (unknown_idx >= 0)
                    throw std::invalid_argument(
                        "ReshapeOp: at most one -1 wildcard allowed (found "
                        "second at dim " + std::to_string(i) + ")");
                unknown_idx = static_cast<int>(i);
            } else if (d > 0) {
                resolved[i] = static_cast<size_t>(d);
                running_product *= resolved[i];
            } else {
                throw std::invalid_argument(
                    "ReshapeOp: shape tensor value " + std::to_string(d) +
                    " at dim " + std::to_string(i) +
                    " is invalid (must be 0, -1, or positive)");
            }
        }

        if (unknown_idx >= 0) {
            // Infer the -1 dimension from the remaining elements
            if (running_product == 0 || input_elems % running_product != 0)
                throw std::invalid_argument(
                    "ReshapeOp: cannot resolve -1 wildcard — input has " +
                    std::to_string(input_elems) +
                    " elements but known dims product is " +
                    std::to_string(running_product));
            resolved[unknown_idx] = input_elems / running_product;
        } else {
            // No wildcard — validate element count
            if (running_product != input_elems)
                throw std::invalid_argument(
                    "ReshapeOp: element count mismatch — shape product is " +
                    std::to_string(running_product) + " but input has " +
                    std::to_string(input_elems) + " elements");
        }

        // Validate output element count matches (the output shape itself
        // is caller-controlled; we verify the data fits)
        if (output->numElements() != input_elems)
            throw std::invalid_argument(
                "ReshapeOp: output has " + std::to_string(output->numElements()) +
                " elements but input has " + std::to_string(input_elems));

        // Byte-identical copy — reshape preserves the row-major data layout
        std::memcpy(output->data(), data_tensor->data(), input_elems * sizeof(float));
    }

private:
    // No params — the target shape is the second input tensor
};

} // namespace forgert
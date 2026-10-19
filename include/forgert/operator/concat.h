#pragma once

#include "forgert/operator/operator.h"
#include <cstring>
#include <stdexcept>
#include <string>

namespace forgert {

/**
 * @brief Concatenate two tensors along a given axis (NCHW float32).
 *
 * SqueezeNet 1.1 uses only:
 *   - 4-D NCHW input tensors
 *   - exactly 2 inputs
 *   - axis = 1  (channel concatenation inside every fire module)
 *
 * The operator is generalised to any valid axis and any number of inputs >= 2
 * so it can serve future models, but SqueezeNet coverage is the tested scope.
 *
 * Shape rules:
 *   All inputs must have the same rank.
 *   All dimensions except the concat axis must be equal.
 *   Output shape equals input shape with dim(axis) summed across inputs.
 *
 * Example (SqueezeNet fire expand):
 *   [1, 64, 55, 55] concat [1, 64, 55, 55] along axis=1 -> [1, 128, 55, 55]
 *
 * CPU implementation: memcpy of each input's axis-slice in row-major order.
 * No CUDA backend: concat is memory-bound; for SqueezeNet sizes the transfer
 * cost makes GPU uncompetitive. A CUDA kernel can be added later if profiling
 * shows it is beneficial.
 */
class ConcatOp : public Operator {
public:
    /// @param axis  Concatenation axis (0-based). Default 1 (channel axis for NCHW).
    explicit ConcatOp(int axis = 1) : axis_(axis) {}

    std::string name() const override { return "Concat"; }

    std::unique_ptr<Operator> clone() const override {
        return std::make_unique<ConcatOp>(axis_);
    }

    int axis() const { return axis_; }

    bool supportsBackend(Backend backend) const override {
        return backend == Backend::CPU;
    }

    // ------------------------------------------------------------------
    // Shape inference
    // ------------------------------------------------------------------
    std::vector<TensorShape> inferOutputShapes(
        const std::vector<TensorShape>& input_shapes) const override
    {
        if (input_shapes.size() < 2)
            throw std::invalid_argument(
                "ConcatOp: requires at least 2 inputs, got " +
                std::to_string(input_shapes.size()));

        const size_t ndim = input_shapes[0].ndim();
        for (size_t i = 1; i < input_shapes.size(); ++i) {
            if (input_shapes[i].ndim() != ndim)
                throw std::invalid_argument(
                    "ConcatOp: all inputs must have the same rank");
        }

        if (axis_ < 0 || static_cast<size_t>(axis_) >= ndim)
            throw std::invalid_argument(
                "ConcatOp: axis " + std::to_string(axis_) +
                " out of range for " + std::to_string(ndim) + "-D inputs");

        // Validate that all non-concat dims are equal
        std::vector<size_t> out_dims(ndim);
        for (size_t d = 0; d < ndim; ++d) {
            out_dims[d] = input_shapes[0].dim(d);
        }
        for (size_t i = 1; i < input_shapes.size(); ++i) {
            for (size_t d = 0; d < ndim; ++d) {
                if (d == static_cast<size_t>(axis_)) {
                    out_dims[d] += input_shapes[i].dim(d);
                } else if (input_shapes[i].dim(d) != out_dims[d]) {
                    throw std::invalid_argument(
                        "ConcatOp: dimension " + std::to_string(d) +
                        " mismatch: input 0 has " + std::to_string(out_dims[d]) +
                        " but input " + std::to_string(i) + " has " +
                        std::to_string(input_shapes[i].dim(d)));
                }
            }
        }
        return { TensorShape(out_dims) };
    }

    // ------------------------------------------------------------------
    // CPU implementation
    // ------------------------------------------------------------------
    void executeCPU(
        const std::vector<const Tensor*>& inputs,
        const std::vector<Tensor*>&       outputs) override
    {
        if (inputs.size() < 2)
            throw std::invalid_argument("ConcatOp: at least 2 inputs required");
        validateInputCount("ConcatOp outputs", 1, outputs.size());

        for (const Tensor* t : inputs) {
            validateDevice("ConcatOp", t, Device::CPU);
            validateDtype("ConcatOp", t, DataType::Float32);
        }
        validateDevice("ConcatOp", outputs[0], Device::CPU);
        validateDtype("ConcatOp", outputs[0], DataType::Float32);

        const size_t ndim  = inputs[0]->shape().ndim();
        const size_t caxis = static_cast<size_t>(axis_);

        // Compute number of outer slices and inner elements
        // outer = product of dims[0..axis-1]
        // inner = product of dims[axis+1..ndim-1]
        size_t outer = 1;
        for (size_t d = 0; d < caxis; ++d)
            outer *= inputs[0]->shape().dim(d);

        size_t inner = 1;
        for (size_t d = caxis + 1; d < ndim; ++d)
            inner *= inputs[0]->shape().dim(d);

        float* out_ptr = static_cast<float*>(outputs[0]->data());

        // For each outer slice, copy each input's slice in order
        for (size_t o = 0; o < outer; ++o) {
            for (const Tensor* inp : inputs) {
                const size_t slice_elems = inp->shape().dim(caxis) * inner;
                const float* src = static_cast<const float*>(inp->data())
                                   + o * slice_elems;
                std::memcpy(out_ptr, src, slice_elems * sizeof(float));
                out_ptr += slice_elems;
            }
        }
    }

private:
    int axis_;
};

} // namespace forgert
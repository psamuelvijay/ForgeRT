#pragma once

#include "forgert/operator/operator.h"

namespace forgert {

/**
 * @brief Element-wise addition operator: C = A + B
 * 
 * Supports broadcasting when shapes are compatible.
 * Phase 1: CPU implementation only, Float32 only
 * Phase 3: CUDA implementation
 * 
 * Broadcasting rules (NumPy-style):
 * - Shapes are compared element-wise from right to left
 * - Two dimensions are compatible if:
 *   1. They are equal, or
 *   2. One of them is 1
 * 
 * Examples:
 *   [3, 4] + [3, 4] = [3, 4]  ✓
 *   [3, 4] + [4]    = [3, 4]  ✓ (broadcast [4] to [1, 4] to [3, 4])
 *   [3, 4] + [1, 4] = [3, 4]  ✓
 *   [3, 4] + [3, 1] = [3, 4]  ✓
 *   [3, 4] + [2, 4] = error   ✗
 */
class AddOp : public Operator {
public:
    std::string name() const override {
        return "Add";
    }

    bool supportsBackend(Backend backend) const override {
        return backend == Backend::CPU;  // Phase 1: CPU only
    }

    std::vector<TensorShape> inferOutputShapes(
        const std::vector<TensorShape>& input_shapes) const override {
        
        validateInputCount("AddOp", 2, input_shapes.size());

        const auto& shape_a = input_shapes[0];
        const auto& shape_b = input_shapes[1];

        // Compute broadcasted output shape
        auto output_shape = broadcastShapes(shape_a, shape_b);
        return {output_shape};
    }

    void executeCPU(
        const std::vector<const Tensor*>& inputs,
        const std::vector<Tensor*>& outputs) override {
        
        validateInputCount("AddOp", 2, inputs.size());
        validateInputCount("AddOp outputs", 1, outputs.size());

        const Tensor* a = inputs[0];
        const Tensor* b = inputs[1];
        Tensor* c = outputs[0];

        // Validate devices
        validateDevice("AddOp", a, Device::CPU);
        validateDevice("AddOp", b, Device::CPU);
        validateDevice("AddOp", c, Device::CPU);

        // Phase 1: Float32 only
        validateDtype("AddOp", a, DataType::Float32);
        validateDtype("AddOp", b, DataType::Float32);
        validateDtype("AddOp", c, DataType::Float32);

        // Get raw pointers
        const float* a_data = static_cast<const float*>(a->data());
        const float* b_data = static_cast<const float*>(b->data());
        float* c_data = static_cast<float*>(c->data());

        // Check if broadcasting is needed
        if (a->shape().numElements() == b->shape().numElements() &&
            a->shape().numElements() == c->shape().numElements()) {
            // Simple case: same shape, element-wise addition
            addElementwise(a_data, b_data, c_data, c->shape().numElements());
        } else {
            // Broadcasting case
            addBroadcast(a, b, c);
        }
    }

private:
    /**
     * @brief Compute broadcasted shape from two input shapes
     */
    static TensorShape broadcastShapes(const TensorShape& shape_a, const TensorShape& shape_b) {
        size_t ndim_a = shape_a.ndim();
        size_t ndim_b = shape_b.ndim();
        size_t ndim_out = std::max(ndim_a, ndim_b);

        std::vector<size_t> out_dims(ndim_out);

        // Process from right to left
        for (size_t i = 0; i < ndim_out; ++i) {
            size_t dim_a = (i < ndim_a) ? shape_a.dim(ndim_a - 1 - i) : 1;
            size_t dim_b = (i < ndim_b) ? shape_b.dim(ndim_b - 1 - i) : 1;

            if (dim_a == dim_b) {
                out_dims[ndim_out - 1 - i] = dim_a;
            } else if (dim_a == 1) {
                out_dims[ndim_out - 1 - i] = dim_b;
            } else if (dim_b == 1) {
                out_dims[ndim_out - 1 - i] = dim_a;
            } else {
                throw std::invalid_argument(
                    "AddOp: incompatible shapes for broadcasting");
            }
        }

        return TensorShape(out_dims);
    }

    /**
     * @brief Simple element-wise addition (no broadcasting)
     */
    static void addElementwise(
        const float* a,
        const float* b,
        float* c,
        size_t n) {
        for (size_t i = 0; i < n; ++i) {
            c[i] = a[i] + b[i];
        }
    }

    /**
     * @brief Addition with broadcasting
     * 
     * This is a naive implementation for correctness.
     * Phase 2: Optimize with better loop structures
     */
    static void addBroadcast(const Tensor* a, const Tensor* b, Tensor* c) {
        const TensorShape& shape_a = a->shape();
        const TensorShape& shape_b = b->shape();
        const TensorShape& shape_c = c->shape();

        const float* a_data = static_cast<const float*>(a->data());
        const float* b_data = static_cast<const float*>(b->data());
        float* c_data = static_cast<float*>(c->data());

        size_t n = c->shape().numElements();

        for (size_t i = 0; i < n; ++i) {
            // Compute multi-dimensional index for output
            std::vector<size_t> c_idx = linearToMultiDim(i, shape_c);

            // Map to input indices with broadcasting
            std::vector<size_t> a_idx = broadcastIndex(c_idx, shape_c, shape_a);
            std::vector<size_t> b_idx = broadcastIndex(c_idx, shape_c, shape_b);

            // Compute flat indices
            size_t a_flat = shape_a.flatIndex(a_idx);
            size_t b_flat = shape_b.flatIndex(b_idx);

            c_data[i] = a_data[a_flat] + b_data[b_flat];
        }
    }

    /**
     * @brief Convert linear index to multi-dimensional index
     */
    static std::vector<size_t> linearToMultiDim(size_t linear, const TensorShape& shape) {
        std::vector<size_t> indices(shape.ndim());
        size_t remaining = linear;

        for (int i = static_cast<int>(shape.ndim()) - 1; i >= 0; --i) {
            indices[i] = remaining % shape.dim(i);
            remaining /= shape.dim(i);
        }

        return indices;
    }

    /**
     * @brief Map output index to input index with broadcasting
     */
    static std::vector<size_t> broadcastIndex(
        const std::vector<size_t>& out_idx,
        const TensorShape& out_shape,
        const TensorShape& in_shape) {
        
        size_t ndim_out = out_shape.ndim();
        size_t ndim_in = in_shape.ndim();

        std::vector<size_t> in_idx(ndim_in);

        // Map from right to left
        for (size_t i = 0; i < ndim_in; ++i) {
            size_t out_i = ndim_out - ndim_in + i;
            size_t dim_in = in_shape.dim(i);

            if (dim_in == 1) {
                in_idx[i] = 0;  // Broadcast this dimension
            } else {
                in_idx[i] = out_idx[out_i];
            }
        }

        return in_idx;
    }
};

} // namespace forgert

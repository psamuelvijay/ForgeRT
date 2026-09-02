#pragma once

#include "forgert/operator/operator.h"
#include <cmath>
#include <algorithm>

namespace forgert {

/**
 * @brief Layer Normalization operator
 * 
 * Normalizes along the last dimension (feature dimension):
 *   output = (input - mean) / sqrt(variance + epsilon)
 * 
 * Where mean and variance are computed per sample over the feature dimension.
 * 
 * Common use case: [Batch, Features] -> [Batch, Features]
 *   Each sample is independently normalized to mean=0, variance=1
 * 
 * Phase 2: CPU implementation only, Float32 only
 * Phase 3: CUDA implementation
 * Phase 4+: Learnable affine parameters (scale and bias)
 * 
 * Shape: Input and output have identical shapes
 * 
 * Examples:
 *   [10, 512] -> [10, 512]  ✓ (10 samples, 512 features each)
 *   [3, 768] -> [3, 768]    ✓ (3 samples, 768 features each)
 *   [1, 128] -> [1, 128]    ✓ (single sample, 128 features)
 *   [512] -> [512]          ✓ (single sample, 512 features)
 */
class LayerNormOp : public Operator {
public:
    /**
     * @brief Construct LayerNorm operator
     * @param epsilon Small constant for numerical stability (default: 1e-5)
     */
    explicit LayerNormOp(float epsilon = 1e-5f) : epsilon_(epsilon) {
        if (epsilon <= 0.0f) {
            throw std::invalid_argument("LayerNormOp: epsilon must be positive");
        }
    }

    std::string name() const override {
        return "LayerNorm";
    }

    bool supportsBackend(Backend backend) const override {
        return backend == Backend::CPU;  // Phase 2: CPU only
    }

    std::vector<TensorShape> inferOutputShapes(
        const std::vector<TensorShape>& input_shapes) const override {
        
        validateInputCount("LayerNormOp", 1, input_shapes.size());
        
        // Output shape is identical to input shape
        return {input_shapes[0]};
    }

    void executeCPU(
        const std::vector<const Tensor*>& inputs,
        const std::vector<Tensor*>& outputs) override {
        
        validateInputCount("LayerNormOp", 1, inputs.size());
        validateInputCount("LayerNormOp outputs", 1, outputs.size());

        const Tensor* input = inputs[0];
        Tensor* output = outputs[0];

        // Validate devices
        validateDevice("LayerNormOp", input, Device::CPU);
        validateDevice("LayerNormOp", output, Device::CPU);

        // Phase 2: Float32 only
        validateDtype("LayerNormOp", input, DataType::Float32);
        validateDtype("LayerNormOp", output, DataType::Float32);

        const float* in_data = static_cast<const float*>(input->data());
        float* out_data = static_cast<float*>(output->data());

        const auto& shape = input->shape();
        size_t ndim = shape.ndim();

        if (ndim == 1) {
            // 1D case: normalize all elements
            applyLayerNorm1D(in_data, out_data, shape.dim(0), epsilon_);
        } else {
            // Multi-dimensional case: normalize along the last dimension
            // Treat tensor as [outer_size, inner_size] where inner_size is the last dimension
            size_t inner_size = shape.dim(ndim - 1);
            size_t outer_size = shape.numElements() / inner_size;

            for (size_t i = 0; i < outer_size; ++i) {
                const float* row_in = in_data + i * inner_size;
                float* row_out = out_data + i * inner_size;
                applyLayerNorm1D(row_in, row_out, inner_size, epsilon_);
            }
        }
    }

private:
    float epsilon_;

    /**
     * @brief Apply layer normalization to a 1D array
     * 
     * Computes: output = (input - mean) / sqrt(variance + epsilon)
     * 
     * Uses numerically stable two-pass algorithm:
     * Pass 1: Compute mean
     * Pass 2: Compute variance and normalize
     * 
     * @param input Input array
     * @param output Output array (can be same as input)
     * @param size Number of elements
     * @param epsilon Small constant for numerical stability
     */
    static void applyLayerNorm1D(const float* input, float* output, size_t size, float epsilon) {
        if (size == 0) {
            return;
        }

        // Pass 1: Compute mean
        float sum = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            sum += input[i];
        }
        float mean = sum / static_cast<float>(size);

        // Pass 2: Compute variance
        float variance_sum = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            float diff = input[i] - mean;
            variance_sum += diff * diff;
        }
        float variance = variance_sum / static_cast<float>(size);

        // Compute normalization factor: 1 / sqrt(variance + epsilon)
        float inv_std = 1.0f / std::sqrt(variance + epsilon);

        // Pass 3: Normalize
        for (size_t i = 0; i < size; ++i) {
            output[i] = (input[i] - mean) * inv_std;
        }
    }
};

} // namespace forgert

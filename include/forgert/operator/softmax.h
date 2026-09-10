#pragma once

#include "forgert/operator/operator.h"
#include <cmath>
#include <algorithm>
#include <limits>

// When FORGERT_CUDA_AVAILABLE is defined, bring in the extern "C" kernel launcher
#ifdef FORGERT_CUDA_AVAILABLE
extern "C" int forgert_softmax_cuda(const float* input, float* output,
                                    size_t outer_size, size_t inner_size);
#endif

namespace forgert {

/**
 * @brief Softmax activation operator
 *
 * Applies softmax along the last dimension:
 *   softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
 *
 * Uses numerically stable implementation with max subtraction to prevent overflow.
 *
 * Common use case: [batch, classes] -> [batch, classes]
 *   Each row is independently normalized to a probability distribution.
 *
 * Phase 2: CPU implementation only, Float32 only
 * Phase 3: CUDA implementation
 *
 * Shape: Input and output have identical shapes
 *
 * Examples:
 *   [3, 4] -> [3, 4]  ✓ (softmax applied to each of 3 rows independently)
 *   [10] -> [10]      ✓ (single softmax over 10 elements)
 *   [2, 3, 4] -> [2, 3, 4]  ✓ (softmax applied along last dimension)
 */
class SoftmaxOp : public Operator {
public:
    std::string name() const override {
        return "Softmax";
    }

    bool supportsBackend(Backend backend) const override {
#ifdef FORGERT_CUDA_AVAILABLE
        return backend == Backend::CPU || backend == Backend::CUDA;
#else
        return backend == Backend::CPU;
#endif
    }

    std::vector<TensorShape> inferOutputShapes(
        const std::vector<TensorShape>& input_shapes) const override {

        validateInputCount("SoftmaxOp", 1, input_shapes.size());

        // Output shape is identical to input shape
        return {input_shapes[0]};
    }

    void executeCPU(
        const std::vector<const Tensor*>& inputs,
        const std::vector<Tensor*>& outputs) override {

        validateInputCount("SoftmaxOp", 1, inputs.size());
        validateInputCount("SoftmaxOp outputs", 1, outputs.size());

        const Tensor* input = inputs[0];
        Tensor* output = outputs[0];

        // Validate devices
        validateDevice("SoftmaxOp", input, Device::CPU);
        validateDevice("SoftmaxOp", output, Device::CPU);

        // Phase 2: Float32 only
        validateDtype("SoftmaxOp", input, DataType::Float32);
        validateDtype("SoftmaxOp", output, DataType::Float32);

        const float* in_data = static_cast<const float*>(input->data());
        float* out_data = static_cast<float*>(output->data());

        const auto& shape = input->shape();
        size_t ndim = shape.ndim();

        if (ndim == 1) {
            // 1D case: single softmax over all elements
            applySoftmax1D(in_data, out_data, shape.dim(0));
        } else {
            // Multi-dimensional case: apply softmax along the last dimension
            // Treat tensor as [outer_size, inner_size] where inner_size is the last dimension
            size_t inner_size = shape.dim(ndim - 1);
            size_t outer_size = shape.numElements() / inner_size;

            for (size_t i = 0; i < outer_size; ++i) {
                const float* row_in = in_data + i * inner_size;
                float* row_out = out_data + i * inner_size;
                applySoftmax1D(row_in, row_out, inner_size);
            }
        }
    }

#ifdef FORGERT_CUDA_AVAILABLE
    void executeCUDA(
        const std::vector<const Tensor*>& inputs,
        const std::vector<Tensor*>& outputs) override {

        validateInputCount("SoftmaxOp", 1, inputs.size());
        validateInputCount("SoftmaxOp outputs", 1, outputs.size());

        const Tensor* input = inputs[0];
        Tensor* output = outputs[0];

        // Validate devices
        validateDevice("SoftmaxOp", input, Device::CUDA);
        validateDevice("SoftmaxOp", output, Device::CUDA);

        // Validate data types
        validateDtype("SoftmaxOp", input, DataType::Float32);
        validateDtype("SoftmaxOp", output, DataType::Float32);

        const float* d_input = static_cast<const float*>(input->data());
        float* d_output = static_cast<float*>(output->data());

        const auto& shape = input->shape();
        size_t ndim = shape.ndim();

        // Calculate outer_size and inner_size for CUDA kernel
        size_t inner_size, outer_size;
        if (ndim == 1) {
            inner_size = shape.dim(0);
            outer_size = 1;
        } else {
            inner_size = shape.dim(ndim - 1);
            outer_size = shape.numElements() / inner_size;
        }

        // Launch CUDA kernel
        const int err = forgert_softmax_cuda(d_input, d_output, outer_size, inner_size);
        if (err != 0) {
            throw std::runtime_error(
                "SoftmaxOp::executeCUDA: kernel launcher failed (cudaError_t=" +
                std::to_string(err) + ")");
        }
    }
#endif // FORGERT_CUDA_AVAILABLE

private:
    /**
     * @brief Apply numerically stable softmax to a 1D array
     *
     * Uses max subtraction to prevent overflow:
     *   softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
     *
     * @param input Input array
     * @param output Output array (can be same as input)
     * @param size Number of elements
     */
    static void applySoftmax1D(const float* input, float* output, size_t size) {
        if (size == 0) {
            return;
        }

        // Find maximum value for numerical stability
        float max_val = input[0];
        for (size_t i = 1; i < size; ++i) {
            max_val = std::max(max_val, input[i]);
        }

        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            output[i] = std::exp(input[i] - max_val);
            sum += output[i];
        }

        // Normalize by sum
        // Guard against sum = 0 (though this shouldn't happen with exp)
        if (sum > 0.0f) {
            for (size_t i = 0; i < size; ++i) {
                output[i] /= sum;
            }
        } else {
            // Fallback: uniform distribution (should be extremely rare)
            float uniform = 1.0f / static_cast<float>(size);
            for (size_t i = 0; i < size; ++i) {
                output[i] = uniform;
            }
        }
    }
};

} // namespace forgert

#pragma once

#include "forgert/operator/operator.h"

namespace forgert {

/**
 * @brief Rectified Linear Unit (ReLU) activation operator
 * 
 * Applies element-wise: output = max(0, input)
 * 
 * Phase 2: CPU implementation only, Float32 only
 * Phase 3: CUDA implementation
 * 
 * Shape: Input and output have identical shapes (element-wise operation)
 */
class ReLUOp : public Operator {
public:
    std::string name() const override {
        return "ReLU";
    }

    bool supportsBackend(Backend backend) const override {
        return backend == Backend::CPU;  // Phase 2: CPU only
    }

    std::vector<TensorShape> inferOutputShapes(
        const std::vector<TensorShape>& input_shapes) const override {
        
        validateInputCount("ReLUOp", 1, input_shapes.size());
        
        // Output shape is identical to input shape
        return {input_shapes[0]};
    }

    void executeCPU(
        const std::vector<const Tensor*>& inputs,
        const std::vector<Tensor*>& outputs) override {
        
        validateInputCount("ReLUOp", 1, inputs.size());
        validateInputCount("ReLUOp outputs", 1, outputs.size());

        const Tensor* input = inputs[0];
        Tensor* output = outputs[0];

        // Validate devices
        validateDevice("ReLUOp", input, Device::CPU);
        validateDevice("ReLUOp", output, Device::CPU);

        // Phase 2: Float32 only
        validateDtype("ReLUOp", input, DataType::Float32);
        validateDtype("ReLUOp", output, DataType::Float32);

        // Perform ReLU: max(0, x)
        const float* in_data = static_cast<const float*>(input->data());
        float* out_data = static_cast<float*>(output->data());
        size_t n = input->numElements();

        for (size_t i = 0; i < n; ++i) {
            out_data[i] = (in_data[i] > 0.0f) ? in_data[i] : 0.0f;
        }
    }
};

} // namespace forgert

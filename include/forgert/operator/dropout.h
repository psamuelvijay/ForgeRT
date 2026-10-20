#pragma once

#include "forgert/operator/operator.h"
#include <cstring>
#include <stdexcept>
#include <string>

namespace forgert {

/**
 * @brief Dropout operator — inference identity (memcpy).
 *
 * At inference, Dropout passes its input through unchanged regardless of the
 * `ratio` attribute. No mask is generated; no scaling by `1/(1-ratio)` is
 * applied. The output is byte-identical to the input.
 *
 * ONNX attribute handling:
 *   - `ratio` (float, opset 7+): stored in params for completeness, NEVER used
 *     at inference.
 *   - `training_mode` (int, opset 12+): not present in SqueezeNet opset 7.
 *     If a future model sets `training_mode=1`, the importer must error —
 *     ForgeRT does not execute training mode.
 *
 * SqueezeNet 1.1 has one Dropout node with `ratio=0.5` (opset 7, no
 * training_mode). At inference it is identity.
 *
 * CPU implementation: memcpy. No CUDA backend — a CUDA kernel would add 20 us
 * launch overhead for zero compute benefit. A CUDA backend is not precluded
 * for future phases but is out of scope for Phase 8.
 *
 * Inputs:  [0] input tensor (any shape, Float32)
 * Outputs: [0] output tensor (same shape as input, Float32)
 */
struct DropoutParams {
    float ratio = 0.5f;  // stored for completeness; ignored at inference
};

class DropoutOp : public Operator {
public:
    explicit DropoutOp(DropoutParams p = {}) : p_(p) {}

    std::string name() const override { return "Dropout"; }

    std::unique_ptr<Operator> clone() const override {
        return std::make_unique<DropoutOp>(p_);
    }

    bool supportsBackend(Backend backend) const override {
        return backend == Backend::CPU;
    }

    // ------------------------------------------------------------------
    // Shape inference — identity
    // ------------------------------------------------------------------
    std::vector<TensorShape> inferOutputShapes(
        const std::vector<TensorShape>& input_shapes) const override
    {
        validateInputCount("DropoutOp", 1, input_shapes.size());
        return {input_shapes[0]};
    }

    // ------------------------------------------------------------------
    // CPU execution — byte-identical memcpy
    // ------------------------------------------------------------------
    void executeCPU(
        const std::vector<const Tensor*>& inputs,
        const std::vector<Tensor*>&       outputs) override
    {
        validateInputCount("DropoutOp", 1, inputs.size());
        validateInputCount("DropoutOp outputs", 1, outputs.size());

        const Tensor* input  = inputs[0];
        Tensor*       output = outputs[0];

        validateDevice("DropoutOp", input,  Device::CPU);
        validateDevice("DropoutOp", output, Device::CPU);
        validateDtype("DropoutOp",  input,  DataType::Float32);
        validateDtype("DropoutOp",  output, DataType::Float32);

        // Byte-identical copy. No mask, no scaling — inference identity.
        std::memcpy(output->data(), input->data(), input->numBytes());
    }

private:
    DropoutParams p_;
};

} // namespace forgert
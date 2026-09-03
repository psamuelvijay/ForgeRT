#pragma once

#include "forgert/operator/operator.h"

// When FORGERT_CUDA_AVAILABLE is defined (set by the forgert_cuda CMake target),
// bring in the extern "C" kernel launcher declaration.  This guard ensures that
// plain CXX translation units which do not link forgert_cuda never see the
// symbol — and therefore never generate a linker dependency on it.
#ifdef FORGERT_CUDA_AVAILABLE
extern "C" int forgert_relu_cuda(const float* input, float* output, size_t n);
#endif

namespace forgert {

/**
 * @brief Rectified Linear Unit (ReLU) activation operator
 *
 * Applies element-wise: output = max(0, input)
 *
 * Phase 2: CPU implementation (Float32)
 * Phase 3: CUDA implementation via forgert_relu_cuda kernel launcher
 *          (available when built with FORGERT_CUDA_AVAILABLE defined,
 *           i.e. when linking against the forgert_cuda library)
 *
 * Shape: input and output shapes are identical (element-wise operation).
 *
 * CUDA memory note (Phase 3 / pre-Phase-4):
 *   Tensor does not yet own CUDA device memory.  Callers must supply
 *   Tensor objects whose data() pointers reference valid device memory
 *   (e.g. allocated via cudaMalloc).  Phase 4 will add Tensor CUDA
 *   allocation so this becomes transparent.
 */
class ReLUOp : public Operator {
public:
    std::string name() const override {
        return "ReLU";
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

        validateInputCount("ReLUOp", 1, input_shapes.size());
        return {input_shapes[0]};
    }

    // -----------------------------------------------------------------
    // CPU path
    // -----------------------------------------------------------------
    void executeCPU(
        const std::vector<const Tensor*>& inputs,
        const std::vector<Tensor*>&       outputs) override {

        validateInputCount("ReLUOp",         1, inputs.size());
        validateInputCount("ReLUOp outputs", 1, outputs.size());

        const Tensor* input  = inputs[0];
        Tensor*       output = outputs[0];

        validateDevice("ReLUOp", input,  Device::CPU);
        validateDevice("ReLUOp", output, Device::CPU);
        validateDtype ("ReLUOp", input,  DataType::Float32);
        validateDtype ("ReLUOp", output, DataType::Float32);

        const float* in_data  = static_cast<const float*>(input->data());
        float*       out_data = static_cast<float*>(output->data());
        const size_t n        = input->numElements();

        for (size_t i = 0; i < n; ++i) {
            out_data[i] = (in_data[i] > 0.0f) ? in_data[i] : 0.0f;
        }
    }

#ifdef FORGERT_CUDA_AVAILABLE
    // -----------------------------------------------------------------
    // CUDA path (Phase 3)
    //
    // Only compiled when FORGERT_CUDA_AVAILABLE is defined, which happens
    // when a target links forgert_cuda (the CMake target that defines the
    // compile definition and provides the kernel symbol).
    //
    // Preconditions (validated below):
    //   - Both tensors must report Device::CUDA
    //   - Both must be Float32
    //   - data() must point to valid device memory (caller-managed until
    //     Phase 4 adds Tensor CUDA allocation)
    //
    // On kernel failure the launcher returns a non-zero cudaError_t cast
    // to int; we surface that as a runtime_error so errors are never
    // silently swallowed.
    // -----------------------------------------------------------------
    void executeCUDA(
        const std::vector<const Tensor*>& inputs,
        const std::vector<Tensor*>&       outputs) override {

        validateInputCount("ReLUOp",         1, inputs.size());
        validateInputCount("ReLUOp outputs", 1, outputs.size());

        const Tensor* input  = inputs[0];
        Tensor*       output = outputs[0];

        validateDevice("ReLUOp", input,  Device::CUDA);
        validateDevice("ReLUOp", output, Device::CUDA);
        validateDtype ("ReLUOp", input,  DataType::Float32);
        validateDtype ("ReLUOp", output, DataType::Float32);

        const float* d_in  = static_cast<const float*>(input->data());
        float*       d_out = static_cast<float*>(output->data());
        const size_t n     = input->numElements();

        const int err = forgert_relu_cuda(d_in, d_out, n);
        if (err != 0) {
            throw std::runtime_error(
                "ReLUOp::executeCUDA: kernel launcher failed (cudaError_t=" +
                std::to_string(err) + ")");
        }
    }
#endif // FORGERT_CUDA_AVAILABLE
};

} // namespace forgert

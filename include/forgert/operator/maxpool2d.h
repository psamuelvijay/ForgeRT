#pragma once

#include "forgert/operator/operator.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace forgert {

/**
 * @brief 2-D max-pooling operator (NCHW layout, float32).
 *
 * SqueezeNet 1.1 uses:
 *   - 3x3 kernel, stride 2, no padding — three times
 *
 * Output size formula (ONNX, no ceil_mode):
 *   H_out = floor((H_in + 2*pad_h - kH) / stride_h) + 1
 *   W_out = floor((W_in + 2*pad_w - kW) / stride_w) + 1
 *
 * Inputs:  [0] input tensor [N, C, H_in, W_in]
 * Outputs: [0] output tensor [N, C, H_out, W_out]
 * No weight tensor.
 *
 * CPU implementation: direct sliding window — correct and simple.
 * No CUDA backend: for SqueezeNet's spatial sizes (55x55 down to 3x3) GPU
 * launch overhead exceeds any compute saving. Add a CUDA kernel if a future
 * model has large spatial maps where pooling becomes compute-bound.
 */
struct MaxPool2DParams {
    int kH = 3, kW = 3;
    int stride_h = 2, stride_w = 2;
    int pad_h    = 0, pad_w    = 0;
};

class MaxPool2DOp : public Operator {
public:
    explicit MaxPool2DOp(MaxPool2DParams p = {}) : p_(p) {}

    std::string name() const override { return "MaxPool2D"; }

    std::unique_ptr<Operator> clone() const override {
        return std::make_unique<MaxPool2DOp>(p_);
    }

    const MaxPool2DParams& params() const { return p_; }

    bool supportsBackend(Backend backend) const override {
        return backend == Backend::CPU;
    }

    // ------------------------------------------------------------------
    // Shape inference
    // ------------------------------------------------------------------
    std::vector<TensorShape> inferOutputShapes(
        const std::vector<TensorShape>& input_shapes) const override
    {
        validateInputCount("MaxPool2DOp", 1, input_shapes.size());
        const auto& s = input_shapes[0];
        if (s.ndim() != 4)
            throw std::invalid_argument(
                "MaxPool2DOp: input must be 4-D (NCHW), got " +
                std::to_string(s.ndim()) + "-D");

        const size_t N    = s.dim(0);
        const size_t C    = s.dim(1);
        const size_t H_in = s.dim(2);
        const size_t W_in = s.dim(3);

        // ONNX floor-mode (default, no ceil_mode attribute used in SqueezeNet)
        const size_t H_out = (H_in + 2*static_cast<size_t>(p_.pad_h)
                               - static_cast<size_t>(p_.kH))
                              / static_cast<size_t>(p_.stride_h) + 1;
        const size_t W_out = (W_in + 2*static_cast<size_t>(p_.pad_w)
                               - static_cast<size_t>(p_.kW))
                              / static_cast<size_t>(p_.stride_w) + 1;

        return { TensorShape({N, C, H_out, W_out}) };
    }

    // ------------------------------------------------------------------
    // CPU implementation — direct sliding-window max reduction
    // ------------------------------------------------------------------
    void executeCPU(
        const std::vector<const Tensor*>& inputs,
        const std::vector<Tensor*>&       outputs) override
    {
        validateInputCount("MaxPool2DOp", 1, inputs.size());
        validateInputCount("MaxPool2DOp outputs", 1, outputs.size());

        const Tensor* input  = inputs[0];
        Tensor*       output = outputs[0];

        validateDevice("MaxPool2DOp", input,  Device::CPU);
        validateDevice("MaxPool2DOp", output, Device::CPU);
        validateDtype("MaxPool2DOp",  input,  DataType::Float32);
        validateDtype("MaxPool2DOp",  output, DataType::Float32);

        const int N     = static_cast<int>(input->shape().dim(0));
        const int C     = static_cast<int>(input->shape().dim(1));
        const int H_in  = static_cast<int>(input->shape().dim(2));
        const int W_in  = static_cast<int>(input->shape().dim(3));
        const int H_out = static_cast<int>(output->shape().dim(2));
        const int W_out = static_cast<int>(output->shape().dim(3));
        const int kH    = p_.kH, kW = p_.kW;
        const int sH    = p_.stride_h, sW = p_.stride_w;
        const int pH    = p_.pad_h,    pW = p_.pad_w;

        const float* in_d  = static_cast<const float*>(input->data());
        float*       out_d = static_cast<float*>(output->data());

        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int oh = 0; oh < H_out; ++oh) {
                    for (int ow = 0; ow < W_out; ++ow) {
                        float max_val = -std::numeric_limits<float>::infinity();
                        for (int kh = 0; kh < kH; ++kh) {
                            const int ih = oh * sH - pH + kh;
                            if (ih < 0 || ih >= H_in) continue;
                            for (int kw2 = 0; kw2 < kW; ++kw2) {
                                const int iw = ow * sW - pW + kw2;
                                if (iw < 0 || iw >= W_in) continue;
                                const float v =
                                    in_d[((n * C + c) * H_in + ih) * W_in + iw];
                                if (v > max_val) max_val = v;
                            }
                        }
                        out_d[((n * C + c) * H_out + oh) * W_out + ow] = max_val;
                    }
                }
            }
        }
    }

private:
    MaxPool2DParams p_;
};

} // namespace forgert
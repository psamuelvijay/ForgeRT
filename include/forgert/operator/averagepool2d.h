#pragma once

#include "forgert/operator/operator.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace forgert {

/**
 * @brief 2-D average pooling operator (NCHW layout, float32).
 *
 * SqueezeNet 1.1 uses:
 *   - kernel [13,13], stride [13,13], pads [0,0,0,0] — the single AveragePool
 *     node is effectively global average pooling on a 13x13 input, producing
 *     [1, C, 1, 1].
 *
 * Output size formula (ONNX floor-mode, no ceil_mode):
 *   H_out = floor((H_in + 2*pad_h - kH) / stride_h) + 1
 *   W_out = floor((W_in + 2*pad_w - kW) / stride_w) + 1
 *
 * count_include_pad:
 *   false (ONNX default): divide window sum by the number of in-bounds elements
 *   true:                  divide window sum by kH * kW (constant)
 *   SqueezeNet uses pads=0 so both modes give identical results.
 *
 * CPU implementation: direct sliding-window mean reduction.
 * No CUDA backend: for SqueezeNet's 13x13 spatial reduction the 20 us launch
 * overhead exceeds any compute benefit. A CUDA backend is not precluded for
 * future phases but is out of scope for Phase 8.
 *
 * Inputs:  [0] input tensor [N, C, H_in, W_in]
 * Outputs: [0] output tensor [N, C, H_out, W_out]
 */
struct AvgPool2DParams {
    int kH = 1, kW = 1;
    int stride_h = 1, stride_w = 1;
    int pad_h    = 0, pad_w    = 0;
    bool count_include_pad = false;  // ONNX default is 0 (false)
};

class AveragePool2DOp : public Operator {
public:
    explicit AveragePool2DOp(AvgPool2DParams p = {}) : p_(p) {
        // Validate params at construction — fail fast on impossible configs.
        if (p_.kH <= 0 || p_.kW <= 0)
            throw std::invalid_argument(
                "AveragePool2DOp: kernel size must be positive (got kH=" +
                std::to_string(p_.kH) + ", kW=" + std::to_string(p_.kW) + ")");
        if (p_.stride_h <= 0 || p_.stride_w <= 0)
            throw std::invalid_argument(
                "AveragePool2DOp: stride must be positive (got stride_h=" +
                std::to_string(p_.stride_h) + ", stride_w=" + std::to_string(p_.stride_w) + ")");
        if (p_.pad_h < 0 || p_.pad_w < 0)
            throw std::invalid_argument(
                "AveragePool2DOp: padding must be non-negative (got pad_h=" +
                std::to_string(p_.pad_h) + ", pad_w=" + std::to_string(p_.pad_w) + ")");
    }

    std::string name() const override { return "AveragePool2D"; }

    std::unique_ptr<Operator> clone() const override {
        return std::make_unique<AveragePool2DOp>(p_);
    }

    const AvgPool2DParams& params() const { return p_; }

    bool supportsBackend(Backend backend) const override {
        return backend == Backend::CPU;
    }

    // ------------------------------------------------------------------
    // Shape inference
    // ------------------------------------------------------------------
    std::vector<TensorShape> inferOutputShapes(
        const std::vector<TensorShape>& input_shapes) const override
    {
        validateInputCount("AveragePool2DOp", 1, input_shapes.size());
        const auto& s = input_shapes[0];
        if (s.ndim() != 4)
            throw std::invalid_argument(
                "AveragePool2DOp: input must be 4-D (NCHW), got " +
                std::to_string(s.ndim()) + "-D");

        const size_t N    = s.dim(0);
        const size_t C    = s.dim(1);
        const size_t H_in = s.dim(2);
        const size_t W_in = s.dim(3);

        // ONNX floor-mode (integer division truncates toward zero for positive
        // values, which is floor for our always-non-negative operands).
        const size_t H_out = (H_in + 2*static_cast<size_t>(p_.pad_h)
                               - static_cast<size_t>(p_.kH))
                              / static_cast<size_t>(p_.stride_h) + 1;
        const size_t W_out = (W_in + 2*static_cast<size_t>(p_.pad_w)
                               - static_cast<size_t>(p_.kW))
                              / static_cast<size_t>(p_.stride_w) + 1;

        return { TensorShape({N, C, H_out, W_out}) };
    }

    // ------------------------------------------------------------------
    // CPU implementation — direct sliding-window mean
    // ------------------------------------------------------------------
    void executeCPU(
        const std::vector<const Tensor*>& inputs,
        const std::vector<Tensor*>&       outputs) override
    {
        validateInputCount("AveragePool2DOp", 1, inputs.size());
        validateInputCount("AveragePool2DOp outputs", 1, outputs.size());

        const Tensor* input  = inputs[0];
        Tensor*       output = outputs[0];

        validateDevice("AveragePool2DOp", input,  Device::CPU);
        validateDevice("AveragePool2DOp", output, Device::CPU);
        validateDtype("AveragePool2DOp",  input,  DataType::Float32);
        validateDtype("AveragePool2DOp",  output, DataType::Float32);

        const int N     = static_cast<int>(input->shape().dim(0));
        const int C     = static_cast<int>(input->shape().dim(1));
        const int H_in  = static_cast<int>(input->shape().dim(2));
        const int W_in  = static_cast<int>(input->shape().dim(3));
        const int H_out = static_cast<int>(output->shape().dim(2));
        const int W_out = static_cast<int>(output->shape().dim(3));
        const int kH    = p_.kH, kW = p_.kW;
        const int sH    = p_.stride_h, sW = p_.stride_w;
        const int pH    = p_.pad_h,    pW = p_.pad_w;
        const float denom_const = static_cast<float>(kH * kW);  // for count_include_pad=true

        const float* in_d  = static_cast<const float*>(input->data());
        float*       out_d = static_cast<float*>(output->data());

        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int oh = 0; oh < H_out; ++oh) {
                    for (int ow = 0; ow < W_out; ++ow) {
                        float sum = 0.0f;
                        int count = 0;
                        for (int kh = 0; kh < kH; ++kh) {
                            const int ih = oh * sH - pH + kh;
                            if (ih < 0 || ih >= H_in) continue;
                            for (int kw2 = 0; kw2 < kW; ++kw2) {
                                const int iw = ow * sW - pW + kw2;
                                if (iw < 0 || iw >= W_in) continue;
                                sum += in_d[((n * C + c) * H_in + ih) * W_in + iw];
                                ++count;
                            }
                        }
                        const float denom = p_.count_include_pad
                                              ? denom_const
                                              : static_cast<float>(count);
                        out_d[((n * C + c) * H_out + oh) * W_out + ow] =
                            (count > 0 || p_.count_include_pad) ? sum / denom : 0.0f;
                    }
                }
            }
        }
    }

private:
    AvgPool2DParams p_;
};

} // namespace forgert
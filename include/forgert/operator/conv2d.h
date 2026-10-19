#pragma once

#include "forgert/operator/operator.h"
#include <array>
#include <stdexcept>
#include <string>

#ifdef FORGERT_CUDA_AVAILABLE
extern "C" int forgert_conv2d_cuda(
    const float* input,   // [N, C_in,  H_in,  W_in]  -- NCHW, device ptr
    const float* weight,  // [C_out, C_in, kH, kW]    -- device ptr
    const float* bias,    // [C_out] or nullptr         -- device ptr or null
    float*       output,  // [N, C_out, H_out, W_out]  -- device ptr
    int N, int C_in, int H_in, int W_in,
    int C_out, int kH, int kW,
    int stride_h, int stride_w,
    int pad_h,    int pad_w);
#endif

namespace forgert {

/**
 * @brief 2-D convolution operator (NCHW layout, float32, group=1, dilation=1).
 *
 * Covers the exact subset used by SqueezeNet 1.1:
 *   - kernel sizes 1×1 and 3×3
 *   - strides 1×1 and 2×2
 *   - symmetric padding 0 or 1
 *   - optional bias
 *   - no depthwise (group == 1 enforced)
 *   - no dilation (dilation == 1 enforced)
 *
 * Tensor layout: NCHW (row-major, C is the contiguous channel axis).
 *   input  [N, C_in,  H_in,  W_in]
 *   weight [C_out, C_in, kH, kW]
 *   bias   [C_out]  (optional; pass nullptr or omit third input)
 *   output [N, C_out, H_out, W_out]
 *
 * Output size (standard formula):
 *   H_out = (H_in + 2*pad_h - kH) / stride_h + 1
 *   W_out = (W_in + 2*pad_w - kW) / stride_w + 1
 *
 * CPU implementation: direct (naive) convolution.
 * CUDA implementation: im2col + tiled GEMM kernel.
 *
 * Inputs to execute():
 *   [0] input  tensor
 *   [1] weight tensor
 *   [2] bias   tensor  (optional — pass only 2 inputs to skip bias)
 */
struct Conv2DParams {
    int kH = 1, kW = 1;
    int stride_h = 1, stride_w = 1;
    int pad_h    = 0, pad_w    = 0;
    // dilation and group are fixed at 1 — this operator does not support
    // depthwise or dilated convolutions.
};

class Conv2DOp : public Operator {
public:
    explicit Conv2DOp(Conv2DParams p = {}) : p_(p) {}

    std::string name() const override { return "Conv2D"; }

    std::unique_ptr<Operator> clone() const override {
        return std::make_unique<Conv2DOp>(p_);
    }

    bool supportsBackend(Backend backend) const override {
#ifdef FORGERT_CUDA_AVAILABLE
        return backend == Backend::CPU || backend == Backend::CUDA;
#else
        return backend == Backend::CPU;
#endif
    }

    const Conv2DParams& params() const { return p_; }

    // ------------------------------------------------------------------
    // Shape inference
    // ------------------------------------------------------------------
    // Inputs: [input, weight] or [input, weight, bias]
    // Shapes: input=[N,C_in,H_in,W_in]  weight=[C_out,C_in,kH,kW]
    std::vector<TensorShape> inferOutputShapes(
        const std::vector<TensorShape>& input_shapes) const override
    {
        if (input_shapes.size() < 2 || input_shapes.size() > 3)
            throw std::invalid_argument(
                "Conv2DOp: expects 2 or 3 inputs (input, weight[, bias])");

        const auto& ishape = input_shapes[0];
        const auto& wshape = input_shapes[1];

        if (ishape.ndim() != 4)
            throw std::invalid_argument(
                "Conv2DOp: input must be 4-D (NCHW), got " +
                std::to_string(ishape.ndim()) + "-D");
        if (wshape.ndim() != 4)
            throw std::invalid_argument(
                "Conv2DOp: weight must be 4-D (C_out,C_in,kH,kW), got " +
                std::to_string(wshape.ndim()) + "-D");

        const size_t N      = ishape.dim(0);
        const size_t C_in   = ishape.dim(1);
        const size_t H_in   = ishape.dim(2);
        const size_t W_in   = ishape.dim(3);
        const size_t C_out  = wshape.dim(0);
        const size_t C_in_w = wshape.dim(1);
        const size_t kH     = wshape.dim(2);
        const size_t kW     = wshape.dim(3);

        if (C_in != C_in_w)
            throw std::invalid_argument(
                "Conv2DOp: input channels (" + std::to_string(C_in) +
                ") != weight C_in (" + std::to_string(C_in_w) + ")");

        // Verify params are consistent with weight shape
        if (static_cast<size_t>(p_.kH) != kH || static_cast<size_t>(p_.kW) != kW)
            throw std::invalid_argument(
                "Conv2DOp: params.kH/kW mismatch with weight shape");

        if (input_shapes.size() == 3) {
            const auto& bshape = input_shapes[2];
            if (bshape.ndim() != 1 || bshape.dim(0) != C_out)
                throw std::invalid_argument(
                    "Conv2DOp: bias shape must be [C_out]");
        }

        const size_t H_out = (H_in + 2 * static_cast<size_t>(p_.pad_h)
                               - kH) / static_cast<size_t>(p_.stride_h) + 1;
        const size_t W_out = (W_in + 2 * static_cast<size_t>(p_.pad_w)
                               - kW) / static_cast<size_t>(p_.stride_w) + 1;

        return { TensorShape({N, C_out, H_out, W_out}) };
    }

    // ------------------------------------------------------------------
    // CPU: direct (naive) convolution — O(N * C_out * H_out * W_out * C_in * kH * kW)
    // ------------------------------------------------------------------
    void executeCPU(
        const std::vector<const Tensor*>& inputs,
        const std::vector<Tensor*>&       outputs) override
    {
        const bool has_bias = (inputs.size() == 3);
        validateInputCount("Conv2DOp", has_bias ? 3 : 2, inputs.size());
        validateInputCount("Conv2DOp outputs", 1, outputs.size());

        const Tensor* input  = inputs[0];
        const Tensor* weight = inputs[1];
        const Tensor* bias   = has_bias ? inputs[2] : nullptr;
        Tensor*       output = outputs[0];

        for (const Tensor* t : inputs)  validateDevice("Conv2DOp", t, Device::CPU);
        validateDevice("Conv2DOp", output, Device::CPU);
        for (const Tensor* t : inputs)  validateDtype("Conv2DOp", t, DataType::Float32);
        validateDtype("Conv2DOp", output, DataType::Float32);

        const int N     = static_cast<int>(input->shape().dim(0));
        const int C_in  = static_cast<int>(input->shape().dim(1));
        const int H_in  = static_cast<int>(input->shape().dim(2));
        const int W_in  = static_cast<int>(input->shape().dim(3));
        const int C_out = static_cast<int>(weight->shape().dim(0));
        const int kH    = p_.kH, kW = p_.kW;
        const int sH    = p_.stride_h, sW = p_.stride_w;
        const int pH    = p_.pad_h,    pW = p_.pad_w;
        const int H_out = (H_in + 2*pH - kH) / sH + 1;
        const int W_out = (W_in + 2*pW - kW) / sW + 1;

        const float* in_d  = static_cast<const float*>(input->data());
        const float* w_d   = static_cast<const float*>(weight->data());
        const float* b_d   = has_bias ? static_cast<const float*>(bias->data()) : nullptr;
        float*       out_d = static_cast<float*>(output->data());

        // Direct convolution — correct reference implementation, not optimised.
        for (int n = 0; n < N; ++n) {
            for (int co = 0; co < C_out; ++co) {
                const float bias_val = b_d ? b_d[co] : 0.0f;
                for (int oh = 0; oh < H_out; ++oh) {
                    for (int ow = 0; ow < W_out; ++ow) {
                        float sum = bias_val;
                        for (int ci = 0; ci < C_in; ++ci) {
                            for (int kh = 0; kh < kH; ++kh) {
                                const int ih = oh * sH - pH + kh;
                                if (ih < 0 || ih >= H_in) continue;
                                for (int kw2 = 0; kw2 < kW; ++kw2) {
                                    const int iw = ow * sW - pW + kw2;
                                    if (iw < 0 || iw >= W_in) continue;
                                    // input [N, C_in, H_in, W_in] row-major
                                    const float in_val =
                                        in_d[((n * C_in + ci) * H_in + ih) * W_in + iw];
                                    // weight [C_out, C_in, kH, kW] row-major
                                    const float w_val =
                                        w_d[((co * C_in + ci) * kH + kh) * kW + kw2];
                                    sum += in_val * w_val;
                                }
                            }
                        }
                        // output [N, C_out, H_out, W_out] row-major
                        out_d[((n * C_out + co) * H_out + oh) * W_out + ow] = sum;
                    }
                }
            }
        }
    }

#ifdef FORGERT_CUDA_AVAILABLE
    void executeCUDA(
        const std::vector<const Tensor*>& inputs,
        const std::vector<Tensor*>&       outputs) override
    {
        const bool has_bias = (inputs.size() == 3);
        validateInputCount("Conv2DOp", has_bias ? 3 : 2, inputs.size());
        validateInputCount("Conv2DOp outputs", 1, outputs.size());

        const Tensor* input  = inputs[0];
        const Tensor* weight = inputs[1];
        const Tensor* bias   = has_bias ? inputs[2] : nullptr;
        Tensor*       output = outputs[0];

        for (const Tensor* t : inputs)  validateDevice("Conv2DOp", t, Device::CUDA);
        validateDevice("Conv2DOp", output, Device::CUDA);
        for (const Tensor* t : inputs)  validateDtype("Conv2DOp", t, DataType::Float32);
        validateDtype("Conv2DOp", output, DataType::Float32);

        const int N     = static_cast<int>(input->shape().dim(0));
        const int C_in  = static_cast<int>(input->shape().dim(1));
        const int H_in  = static_cast<int>(input->shape().dim(2));
        const int W_in  = static_cast<int>(input->shape().dim(3));
        const int C_out = static_cast<int>(weight->shape().dim(0));

        const float* d_in     = static_cast<const float*>(input->data());
        const float* d_weight = static_cast<const float*>(weight->data());
        const float* d_bias   = has_bias
                                  ? static_cast<const float*>(bias->data())
                                  : nullptr;
        float*       d_out    = static_cast<float*>(output->data());

        const int err = forgert_conv2d_cuda(
            d_in, d_weight, d_bias, d_out,
            N, C_in, H_in, W_in,
            C_out, p_.kH, p_.kW,
            p_.stride_h, p_.stride_w,
            p_.pad_h,    p_.pad_w);

        if (err != 0)
            throw std::runtime_error(
                "Conv2DOp::executeCUDA: kernel failed (cudaError_t=" +
                std::to_string(err) + ")");
    }
#endif

private:
    Conv2DParams p_;
};

} // namespace forgert

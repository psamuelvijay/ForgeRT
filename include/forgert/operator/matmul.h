#pragma once

#include "forgert/operator/operator.h"

// When FORGERT_CUDA_AVAILABLE is defined, bring in the extern "C" kernel launcher
#ifdef FORGERT_CUDA_AVAILABLE
extern "C" int forgert_matmul_cuda(const float* A, const float* B, float* C,
                                   size_t M, size_t K, size_t N);
#endif

namespace forgert {

/**
 * @brief Matrix multiplication operator: C = A @ B
 * 
 * Performs standard matrix multiplication on 2D tensors.
 * 
 * Shape requirements:
 * - A: [M, K]
 * - B: [K, N]
 * - C: [M, N]
 * 
 * The inner dimensions (K) must match.
 * 
 * Phase 2: CPU implementation only, Float32 only, 2D matrices only
 * Phase 3: CUDA implementation
 * Phase 4+: Batched matrix multiplication, mixed precision
 * 
 * Examples:
 *   [3, 4] @ [4, 5] = [3, 5]  ✓
 *   [2, 3] @ [3, 1] = [2, 1]  ✓
 *   [1, 5] @ [5, 1] = [1, 1]  ✓
 *   [3, 4] @ [5, 6] = error   ✗ (inner dimensions don't match)
 *   [3]    @ [3, 4] = error   ✗ (not 2D)
 */
class MatMulOp : public Operator {
public:
    std::string name() const override {
        return "MatMul";
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
        
        validateInputCount("MatMulOp", 2, input_shapes.size());

        const auto& shape_a = input_shapes[0];
        const auto& shape_b = input_shapes[1];

        // Phase 2: Only 2D matrices supported
        if (shape_a.ndim() != 2) {
            throw std::invalid_argument(
                "MatMulOp: First input must be 2D, got " + 
                std::to_string(shape_a.ndim()) + "D");
        }
        if (shape_b.ndim() != 2) {
            throw std::invalid_argument(
                "MatMulOp: Second input must be 2D, got " + 
                std::to_string(shape_b.ndim()) + "D");
        }

        size_t M = shape_a.dim(0);  // Rows of A
        size_t K_a = shape_a.dim(1);  // Cols of A
        size_t K_b = shape_b.dim(0);  // Rows of B
        size_t N = shape_b.dim(1);  // Cols of B

        // Inner dimensions must match
        if (K_a != K_b) {
            throw std::invalid_argument(
                "MatMulOp: Inner dimensions must match. Got A=[" +
                std::to_string(M) + "," + std::to_string(K_a) + "] and B=[" +
                std::to_string(K_b) + "," + std::to_string(N) + "]");
        }

        // Output shape is [M, N]
        return {TensorShape({M, N})};
    }

    void executeCPU(
        const std::vector<const Tensor*>& inputs,
        const std::vector<Tensor*>& outputs) override {
        
        validateInputCount("MatMulOp", 2, inputs.size());
        validateInputCount("MatMulOp outputs", 1, outputs.size());

        const Tensor* a = inputs[0];
        const Tensor* b = inputs[1];
        Tensor* c = outputs[0];

        // Validate devices
        validateDevice("MatMulOp", a, Device::CPU);
        validateDevice("MatMulOp", b, Device::CPU);
        validateDevice("MatMulOp", c, Device::CPU);

        // Phase 2: Float32 only
        validateDtype("MatMulOp", a, DataType::Float32);
        validateDtype("MatMulOp", b, DataType::Float32);
        validateDtype("MatMulOp", c, DataType::Float32);

        // Get dimensions
        size_t M = a->shape().dim(0);
        size_t K = a->shape().dim(1);
        size_t N = b->shape().dim(1);

        // Get raw pointers
        const float* a_data = static_cast<const float*>(a->data());
        const float* b_data = static_cast<const float*>(b->data());
        float* c_data = static_cast<float*>(c->data());

        // Perform matrix multiplication: C[i,j] = sum(A[i,k] * B[k,j])
        // Simple O(M*N*K) implementation
        // Phase 3: Optimize with blocking, SIMD, or CUDA
        
        // Initialize output to zero
        for (size_t i = 0; i < M * N; ++i) {
            c_data[i] = 0.0f;
        }

        // Standard matrix multiplication (row-major layout)
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    // A is row-major: A[i,k] = a_data[i * K + k]
                    // B is row-major: B[k,j] = b_data[k * N + j]
                    sum += a_data[i * K + k] * b_data[k * N + j];
                }
                // C is row-major: C[i,j] = c_data[i * N + j]
                c_data[i * N + j] = sum;
            }
        }
    }

#ifdef FORGERT_CUDA_AVAILABLE
    void executeCUDA(
        const std::vector<const Tensor*>& inputs,
        const std::vector<Tensor*>& outputs) override {
        
        validateInputCount("MatMulOp", 2, inputs.size());
        validateInputCount("MatMulOp outputs", 1, outputs.size());

        const Tensor* a = inputs[0];
        const Tensor* b = inputs[1];
        Tensor* c = outputs[0];

        // Validate devices
        validateDevice("MatMulOp", a, Device::CUDA);
        validateDevice("MatMulOp", b, Device::CUDA);
        validateDevice("MatMulOp", c, Device::CUDA);

        // Validate data types
        validateDtype("MatMulOp", a, DataType::Float32);
        validateDtype("MatMulOp", b, DataType::Float32);
        validateDtype("MatMulOp", c, DataType::Float32);

        // Get dimensions
        size_t M = a->shape().dim(0);
        size_t K = a->shape().dim(1);
        size_t N = b->shape().dim(1);

        // Get device pointers
        const float* d_a = static_cast<const float*>(a->data());
        const float* d_b = static_cast<const float*>(b->data());
        float* d_c = static_cast<float*>(c->data());

        // Launch CUDA kernel
        const int err = forgert_matmul_cuda(d_a, d_b, d_c, M, K, N);
        if (err != 0) {
            throw std::runtime_error(
                "MatMulOp::executeCUDA: kernel launcher failed (cudaError_t=" +
                std::to_string(err) + ")");
        }
    }
#endif // FORGERT_CUDA_AVAILABLE
};

} // namespace forgert

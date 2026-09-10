#pragma once
#include <cstddef>

/*
 * Host-callable Softmax kernel launcher.
 *
 * Computes softmax along the last dimension:
 *   softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
 *
 * Input/output tensor is treated as [outer_size, inner_size] where:
 * - outer_size = total_elements / last_dim_size
 * - inner_size = last_dim_size (the dimension to normalize over)
 *
 * Each of the outer_size rows is independently normalized.
 * Uses numerically stable implementation with max subtraction.
 *
 * Returns 0 on success, non-zero (cudaError_t cast to int) on failure.
 * Declared extern "C" so CXX translation units can call it without
 * including cuda_runtime.h.
 */
extern "C" int forgert_softmax_cuda(const float* input, float* output,
                                    size_t outer_size, size_t inner_size);
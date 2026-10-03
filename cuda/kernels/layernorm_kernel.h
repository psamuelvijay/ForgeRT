#pragma once
#include <cstddef>

/*
 * Host-callable LayerNorm kernel launcher.
 *
 * Computes LayerNorm along the last dimension:
 *   output = (input - mean) / sqrt(variance + epsilon)
 *
 * Where mean and variance are computed per sample over the last dimension.
 * 
 * Input/output tensor is treated as [outer_size, inner_size] where:
 * - outer_size = total_elements / last_dim_size  
 * - inner_size = last_dim_size (the dimension to normalize over)
 *
 * Each of the outer_size rows is independently normalized.
 * Uses numerically stable two-pass algorithm:
 *   Pass 1: Compute mean
 *   Pass 2: Compute variance and normalize
 *
 * Returns 0 on success, non-zero (cudaError_t cast to int) on failure.
 * Declared extern "C" so CXX translation units can call it without
 * including cuda_runtime.h.
 */
extern "C" int forgert_layernorm_cuda(const float* input, float* output,
                                      size_t outer_size, size_t inner_size,
                                      float epsilon);
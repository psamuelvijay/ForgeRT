#pragma once
#include <cstddef>

/*
 * Host-callable ReLU kernel launcher.
 *
 * Returns 0 on success, non-zero (cudaError_t cast to int) on failure.
 * Declared extern "C" so CXX translation units can call it without
 * including cuda_runtime.h.
 */
extern "C" int forgert_relu_cuda(const float* input, float* output, size_t n);

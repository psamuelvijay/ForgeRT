#pragma once
#include <cstddef>

/*
 * Host-callable matrix multiplication kernel launcher.
 *
 * Computes C = A @ B where:
 * - A is [M x K]
 * - B is [K x N] 
 * - C is [M x N]
 *
 * All matrices are in row-major layout.
 *
 * Returns 0 on success, non-zero (cudaError_t cast to int) on failure.
 * Declared extern "C" so CXX translation units can call it without
 * including cuda_runtime.h.
 */
extern "C" int forgert_matmul_cuda(const float* A, const float* B, float* C,
                                   size_t M, size_t K, size_t N);
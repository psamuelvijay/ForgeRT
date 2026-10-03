#pragma once
#include <cstddef>

/*
 * Host-callable Add kernel launcher.
 *
 * Computes element-wise addition C = A + B with NumPy-style broadcasting.
 *
 * Simple path  (a_size == b_size == c_size):
 *   One thread per element; output[i] = a[i] + b[i].
 *
 * Broadcast path:
 *   Each output element is computed from the corresponding A/B positions
 *   after applying broadcasting rules (dimensions of size 1 are broadcast
 *   to match the other operand).  Up to FORGERT_ADD_MAX_DIMS dimensions
 *   are supported.
 *
 * Returns 0 on success, non-zero (cudaError_t cast to int) on failure.
 * Declared extern "C" so CXX translation units can call it without
 * including cuda_runtime.h.
 *
 * Parameters
 * ----------
 * a, b         : input device pointers (Float32)
 * c            : output device pointer  (Float32)
 * a_dims       : host array of A's dimensions (length ndim)
 * b_dims       : host array of B's dimensions (length ndim)
 * c_dims       : host array of C's dimensions (length ndim)
 * ndim         : number of dimensions (same for A, B, C after left-padding with 1s)
 * c_size       : total number of output elements
 */
extern "C" int forgert_add_cuda(const float* a, const float* b, float* c,
                                 const size_t* a_dims, const size_t* b_dims,
                                 const size_t* c_dims, int ndim,
                                 size_t c_size);

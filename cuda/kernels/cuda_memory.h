#pragma once
#include <cstddef>

/*
 * Plain-C wrappers around the CUDA memory API so that tensor.h can call
 * cudaMalloc / cudaFree / cudaMemcpy / cudaMemset without including
 * cuda_runtime.h in a host-facing header.
 *
 * Every function returns 0 on success or the raw cudaError_t cast to int.
 *
 * cudaMemcpy direction constants are replicated here as an enum so callers
 * do not need cuda_runtime.h either.
 */

enum class CudaMemcpyKind {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
};

extern "C" {

/* Allocate `bytes` bytes of device memory. *ptr receives the address. */
int forgert_cuda_malloc(void** ptr, size_t bytes);

/* Free device memory previously allocated by forgert_cuda_malloc. */
int forgert_cuda_free(void* ptr);

/* Copy `bytes` bytes between host and device. */
int forgert_cuda_memcpy(void* dst, const void* src, size_t bytes,
                        CudaMemcpyKind kind);

/* Zero-fill `bytes` bytes of device memory. */
int forgert_cuda_memset(void* ptr, int value, size_t bytes);

} // extern "C"

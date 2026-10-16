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
 *
 * NOTE: The enum class must be declared OUTSIDE the extern "C" block.
 * On CUDA 12.6 + MSVC 14.39 on Windows, having enum class inside an
 * extern "C" block causes cudafe++ to crash with ACCESS_VIOLATION.
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

/* Query available CUDA device memory (Phase 4).
 * Sets *free_bytes and *total_bytes to the free and total device memory.
 * Returns 0 on success or the raw cudaError_t cast to int. */
int forgert_cuda_mem_info(size_t* free_bytes, size_t* total_bytes);

} // extern "C"

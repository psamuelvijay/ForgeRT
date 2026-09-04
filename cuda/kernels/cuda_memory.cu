/*
 * cuda_memory.cu
 *
 * Thin extern-C wrappers around the CUDA memory API.
 * Compiled by nvcc; callable from plain CXX translation units via
 * cuda_memory.h without requiring cuda_runtime.h in host headers.
 *
 * All functions return 0 on success or the raw cudaError_t cast to int.
 */

#include <cuda_runtime.h>
#include <cstring>    // memcpy for the H2D fill helper
#include "cuda_memory.h"

extern "C" {

int forgert_cuda_malloc(void** ptr, size_t bytes) {
    return static_cast<int>(cudaMalloc(ptr, bytes));
}

int forgert_cuda_free(void* ptr) {
    return static_cast<int>(cudaFree(ptr));
}

int forgert_cuda_memcpy(void* dst, const void* src, size_t bytes,
                        CudaMemcpyKind kind) {
    cudaMemcpyKind ck;
    switch (kind) {
        case CudaMemcpyKind::HostToDevice:   ck = cudaMemcpyHostToDevice;   break;
        case CudaMemcpyKind::DeviceToHost:   ck = cudaMemcpyDeviceToHost;   break;
        case CudaMemcpyKind::DeviceToDevice: ck = cudaMemcpyDeviceToDevice; break;
        default:                             return static_cast<int>(cudaErrorInvalidValue);
    }
    return static_cast<int>(cudaMemcpy(dst, src, bytes, ck));
}

int forgert_cuda_memset(void* ptr, int value, size_t bytes) {
    return static_cast<int>(cudaMemset(ptr, value, bytes));
}

} // extern "C"

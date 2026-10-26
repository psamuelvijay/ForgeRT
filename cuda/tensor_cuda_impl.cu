/**
 * @file tensor_cuda_impl.cu
 * @brief Real (CUDA-backed) definition of Tensor::cudaTransferImpl.
 *
 * Compiled into forgert_cuda (FORGERT_CUDA_AVAILABLE defined). Provides the
 * strong definition of the out-of-line transfer helper that Tensor::copyTo
 * calls. When forgert_cuda is linked, this definition is used; otherwise the
 * CPU-only fallback in src/tensor/tensor_cuda_stub.cpp is pulled instead.
 * See that file for the link-time selection contract.
 *
 * This file performs no device kernels -- it only maps the Tensor device tags
 * to a CudaMemcpyKind and forwards to forgert_cuda_memcpy (the plain-C wrapper
 * around cudaMemcpy). Memory-transfer accounting (recordTransfer) stays in the
 * inline copyTo body in tensor.h, so it is identical for both backends.
 */
#include "forgert/tensor/tensor.h"
#include "kernels/cuda_memory.h"

#include <stdexcept>
#include <string>

namespace forgert {

void Tensor::cudaTransferImpl(void* dst, const void* src, size_t nbytes,
                              Device src_dev, Device dst_dev) {
    CudaMemcpyKind kind;
    if (src_dev == Device::CPU && dst_dev == Device::CUDA)
        kind = CudaMemcpyKind::HostToDevice;
    else if (src_dev == Device::CUDA && dst_dev == Device::CPU)
        kind = CudaMemcpyKind::DeviceToHost;
    else
        kind = CudaMemcpyKind::DeviceToDevice;

    const int err = forgert_cuda_memcpy(dst, src, nbytes, kind);
    if (err != 0) {
        throw std::runtime_error(
            "Tensor::copyTo: cudaMemcpy failed (cudaError_t=" +
            std::to_string(err) + ")");
    }
}

void Tensor::cudaAllocImpl(void** ptr, size_t nbytes) {
    const int err = forgert_cuda_malloc(ptr, nbytes);
    if (err != 0) {
        throw std::runtime_error(
            "Tensor: CUDA memory allocation failed (cudaError_t=" +
            std::to_string(err) + ")");
    }
}

void Tensor::cudaFreeImpl(void* ptr) noexcept {
    (void)forgert_cuda_free(ptr);  // dtor path: errors are intentionally ignored
}

void Tensor::cudaMemsetImpl(void* ptr, int value, size_t nbytes) {
    const int err = forgert_cuda_memset(ptr, value, nbytes);
    if (err != 0) {
        throw std::runtime_error(
            "Tensor::zero: cudaMemset failed (cudaError_t=" +
            std::to_string(err) + ")");
    }
}

} // namespace forgert
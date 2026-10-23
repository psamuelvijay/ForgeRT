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
#include "forgert/tensor/tensor.h"   // FORGERT_CUDA_AVAILABLE pulls in cuda_memory.h

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

} // namespace forgert
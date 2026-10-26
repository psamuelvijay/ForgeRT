/**
 * @file tensor_cuda_stub.cpp
 * @brief CPU-only fallback definition of Tensor::cudaTransferImpl.
 *
 * Tensor::copyTo() dispatches every CUDA (H2D / D2H / D2D) transfer to the
 * out-of-line Tensor::cudaTransferImpl so that the inline copyTo body does not
 * depend on FORGERT_CUDA_AVAILABLE and is therefore byte-identical in every
 * translation unit. Two definitions of cudaTransferImpl exist and are chosen at
 * LINK time:
 *
 *   - forgert_cuda provides the REAL implementation (cuda/tensor_cuda_impl.cu),
 *     which calls forgert_cuda_memcpy. It is used whenever forgert_cuda is
 *     linked into the final binary.
 *   - THIS file provides the CPU-only fallback, compiled into the `forgert`
 *     core library (no CUDA dependency). It throws, preserving the previous
 *     behaviour of a CPU-only build that attempts a CUDA transfer.
 *
 * Link-time selection: each definition lives in its own object file, referenced
 * only through the single symbol Tensor::cudaTransferImpl. A static library
 * contributes an object only to resolve an otherwise-undefined symbol, so when
 * forgert_cuda is on the link line its real definition satisfies the reference
 * first and this fallback object is never pulled in -- no duplicate symbol. A
 * pure CPU build (forgert without forgert_cuda) pulls this fallback instead.
 */
#include "forgert/tensor/tensor.h"

#include <stdexcept>

namespace forgert {

void Tensor::cudaTransferImpl(void* /*dst*/, const void* /*src*/,
                              size_t /*nbytes*/, Device /*src_dev*/,
                              Device /*dst_dev*/) {
    throw std::runtime_error(
        "Tensor::copyTo: CUDA transfer requested but no CUDA backend is linked "
        "(link forgert_cuda to enable H2D/D2H/D2D transfers)");
}

void Tensor::cudaAllocImpl(void** /*ptr*/, size_t /*nbytes*/) {
    throw std::runtime_error(
        "Tensor: CUDA device support requires a CUDA backend "
        "(link against forgert_cuda)");
}

// Nothing can have been allocated without a CUDA backend; nothing to free.
void Tensor::cudaFreeImpl(void* /*ptr*/) noexcept {}

void Tensor::cudaMemsetImpl(void* /*ptr*/, int /*value*/, size_t /*nbytes*/) {
    throw std::runtime_error(
        "Tensor::zero: CUDA support not available (link forgert_cuda)");
}

} // namespace forgert
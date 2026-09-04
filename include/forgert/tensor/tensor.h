#pragma once

#include "forgert/tensor/shape.h"
#include "forgert/tensor/dtype.h"
#include <memory>
#include <cstring>
#include <stdexcept>
#include <string>

// When built as part of a CUDA-enabled target (FORGERT_CUDA_AVAILABLE defined),
// pull in the plain-C memory wrappers so the constructor / destructor / helpers
// can call cudaMalloc/cudaFree/cudaMemcpy without exposing cuda_runtime.h here.
#ifdef FORGERT_CUDA_AVAILABLE
#include "kernels/cuda_memory.h"
#endif

namespace forgert {

/**
 * @brief Memory location for tensor data
 */
enum class Device {
    CPU,    // Host memory allocated with std::malloc
    CUDA    // GPU device memory allocated with cudaMalloc
};

/**
 * @brief Tensor abstraction for ForgeRT
 *
 * Owns (or borrows) a contiguous buffer of elements described by a TensorShape
 * and DataType.  Memory is either on the CPU host or on a CUDA device.
 *
 * Ownership rules:
 *   - Owning tensor:   allocated in constructor, freed in destructor.
 *   - Non-owning tensor (external memory): data() pointer supplied by caller;
 *     destructor does nothing for the buffer.  Use this to wrap cudaMalloc
 *     buffers managed outside ForgeRT (e.g. in tests or future allocators).
 *
 * CUDA support:
 *   - Only available when FORGERT_CUDA_AVAILABLE is defined.
 *   - Allocation: forgert_cuda_malloc (wraps cudaMalloc).
 *   - Deallocation: forgert_cuda_free (wraps cudaFree).
 *   - zero() uses cudaMemset; fill() stages a host buffer and copies H2D.
 *   - copyTo() / copyFrom() provide explicit H2D and D2H transfers.
 *   - All CUDA errors are surfaced as std::runtime_error.
 *
 * Without FORGERT_CUDA_AVAILABLE any attempt to create a Device::CUDA tensor
 * throws std::runtime_error, preserving the previous behaviour for CPU-only
 * builds.
 */
class Tensor {
public:
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /**
     * @brief Allocate an owning tensor on the given device.
     *
     * @param shape  Tensor shape.
     * @param dtype  Element data type.
     * @param device Memory location (CPU or CUDA).
     * @throws std::runtime_error on allocation failure or if CUDA is requested
     *         but FORGERT_CUDA_AVAILABLE is not defined.
     */
    Tensor(const TensorShape& shape, DataType dtype, Device device = Device::CPU)
        : shape_(shape), dtype_(dtype), device_(device), owns_memory_(true) {

        const size_t num_bytes = shape.numElements() * sizeOf(dtype);

        if (device == Device::CPU) {
            data_ = std::malloc(num_bytes);
            if (!data_) {
                throw std::runtime_error("Tensor: CPU memory allocation failed");
            }
        } else {
#ifdef FORGERT_CUDA_AVAILABLE
            const int err = forgert_cuda_malloc(&data_, num_bytes);
            if (err != 0) {
                throw std::runtime_error(
                    "Tensor: CUDA memory allocation failed (cudaError_t=" +
                    std::to_string(err) + ")");
            }
#else
            throw std::runtime_error(
                "Tensor: CUDA device support requires FORGERT_CUDA_AVAILABLE "
                "(link against forgert_cuda)");
#endif
        }
    }

    /**
     * @brief Non-owning tensor wrapping externally-managed memory.
     *
     * The caller retains ownership of the buffer; this tensor's destructor
     * will not free it.  Use this to wrap raw cudaMalloc/malloc pointers.
     *
     * @param shape  Tensor shape.
     * @param dtype  Element data type.
     * @param data   Pointer to the pre-allocated buffer.
     * @param device Memory location the pointer lives in.
     */
    Tensor(const TensorShape& shape, DataType dtype, void* data,
           Device device = Device::CPU)
        : shape_(shape), dtype_(dtype), device_(device),
          data_(data), owns_memory_(false) {}

    // Disable copy
    Tensor(const Tensor&)            = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Enable move
    Tensor(Tensor&& other) noexcept
        : shape_(std::move(other.shape_))
        , dtype_(other.dtype_)
        , device_(other.device_)
        , data_(other.data_)
        , owns_memory_(other.owns_memory_) {
        other.data_        = nullptr;
        other.owns_memory_ = false;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            cleanup();
            shape_       = std::move(other.shape_);
            dtype_       = other.dtype_;
            device_      = other.device_;
            data_        = other.data_;
            owns_memory_ = other.owns_memory_;
            other.data_        = nullptr;
            other.owns_memory_ = false;
        }
        return *this;
    }

    ~Tensor() { cleanup(); }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------
    const TensorShape& shape()    const { return shape_; }
    DataType           dtype()    const { return dtype_; }
    Device             device()   const { return device_; }
    void*              data()           { return data_; }
    const void*        data()     const { return data_; }
    size_t             numElements() const { return shape_.numElements(); }
    size_t             numBytes()    const { return numElements() * sizeOf(dtype_); }

    // ------------------------------------------------------------------
    // In-place initialisation
    // ------------------------------------------------------------------

    /**
     * @brief Zero-fill the tensor buffer (CPU: memset; CUDA: cudaMemset).
     */
    void zero() {
        if (device_ == Device::CPU) {
            std::memset(data_, 0, numBytes());
        } else {
#ifdef FORGERT_CUDA_AVAILABLE
            const int err = forgert_cuda_memset(data_, 0, numBytes());
            if (err != 0) {
                throw std::runtime_error(
                    "Tensor::zero: cudaMemset failed (cudaError_t=" +
                    std::to_string(err) + ")");
            }
#else
            throw std::runtime_error("Tensor::zero: CUDA support not available");
#endif
        }
    }

    /**
     * @brief Fill with a scalar value.
     *
     * CPU: direct loop.
     * CUDA: allocate a host staging buffer, fill it, then copy H2D.
     *
     * Only Float32 is supported (consistent with Phase 2/3 operators).
     */
    void fill(float value) {
        if (dtype_ != DataType::Float32) {
            throw std::runtime_error("Tensor::fill: only Float32 supported");
        }

        if (device_ == Device::CPU) {
            float* ptr = static_cast<float*>(data_);
            const size_t n = numElements();
            for (size_t i = 0; i < n; ++i) ptr[i] = value;
        } else {
#ifdef FORGERT_CUDA_AVAILABLE
            // Stage on host then copy H2D.
            const size_t nb = numBytes();
            std::unique_ptr<float[]> host(new float[numElements()]);
            for (size_t i = 0; i < numElements(); ++i) host[i] = value;
            const int err = forgert_cuda_memcpy(
                data_, host.get(), nb, CudaMemcpyKind::HostToDevice);
            if (err != 0) {
                throw std::runtime_error(
                    "Tensor::fill: H2D memcpy failed (cudaError_t=" +
                    std::to_string(err) + ")");
            }
#else
            throw std::runtime_error("Tensor::fill: CUDA support not available");
#endif
        }
    }

    // ------------------------------------------------------------------
    // Explicit memory transfers (Phase 4)
    // ------------------------------------------------------------------

    /**
     * @brief Copy data into another tensor.
     *
     * Supports all combinations of CPU and CUDA:
     *   CPU  → CPU   (memcpy)
     *   CPU  → CUDA  (H2D)
     *   CUDA → CPU   (D2H)
     *   CUDA → CUDA  (D2D)
     *
     * @param dst Destination tensor.  Must have the same element count and
     *            dtype as this tensor.
     * @throws std::invalid_argument if shapes / dtypes are incompatible.
     * @throws std::runtime_error    on transfer failure.
     */
    void copyTo(Tensor& dst) const {
        if (dst.numElements() != numElements()) {
            throw std::invalid_argument(
                "Tensor::copyTo: element count mismatch (" +
                std::to_string(numElements()) + " vs " +
                std::to_string(dst.numElements()) + ")");
        }
        if (dst.dtype_ != dtype_) {
            throw std::invalid_argument(
                "Tensor::copyTo: dtype mismatch (" +
                toString(dtype_) + " vs " + toString(dst.dtype_) + ")");
        }

        const size_t nb = numBytes();

        // Both CPU
        if (device_ == Device::CPU && dst.device_ == Device::CPU) {
            std::memcpy(dst.data_, data_, nb);
            return;
        }

#ifdef FORGERT_CUDA_AVAILABLE
        CudaMemcpyKind kind;
        if (device_ == Device::CPU  && dst.device_ == Device::CUDA)
            kind = CudaMemcpyKind::HostToDevice;
        else if (device_ == Device::CUDA && dst.device_ == Device::CPU)
            kind = CudaMemcpyKind::DeviceToHost;
        else
            kind = CudaMemcpyKind::DeviceToDevice;

        const int err = forgert_cuda_memcpy(dst.data_, data_, nb, kind);
        if (err != 0) {
            throw std::runtime_error(
                "Tensor::copyTo: cudaMemcpy failed (cudaError_t=" +
                std::to_string(err) + ")");
        }
#else
        throw std::runtime_error(
            "Tensor::copyTo: CUDA transfers require FORGERT_CUDA_AVAILABLE");
#endif
    }

private:
    TensorShape shape_;
    DataType    dtype_;
    Device      device_;
    void*       data_        = nullptr;
    bool        owns_memory_ = false;

    void cleanup() noexcept {
        if (!owns_memory_ || !data_) return;
        if (device_ == Device::CPU) {
            std::free(data_);
        } else {
#ifdef FORGERT_CUDA_AVAILABLE
            forgert_cuda_free(data_);  // ignore return; can't throw in dtor
#endif
        }
        data_        = nullptr;
        owns_memory_ = false;
    }
};

} // namespace forgert

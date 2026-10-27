#pragma once

#include "forgert/tensor/shape.h"
#include "forgert/tensor/dtype.h"
#include "forgert/tensor/device.h"
#include <cstdint>
#include <memory>
#include <cstring>
#include <stdexcept>
#include <string>

// NOTE: this header deliberately does NOT depend on FORGERT_CUDA_AVAILABLE or on
// any CUDA header. All device memory operations (alloc, free, memset, memcpy) go
// through out-of-line Tensor::cuda*Impl functions whose definition is selected at
// link time (forgert_cuda = real, forgert_cpu_stub = throwing fallback). Every
// inline Tensor body is therefore identical in every translation unit.

namespace forgert {

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
        : shape_(shape), dtype_(dtype), device_(device), owns_memory_(true),
          id_(nextId()) {

        const size_t num_bytes = shape.numElements() * sizeOf(dtype);

        if (device == Device::CPU) {
            data_ = std::malloc(num_bytes);
            if (!data_) {
                throw std::runtime_error("Tensor: CPU memory allocation failed");
            }
        } else {
            // Link-time dispatched (see cudaAllocImpl); throws if no CUDA backend.
            cudaAllocImpl(&data_, num_bytes);
        }
        
        // Track allocation (Phase 4)
        recordAllocation(num_bytes);
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
          data_(data), owns_memory_(false), id_(nextId()) {}

    // Disable copy
    Tensor(const Tensor&)            = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Enable move
    Tensor(Tensor&& other) noexcept
        : shape_(std::move(other.shape_))
        , dtype_(other.dtype_)
        , device_(other.device_)
        , data_(other.data_)
        , owns_memory_(other.owns_memory_)
        , id_(other.id_)
        , frozen_gen_(other.frozen_gen_) {
        // Identity (id + freeze state) travels with the buffer.
        other.data_        = nullptr;
        other.owns_memory_ = false;
        other.id_          = 0;
        other.frozen_gen_  = 0;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            cleanup();
            shape_       = std::move(other.shape_);
            dtype_       = other.dtype_;
            device_      = other.device_;
            data_        = other.data_;
            owns_memory_ = other.owns_memory_;
            id_          = other.id_;
            frozen_gen_  = other.frozen_gen_;
            other.data_        = nullptr;
            other.owns_memory_ = false;
            other.id_          = 0;
            other.frozen_gen_  = 0;
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
    /// Mutable access. Throws std::logic_error if the tensor is frozen: use the
    /// const overload (or a const Tensor&) to read an immutable tensor.
    void*              data()           { if (frozen_gen_) throwFrozen("data()"); return data_; }
    const void*        data()     const { return data_; }
    size_t             numElements() const { return shape_.numElements(); }
    size_t             numBytes()    const { return numElements() * sizeOf(dtype_); }

    // ------------------------------------------------------------------
    // Identity and immutability (used by the persistent device-weight cache)
    // ------------------------------------------------------------------

    /// Process-unique identity of this buffer (never reused, never an address).
    /// Moving a tensor transfers its id; a moved-from tensor has id 0.
    uint64_t id() const { return id_; }

    /// True while the tensor is frozen (contents guaranteed immutable).
    bool isFrozen() const { return frozen_gen_ != 0; }

    /// Identifies one freeze epoch. Changes every time the tensor is frozen
    /// again after an unfreeze(), so anything cached against an earlier epoch is
    /// detectably stale. 0 while unfrozen.
    uint64_t freezeGeneration() const { return frozen_gen_; }

    /**
     * @brief Declare the contents immutable (model weights, constants).
     *
     * While frozen, every mutation path throws std::logic_error: mutable data(),
     * zero(), fill(), and being the destination of copyTo(). This is what makes
     * it safe for a cache to keep a device copy keyed on (id, generation): the
     * content cannot change without an explicit unfreeze() that invalidates it.
     * Only owning tensors can be frozen -- ForgeRT cannot police memory it does
     * not own, so a non-owning tensor throws std::logic_error. Idempotent.
     */
    void freeze() {
        if (frozen_gen_) return;
        if (!owns_memory_ || !data_) {
            throw std::logic_error(
                "Tensor::freeze: only an owning, allocated tensor can be frozen "
                "(non-owning memory cannot be protected from outside writes)");
        }
        frozen_gen_ = nextId();
    }

    /// Make the tensor mutable again. Any device copy cached for the previous
    /// freeze epoch is invalidated (re-freezing yields a new generation).
    void unfreeze() { frozen_gen_ = 0; }

    // ------------------------------------------------------------------
    // In-place initialisation
    // ------------------------------------------------------------------

    /**
     * @brief Zero-fill the tensor buffer (CPU: memset; CUDA: cudaMemset).
     */
    void zero() {
        if (frozen_gen_) throwFrozen("zero()");
        if (device_ == Device::CPU) {
            std::memset(data_, 0, numBytes());
        } else {
            cudaMemsetImpl(data_, 0, numBytes());
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
        if (frozen_gen_) throwFrozen("fill()");
        if (dtype_ != DataType::Float32) {
            throw std::runtime_error("Tensor::fill: only Float32 supported");
        }

        if (device_ == Device::CPU) {
            float* ptr = static_cast<float*>(data_);
            const size_t n = numElements();
            for (size_t i = 0; i < n; ++i) ptr[i] = value;
        } else {
            // Stage on host then copy H2D through the link-time transfer provider.
            const size_t nb = numBytes();
            std::unique_ptr<float[]> host(new float[numElements()]);
            for (size_t i = 0; i < numElements(); ++i) host[i] = value;
            cudaTransferImpl(data_, host.get(), nb, Device::CPU, Device::CUDA);
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

        if (dst.frozen_gen_) throwFrozen("copyTo() destination");

        const size_t nb = numBytes();

        // Both CPU
        if (device_ == Device::CPU && dst.device_ == Device::CPU) {
            std::memcpy(dst.data_, data_, nb);
            recordTransfer(Device::CPU, Device::CPU, nb);
            return;
        }

        // At least one side is CUDA. The actual device memcpy is dispatched
        // out-of-line (cudaTransferImpl) so this inline body does NOT depend on
        // FORGERT_CUDA_AVAILABLE. That keeps copyTo byte-identical across every
        // translation unit, which is what lets ScheduledGraph -- compiled into
        // the CPU-only `forgert` core library -- perform real H2D/D2H transfers
        // whenever `forgert_cuda` is linked, with no per-TU ODR divergence.
        //
        // Two definitions of cudaTransferImpl exist and are selected at LINK
        // time: a real one in forgert_cuda (calls forgert_cuda_memcpy) and a
        // throwing fallback in forgert_cpu_stub. src/tensor/tensor_cuda_stub.cpp
        // documents the contract.
        cudaTransferImpl(dst.data_, data_, nb, device_, dst.device_);

        // Track transfer (Phase 4). Only reached on success (impl throws on error).
        recordTransfer(device_, dst.device_, nb);
    }

private:
    TensorShape shape_;
    DataType    dtype_;
    Device      device_;
    void*       data_        = nullptr;
    bool        owns_memory_ = false;
    uint64_t    id_          = 0;
    uint64_t    frozen_gen_  = 0;   // 0 = mutable; otherwise the freeze epoch
    void cleanup() noexcept {
        if (!owns_memory_ || !data_) return;
        
        // Track deallocation (Phase 4)
        const size_t nb = numBytes();
        
        if (device_ == Device::CPU) {
            std::free(data_);
            recordDeallocation(Device::CPU, nb);
        } else {
            cudaFreeImpl(data_);  // noexcept; ignores errors (dtor must not throw)
            recordDeallocation(Device::CUDA, nb);
        }
        data_        = nullptr;
        owns_memory_ = false;
    }
    
    // Phase 4: Memory tracking helpers (implemented out-of-line to avoid
    // including memory_stats.h here, which would create circular dependency)
    void recordAllocation(size_t bytes) const;
    void recordDeallocation(Device device, size_t bytes) const;
    void recordTransfer(Device src, Device dst, size_t bytes) const;

    // Out-of-line device<->host (or device<->device) memcpy for copyTo().
    // Declared here (not macro-gated) so the inline copyTo body is identical in
    // every TU. Defined twice and chosen at link time:
    //   - forgert_cuda: real implementation (forgert_cuda_memcpy); used whenever
    //     forgert_cuda is linked.
    //   - forgert core (CPU-only): throwing fallback for builds without CUDA.
    // Throws std::runtime_error on a failed/unavailable transfer; returns
    // normally on success. `src_dev`/`dst_dev` carry the Tensor device tags.
    static void cudaTransferImpl(void* dst, const void* src, size_t nbytes,
                                 Device src_dev, Device dst_dev);

    // Device allocation / free / memset, dispatched at link time exactly like
    // cudaTransferImpl (real versions in forgert_cuda, throwing/no-op fallbacks
    // in forgert_cpu_stub). Keeping these out of line makes every inline Tensor
    // body identical in every TU, so objects built into the CPU-only `forgert`
    // library (e.g. ScheduledGraph creating CUDA tensors) still allocate real
    // device memory when forgert_cuda is linked. cudaAllocImpl/cudaMemsetImpl
    // throw on failure; cudaFreeImpl never throws.
    static void cudaAllocImpl(void** ptr, size_t nbytes);
    static void cudaFreeImpl(void* ptr) noexcept;
    static void cudaMemsetImpl(void* ptr, int value, size_t nbytes);

    // Out-of-line (src/tensor/tensor.cpp): unique-id source and the frozen-write error.
    static uint64_t nextId();
    [[noreturn]] static void throwFrozen(const char* operation);
};

} // namespace forgert

#pragma once

#include "forgert/tensor/shape.h"
#include "forgert/tensor/dtype.h"
#include <memory>
#include <cstring>
#include <stdexcept>

namespace forgert {

/**
 * @brief Memory location for tensor data
 */
enum class Device {
    CPU,    // Host memory
    CUDA    // GPU device memory
};

/**
 * @brief Basic tensor abstraction for ForgeRT
 * 
 * Phase 1: CPU-only implementation
 * Phase 4: Will add CUDA device memory support
 * 
 * Design principles:
 * - Explicit memory management (no automatic GPU transfers)
 * - Clear ownership semantics
 * - Shape and stride support for flexible layouts
 */
class Tensor {
public:
    /**
     * @brief Construct an uninitialized tensor
     * @param shape Tensor shape
     * @param dtype Data type
     * @param device Memory location (CPU or CUDA)
     */
    Tensor(const TensorShape& shape, DataType dtype, Device device = Device::CPU)
        : shape_(shape), dtype_(dtype), device_(device), owns_memory_(true) {
        
        size_t num_bytes = shape.numElements() * sizeOf(dtype);
        
        if (device == Device::CPU) {
            data_ = std::malloc(num_bytes);
            if (!data_) {
                throw std::runtime_error("Tensor: CPU memory allocation failed");
            }
        } else {
            throw std::runtime_error("Tensor: CUDA device support not yet implemented (Phase 4)");
        }
    }

    /**
     * @brief Construct a tensor with external memory (non-owning)
     * @param shape Tensor shape
     * @param dtype Data type
     * @param data External data pointer
     * @param device Memory location
     */
    Tensor(const TensorShape& shape, DataType dtype, void* data, Device device = Device::CPU)
        : shape_(shape), dtype_(dtype), device_(device), data_(data), owns_memory_(false) {
    }

    // Disable copy (for now - Phase 1)
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Enable move
    Tensor(Tensor&& other) noexcept
        : shape_(std::move(other.shape_))
        , dtype_(other.dtype_)
        , device_(other.device_)
        , data_(other.data_)
        , owns_memory_(other.owns_memory_) {
        other.data_ = nullptr;
        other.owns_memory_ = false;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            cleanup();
            shape_ = std::move(other.shape_);
            dtype_ = other.dtype_;
            device_ = other.device_;
            data_ = other.data_;
            owns_memory_ = other.owns_memory_;
            other.data_ = nullptr;
            other.owns_memory_ = false;
        }
        return *this;
    }

    ~Tensor() {
        cleanup();
    }

    // Accessors
    const TensorShape& shape() const { return shape_; }
    DataType dtype() const { return dtype_; }
    Device device() const { return device_; }
    void* data() { return data_; }
    const void* data() const { return data_; }

    size_t numElements() const { return shape_.numElements(); }
    size_t numBytes() const { return numElements() * sizeOf(dtype_); }

    /**
     * @brief Zero-initialize the tensor memory
     */
    void zero() {
        if (device_ == Device::CPU) {
            std::memset(data_, 0, numBytes());
        } else {
            throw std::runtime_error("Tensor::zero: CUDA support not yet implemented");
        }
    }

    /**
     * @brief Fill tensor with a value (CPU only, float32 only for Phase 1)
     */
    void fill(float value) {
        if (device_ != Device::CPU) {
            throw std::runtime_error("Tensor::fill: CUDA support not yet implemented");
        }
        if (dtype_ != DataType::Float32) {
            throw std::runtime_error("Tensor::fill: only Float32 supported in Phase 1");
        }
        
        float* ptr = static_cast<float*>(data_);
        size_t n = numElements();
        for (size_t i = 0; i < n; ++i) {
            ptr[i] = value;
        }
    }

private:
    TensorShape shape_;
    DataType dtype_;
    Device device_;
    void* data_ = nullptr;
    bool owns_memory_ = false;

    void cleanup() {
        if (owns_memory_ && data_) {
            if (device_ == Device::CPU) {
                std::free(data_);
            }
            // Phase 4: add cudaFree for CUDA device memory
            data_ = nullptr;
        }
    }
};

} // namespace forgert

#pragma once

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <numeric>

namespace forgert {

/**
 * @brief Represents the shape and stride of a tensor
 * 
 * Shape defines the dimensions: [N, C, H, W] or [batch, features] etc.
 * Stride defines memory layout - number of elements to skip to move along each dimension.
 * 
 * Example for a 2x3 matrix stored row-major:
 *   shape = [2, 3]
 *   stride = [3, 1]  (skip 3 elements to move one row, skip 1 element to move one column)
 */
class TensorShape {
public:
    /**
     * @brief Default constructor (creates empty shape)
     * Only for use in containers that require default construction.
     * Should not be used directly.
     */
    TensorShape() : dimensions_({1}), strides_({1}) {}

    /**
     * @brief Construct a tensor shape
     * @param dims Dimensions of the tensor
     * @param row_major If true, compute row-major strides; if false, column-major
     */
    explicit TensorShape(const std::vector<size_t>& dims, bool row_major = true)
        : dimensions_(dims) {
        if (dims.empty()) {
            throw std::invalid_argument("TensorShape: dimensions cannot be empty");
        }
        
        strides_ = computeStrides(dims, row_major);
    }

    /**
     * @brief Construct a tensor shape with explicit strides
     * @param dims Dimensions of the tensor
     * @param strides Explicit stride values
     */
    TensorShape(const std::vector<size_t>& dims, const std::vector<size_t>& strides)
        : dimensions_(dims), strides_(strides) {
        if (dims.empty()) {
            throw std::invalid_argument("TensorShape: dimensions cannot be empty");
        }
        if (dims.size() != strides.size()) {
            throw std::invalid_argument("TensorShape: dimensions and strides size mismatch");
        }
    }

    // Accessors
    size_t ndim() const { return dimensions_.size(); }
    size_t dim(size_t index) const { 
        if (index >= dimensions_.size()) {
            throw std::out_of_range("TensorShape: dimension index out of range");
        }
        return dimensions_[index]; 
    }
    size_t stride(size_t index) const { 
        if (index >= strides_.size()) {
            throw std::out_of_range("TensorShape: stride index out of range");
        }
        return strides_[index]; 
    }

    const std::vector<size_t>& dimensions() const { return dimensions_; }
    const std::vector<size_t>& strides() const { return strides_; }

    /**
     * @brief Total number of elements in the tensor
     */
    size_t numElements() const {
        return std::accumulate(dimensions_.begin(), dimensions_.end(), 
                              size_t(1), std::multiplies<size_t>());
    }

    /**
     * @brief Compute flat index from multi-dimensional indices
     * @param indices Multi-dimensional indices
     * @return Flat memory index
     */
    size_t flatIndex(const std::vector<size_t>& indices) const {
        if (indices.size() != dimensions_.size()) {
            throw std::invalid_argument("TensorShape: indices size mismatch");
        }
        
        size_t flat = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= dimensions_[i]) {
                throw std::out_of_range("TensorShape: index out of range");
            }
            flat += indices[i] * strides_[i];
        }
        return flat;
    }

    /**
     * @brief String representation for debugging
     */
    std::string toString() const {
        std::string result = "Shape[";
        for (size_t i = 0; i < dimensions_.size(); ++i) {
            if (i > 0) result += ", ";
            result += std::to_string(dimensions_[i]);
        }
        result += "] Stride[";
        for (size_t i = 0; i < strides_.size(); ++i) {
            if (i > 0) result += ", ";
            result += std::to_string(strides_[i]);
        }
        result += "]";
        return result;
    }

private:
    std::vector<size_t> dimensions_;
    std::vector<size_t> strides_;

    /**
     * @brief Compute strides for given dimensions
     * @param dims Dimensions
     * @param row_major Layout order
     * @return Computed strides
     */
    static std::vector<size_t> computeStrides(const std::vector<size_t>& dims, bool row_major) {
        std::vector<size_t> strides(dims.size());
        
        if (row_major) {
            // Row-major (C-style): rightmost dimension has stride 1
            size_t stride = 1;
            for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
                strides[i] = stride;
                stride *= dims[i];
            }
        } else {
            // Column-major (Fortran-style): leftmost dimension has stride 1
            size_t stride = 1;
            for (size_t i = 0; i < dims.size(); ++i) {
                strides[i] = stride;
                stride *= dims[i];
            }
        }
        
        return strides;
    }
};

} // namespace forgert

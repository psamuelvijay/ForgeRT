#pragma once

#include <cstddef>
#include <string>

namespace forgert {

/**
 * @brief Supported data types for tensors
 * 
 * Starting with basic types needed for Phase 1.
 * More types (float16, int8, etc.) can be added later.
 */
enum class DataType {
    Float32,
    Int32,
    Unknown
};

/**
 * @brief Get the size in bytes of a data type
 */
inline size_t sizeOf(DataType dtype) {
    switch (dtype) {
        case DataType::Float32: return 4;
        case DataType::Int32:   return 4;
        case DataType::Unknown: return 0;
    }
    return 0;
}

/**
 * @brief Get string name of data type
 */
inline std::string toString(DataType dtype) {
    switch (dtype) {
        case DataType::Float32: return "float32";
        case DataType::Int32:   return "int32";
        case DataType::Unknown: return "unknown";
    }
    return "unknown";
}

} // namespace forgert

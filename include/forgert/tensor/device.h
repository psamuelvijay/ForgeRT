#pragma once

namespace forgert {

/**
 * @brief Memory location for tensor data
 */
enum class Device {
    CPU,    // Host memory allocated with std::malloc
    CUDA    // GPU device memory allocated with cudaMalloc
};

} // namespace forgert

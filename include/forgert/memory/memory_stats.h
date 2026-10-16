#pragma once

#include "forgert/tensor/device.h"
#include <cstddef>

namespace forgert {

/**
 * @brief Global memory usage and transfer statistics tracker
 * 
 * Singleton that tracks:
 * - CPU and CUDA memory allocations
 * - Host-to-device and device-to-host transfer counts and bytes
 * - Device-to-device transfer counts and bytes
 * 
 * Thread-safety: NOT thread-safe. Assumes single-threaded execution
 * for Phase 4. Add mutex if multi-threaded execution is needed later.
 * 
 * Usage:
 *   - Tensor constructor/destructor automatically records allocations
 *   - Tensor::copyTo() automatically records transfers
 *   - Benchmarks can query stats and reset between runs
 */
class MemoryStats {
public:
    /**
     * @brief Get the singleton instance
     */
    static MemoryStats& instance();

    // -------------------------------------------------------------------
    // Allocation Tracking
    // -------------------------------------------------------------------

    /**
     * @brief Record a memory allocation
     * @param device Device where memory was allocated (CPU or CUDA)
     * @param bytes Number of bytes allocated
     */
    void recordAllocation(Device device, size_t bytes);

    /**
     * @brief Record a memory deallocation
     * @param device Device where memory was deallocated
     * @param bytes Number of bytes deallocated
     */
    void recordDeallocation(Device device, size_t bytes);

    /**
     * @brief Get total allocated bytes on a device
     * @param device Device to query (CPU or CUDA)
     * @return Current allocated bytes (allocations - deallocations)
     */
    size_t getAllocatedBytes(Device device) const;

    /**
     * @brief Get peak allocated bytes on a device
     * @param device Device to query (CPU or CUDA)
     * @return Peak allocated bytes since last reset
     */
    size_t getPeakAllocatedBytes(Device device) const;

    // -------------------------------------------------------------------
    // Transfer Tracking
    // -------------------------------------------------------------------

    /**
     * @brief Record a memory transfer
     * @param src Source device
     * @param dst Destination device
     * @param bytes Number of bytes transferred
     */
    void recordTransfer(Device src, Device dst, size_t bytes);

    /**
     * @brief Get total transfer bytes between devices
     * @param src Source device
     * @param dst Destination device
     * @return Total bytes transferred
     */
    size_t getTransferBytes(Device src, Device dst) const;

    /**
     * @brief Get transfer count between devices
     * @param src Source device
     * @param dst Destination device
     * @return Number of transfers
     */
    size_t getTransferCount(Device src, Device dst) const;

    // -------------------------------------------------------------------
    // Control
    // -------------------------------------------------------------------

    /**
     * @brief Reset all statistics to zero
     * 
     * Useful for benchmarks that want clean stats per run.
     * Note: Does NOT reset actual allocated memory, only the counters.
     */
    void reset();

    /**
     * @brief Print current statistics to stdout
     */
    void print() const;

private:
    MemoryStats() = default;
    ~MemoryStats() = default;
    
    // Disable copy and move
    MemoryStats(const MemoryStats&) = delete;
    MemoryStats& operator=(const MemoryStats&) = delete;
    MemoryStats(MemoryStats&&) = delete;
    MemoryStats& operator=(MemoryStats&&) = delete;

    // Allocation counters
    size_t cpu_allocated_bytes_ = 0;
    size_t cuda_allocated_bytes_ = 0;
    size_t cpu_peak_bytes_ = 0;
    size_t cuda_peak_bytes_ = 0;

    // Transfer counters: [src][dst] -> {bytes, count}
    // Index: 0=CPU, 1=CUDA
    struct TransferStats {
        size_t bytes = 0;
        size_t count = 0;
    };
    TransferStats transfers_[2][2];  // [src][dst]

    // Helper to convert Device enum to index
    static size_t deviceIndex(Device dev) {
        return (dev == Device::CPU) ? 0 : 1;
    }
};

} // namespace forgert

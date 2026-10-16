#include "forgert/memory/memory_stats.h"
#include <algorithm>
#include <cstdio>

namespace forgert {

MemoryStats& MemoryStats::instance() {
    static MemoryStats instance;
    return instance;
}

void MemoryStats::recordAllocation(Device device, size_t bytes) {
    if (device == Device::CPU) {
        cpu_allocated_bytes_ += bytes;
        cpu_peak_bytes_ = std::max(cpu_peak_bytes_, cpu_allocated_bytes_);
    } else {
        cuda_allocated_bytes_ += bytes;
        cuda_peak_bytes_ = std::max(cuda_peak_bytes_, cuda_allocated_bytes_);
    }
}

void MemoryStats::recordDeallocation(Device device, size_t bytes) {
    if (device == Device::CPU) {
        if (cpu_allocated_bytes_ >= bytes) {
            cpu_allocated_bytes_ -= bytes;
        } else {
            // Defensive: underflow protection
            cpu_allocated_bytes_ = 0;
        }
    } else {
        if (cuda_allocated_bytes_ >= bytes) {
            cuda_allocated_bytes_ -= bytes;
        } else {
            cuda_allocated_bytes_ = 0;
        }
    }
}

size_t MemoryStats::getAllocatedBytes(Device device) const {
    return (device == Device::CPU) ? cpu_allocated_bytes_ : cuda_allocated_bytes_;
}

size_t MemoryStats::getPeakAllocatedBytes(Device device) const {
    return (device == Device::CPU) ? cpu_peak_bytes_ : cuda_peak_bytes_;
}

void MemoryStats::recordTransfer(Device src, Device dst, size_t bytes) {
    size_t src_idx = deviceIndex(src);
    size_t dst_idx = deviceIndex(dst);
    transfers_[src_idx][dst_idx].bytes += bytes;
    transfers_[src_idx][dst_idx].count += 1;
}

size_t MemoryStats::getTransferBytes(Device src, Device dst) const {
    size_t src_idx = deviceIndex(src);
    size_t dst_idx = deviceIndex(dst);
    return transfers_[src_idx][dst_idx].bytes;
}

size_t MemoryStats::getTransferCount(Device src, Device dst) const {
    size_t src_idx = deviceIndex(src);
    size_t dst_idx = deviceIndex(dst);
    return transfers_[src_idx][dst_idx].count;
}

void MemoryStats::reset() {
    cpu_allocated_bytes_ = 0;
    cuda_allocated_bytes_ = 0;
    cpu_peak_bytes_ = 0;
    cuda_peak_bytes_ = 0;
    
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            transfers_[i][j].bytes = 0;
            transfers_[i][j].count = 0;
        }
    }
}

void MemoryStats::print() const {
    printf("\n=== ForgeRT Memory Statistics ===\n");
    
    printf("\nAllocated Memory:\n");
    printf("  CPU:  %zu bytes (%.2f MB) [peak: %zu bytes (%.2f MB)]\n",
           cpu_allocated_bytes_,
           cpu_allocated_bytes_ / (1024.0 * 1024.0),
           cpu_peak_bytes_,
           cpu_peak_bytes_ / (1024.0 * 1024.0));
    printf("  CUDA: %zu bytes (%.2f MB) [peak: %zu bytes (%.2f MB)]\n",
           cuda_allocated_bytes_,
           cuda_allocated_bytes_ / (1024.0 * 1024.0),
           cuda_peak_bytes_,
           cuda_peak_bytes_ / (1024.0 * 1024.0));
    
    printf("\nMemory Transfers:\n");
    
    // CPU -> CPU
    size_t cpu_cpu_bytes = getTransferBytes(Device::CPU, Device::CPU);
    size_t cpu_cpu_count = getTransferCount(Device::CPU, Device::CPU);
    if (cpu_cpu_count > 0) {
        printf("  CPU  -> CPU:  %zu bytes (%.2f MB), %zu transfers\n",
               cpu_cpu_bytes, cpu_cpu_bytes / (1024.0 * 1024.0), cpu_cpu_count);
    }
    
    // CPU -> CUDA (H2D)
    size_t h2d_bytes = getTransferBytes(Device::CPU, Device::CUDA);
    size_t h2d_count = getTransferCount(Device::CPU, Device::CUDA);
    if (h2d_count > 0) {
        printf("  CPU  -> CUDA: %zu bytes (%.2f MB), %zu transfers [H2D]\n",
               h2d_bytes, h2d_bytes / (1024.0 * 1024.0), h2d_count);
    }
    
    // CUDA -> CPU (D2H)
    size_t d2h_bytes = getTransferBytes(Device::CUDA, Device::CPU);
    size_t d2h_count = getTransferCount(Device::CUDA, Device::CPU);
    if (d2h_count > 0) {
        printf("  CUDA -> CPU:  %zu bytes (%.2f MB), %zu transfers [D2H]\n",
               d2h_bytes, d2h_bytes / (1024.0 * 1024.0), d2h_count);
    }
    
    // CUDA -> CUDA (D2D)
    size_t d2d_bytes = getTransferBytes(Device::CUDA, Device::CUDA);
    size_t d2d_count = getTransferCount(Device::CUDA, Device::CUDA);
    if (d2d_count > 0) {
        printf("  CUDA -> CUDA: %zu bytes (%.2f MB), %zu transfers [D2D]\n",
               d2d_bytes, d2d_bytes / (1024.0 * 1024.0), d2d_count);
    }
    
    printf("\n");
}

} // namespace forgert

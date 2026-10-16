#include "forgert/tensor/tensor.h"
#include "forgert/memory/memory_stats.h"

namespace forgert {

void Tensor::recordAllocation(size_t bytes) const {
    MemoryStats::instance().recordAllocation(device_, bytes);
}

void Tensor::recordDeallocation(Device device, size_t bytes) const {
    MemoryStats::instance().recordDeallocation(device, bytes);
}

void Tensor::recordTransfer(Device src, Device dst, size_t bytes) const {
    MemoryStats::instance().recordTransfer(src, dst, bytes);
}

} // namespace forgert

#include "forgert/tensor/tensor.h"
#include "forgert/memory/memory_stats.h"

#include <atomic>
#include <stdexcept>

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

uint64_t Tensor::nextId() {
    // Shared by tensor ids and freeze generations; starts at 1 so 0 means "none".
    static std::atomic<uint64_t> counter{1};
    return counter.fetch_add(1, std::memory_order_relaxed);
}

void Tensor::throwFrozen(const char* operation) {
    throw std::logic_error(
        std::string("Tensor: ") + operation +
        " on a frozen (immutable) tensor; call unfreeze() first if the contents "
        "must change (this invalidates any cached device copy)");
}

} // namespace forgert

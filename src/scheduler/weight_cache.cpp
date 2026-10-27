#include "forgert/scheduler/weight_cache.h"

#include <cstring>

namespace forgert {

uint64_t DeviceWeightCache::hashBytes(const void* data, size_t bytes) {
    // FNV-1a over 8-byte words (then the tail). Not cryptographic: it detects
    // accidental/escaped writes, not an adversary.
    const unsigned char* p = static_cast<const unsigned char*>(data);
    uint64_t h = 1469598103934665603ULL;
    size_t i = 0;
    for (; i + 8 <= bytes; i += 8) {
        uint64_t w;
        std::memcpy(&w, p + i, 8);
        h ^= w;
        h *= 1099511628211ULL;
    }
    for (; i < bytes; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

bool DeviceWeightCache::isResident(const Tensor& t) const {
    if (!isCacheable(t)) return false;
    auto it = entries_.find(t.id());
    if (it == entries_.end()) return false;
    const Entry& e = it->second;
    return e.generation == t.freezeGeneration() &&
           e.device->dtype() == t.dtype() &&
           e.device->shape().dimensions() == t.shape().dimensions();
}

Tensor* DeviceWeightCache::acquire(const Tensor& host) {
    if (!isCacheable(host)) return nullptr;

    auto it = entries_.find(host.id());
    if (it != entries_.end()) {
        Entry& e = it->second;
        bool valid = isResident(host);
        if (valid && verify_content_ &&
            hashBytes(host.data(), host.numBytes()) != e.content_hash) {
            ++integrity_failures_;
            valid = false;
        }
        if (valid) {
            ++hits_;
            avoided_ += e.bytes;
            return e.device.get();
        }
        // Stale: the tensor was re-frozen, reshaped, or its content changed.
        resident_bytes_ -= e.bytes;
        entries_.erase(it);
        ++invalidations_;
    }

    const size_t bytes = host.numBytes();
    if (bytes > budget_bytes_ || resident_bytes_ + bytes > budget_bytes_) {
        ++budget_skips_;
        return nullptr;
    }

    // Allocation/upload failures (including "no CUDA backend linked") propagate.
    auto dev = std::make_unique<Tensor>(host.shape(), host.dtype(), Device::CUDA);
    host.copyTo(*dev);
    dev->freeze();  // inputs are only ever read; make accidental in-place writes throw

    Entry e;
    e.generation = host.freezeGeneration();
    e.bytes = bytes;
    e.content_hash = hashBytes(host.data(), bytes);
    e.device = std::move(dev);
    Tensor* out = e.device.get();
    entries_[host.id()] = std::move(e);
    resident_bytes_ += bytes;
    ++misses_;
    uploaded_ += bytes;
    return out;
}

void DeviceWeightCache::retainOnly(const std::unordered_set<uint64_t>& live_ids) {
    for (auto it = entries_.begin(); it != entries_.end();) {
        if (live_ids.count(it->first) == 0) {
            resident_bytes_ -= it->second.bytes;
            ++evictions_;
            it = entries_.erase(it);
        } else {
            ++it;
        }
    }
}

void DeviceWeightCache::clear() {
    evictions_ += entries_.size();
    entries_.clear();   // Tensor destructors free the device memory
    resident_bytes_ = 0;
}

WeightCacheStats DeviceWeightCache::stats() const {
    WeightCacheStats s;
    s.hits = hits_; s.misses = misses_;
    s.uploaded_bytes = uploaded_; s.avoided_bytes = avoided_;
    s.invalidations = invalidations_; s.integrity_failures = integrity_failures_;
    s.budget_skips = budget_skips_; s.evictions = evictions_;
    s.resident_bytes = resident_bytes_; s.entries = entries_.size();
    return s;
}

} // namespace forgert
#pragma once

/**
 * @file weight_cache.h
 * @brief Persistent device-resident copies of immutable (frozen) host tensors.
 *
 * Problem: ScheduledGraph used to upload every CPU-resident input to the GPU on
 * every execute() and free the copy on return, so model weights were re-sent on
 * every run. The graph has no notion of a constant input, so weights and
 * activations were indistinguishable.
 *
 * Contract: only tensors the caller has explicitly freeze()d are cacheable. A
 * frozen Tensor rejects every mutation path (mutable data(), zero(), fill(), being
 * a copyTo() destination), so its device copy cannot go stale behind the cache's
 * back. Entries are keyed on (Tensor::id(), Tensor::freezeGeneration()) -- never
 * on an address, because a freed buffer's address can be reused by a different
 * tensor holding different weights. unfreeze() followed by freeze() yields a new
 * generation, so a changed tensor is detected and re-uploaded.
 *
 * Residual risk: C++ cannot stop code that kept a raw pointer from before freeze()
 * (or const_casts) from writing. setVerifyContent(true) re-hashes the host data on
 * every hit and treats a mismatch as invalidation, trading an O(bytes) host pass
 * for protection against such escapes. It is off by default.
 *
 * Lifetime: the cache owns its device tensors (themselves frozen). A pointer
 * returned by acquire() is valid until the next call that mutates the cache
 * (acquire for the same tensor after invalidation, retainOnly, clear, destruction).
 * VRAM is bounded by a byte budget; a tensor that would exceed it is simply not
 * cached (acquire returns nullptr) and the caller falls back to a transient upload.
 *
 * Not thread-safe (same as ScheduledGraph).
 */

#include "forgert/tensor/tensor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace forgert {

struct WeightCacheStats {
    uint64_t hits = 0;               ///< acquire() served from a resident copy
    uint64_t misses = 0;             ///< acquire() had to upload
    uint64_t uploaded_bytes = 0;     ///< host->device bytes performed by misses
    uint64_t avoided_bytes = 0;      ///< host->device bytes NOT sent thanks to hits
    uint64_t invalidations = 0;      ///< entries dropped: freeze generation/shape changed
    uint64_t integrity_failures = 0; ///< verify mode: host content no longer matched
    uint64_t budget_skips = 0;       ///< not cached because it would exceed the budget
    uint64_t evictions = 0;          ///< entries released by retainOnly()/clear()
    size_t   resident_bytes = 0;     ///< device bytes currently held by the cache
    size_t   entries = 0;            ///< device tensors currently held
};

class DeviceWeightCache {
public:
    /// Default budget: comfortably inside a 4 GB card (512 MiB).
    static constexpr size_t kDefaultBudgetBytes = size_t(512) << 20;

    explicit DeviceWeightCache(size_t budget_bytes = kDefaultBudgetBytes)
        : budget_bytes_(budget_bytes) {}

    DeviceWeightCache(const DeviceWeightCache&) = delete;
    DeviceWeightCache& operator=(const DeviceWeightCache&) = delete;

    /// True if `t` may be cached at all: frozen and CPU-resident.
    static bool isCacheable(const Tensor& t) {
        return t.isFrozen() && t.device() == Device::CPU;
    }

    /// True if a valid resident device copy of `t` exists right now. Does not
    /// upload, hash, or touch the stats (used by the scheduler's cost model).
    bool isResident(const Tensor& t) const;

    /// True if acquire() would be able to serve `t` from the cache: it is already
    /// resident, or it is cacheable and fits in the remaining budget. Pure query.
    bool canAdmit(const Tensor& t) const {
        if (!isCacheable(t)) return false;
        if (isResident(t)) return true;
        const size_t bytes = t.numBytes();
        return bytes <= budget_bytes_ && resident_bytes_ + bytes <= budget_bytes_;
    }

    /**
     * @brief Return the device copy of a frozen CPU tensor, uploading on a miss.
     * @return device tensor (cache-owned, frozen), or nullptr if `t` is not
     *         cacheable (unfrozen / not on CPU) or would exceed the budget.
     * @throws std::runtime_error if the upload fails, or if no CUDA backend is
     *         linked (explicit error; never a silent CPU fallback).
     */
    Tensor* acquire(const Tensor& host);

    /// Release every entry whose host tensor id is not in `live_ids`.
    void retainOnly(const std::unordered_set<uint64_t>& live_ids);

    /// Release all device memory held by the cache.
    void clear();

    void setBudgetBytes(size_t bytes) { budget_bytes_ = bytes; }
    size_t budgetBytes() const { return budget_bytes_; }

    void setVerifyContent(bool on) { verify_content_ = on; }
    bool verifyContent() const { return verify_content_; }

    WeightCacheStats stats() const;

private:
    struct Entry {
        uint64_t generation = 0;
        std::unique_ptr<Tensor> device;  // frozen
        size_t bytes = 0;
        uint64_t content_hash = 0;
    };

    static uint64_t hashBytes(const void* data, size_t bytes);

    std::unordered_map<uint64_t, Entry> entries_;  // key: host Tensor::id()
    size_t budget_bytes_;
    size_t resident_bytes_ = 0;
    bool verify_content_ = false;

    uint64_t hits_ = 0, misses_ = 0, uploaded_ = 0, avoided_ = 0;
    uint64_t invalidations_ = 0, integrity_failures_ = 0, budget_skips_ = 0, evictions_ = 0;
};

} // namespace forgert
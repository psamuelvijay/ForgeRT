/**
 * @file scheduler.h
 * @brief Phase 6: Resource-aware CPU/GPU scheduler for ForgeRT.
 *
 * Overview
 * ========
 * The scheduler answers one question per operator node:
 *
 *   "Should this node run on CPU or GPU?"
 *
 * It does so with a simple, deterministic cost model:
 *
 *   cost_cpu  = ns_per_element_cpu[op] * num_elements
 *   cost_gpu  = h2d_transfer_ns(input_bytes)
 *             + ns_per_element_gpu[op] * num_elements
 *             + d2h_transfer_ns(output_bytes)
 *             + memory_pressure_penalty
 *
 *   decision  = argmin(cost_cpu, cost_gpu)
 *             subject to:
 *               - op.supportsBackend(CUDA)
 *               - available_vram >= tensor_bytes
 *
 * Cost table calibration
 * ======================
 * CPU ns/element values come from the Phase 3 benchmark
 * (benchmark_operators, 100 iterations, Release build, i5-9300H):
 *
 *   ReLU:      0.04 ns/elem  ([128,768] → 0.004 ms / 98304 elem)
 *   Add:       0.04 ns/elem  (estimated same as ReLU, element-wise)
 *   Softmax:   0.30 ns/elem  ([128,512] → 0.020 ms / 65536 elem)
 *   LayerNorm: 0.43 ns/elem  ([128,1024] → 0.056 ms / 131072 elem)
 *   MatMul:   20.3  ns/elem  ([128,768]@[768,768] → 2.0 ms / 98304 elem)
 *
 * GPU (GTX 1050, sm_61) kernel speedup estimates relative to CPU:
 * These are conservative lower bounds.  We do NOT claim measurements
 * for kernels we cannot currently run end-to-end:
 *
 *   ReLU/Add:  5×   (memory-bound, limited by PCIe)
 *   Softmax:  10×
 *   LayerNorm: 8×
 *   MatMul:   40×   (compute-bound; 16×16 tiled GEMM on GTX 1050)
 *
 * Transfer bandwidth
 * ==================
 * PCIe 3.0 x16 peak: ~16 GB/s; practical H2D+D2H combined: ~6 GB/s.
 * We use 6 GB/s = 6 bytes/ns as the effective transfer rate.
 * Transfer cost for a tensor of B bytes:
 *   transfer_ns = B / (6.0 bytes/ns)
 *
 * Memory pressure
 * ===============
 * The scheduler queries MemoryStats::getAllocatedBytes(Device::CUDA)
 * for currently live CUDA allocations.  If the projected peak would
 * exceed the VRAM budget (configurable, default 3.8 GB for GTX 1050
 * leaving 200 MB headroom), GPU is ruled out regardless of compute cost.
 *
 * Uncertainty / fallback
 * ======================
 * If the operator type is unrecognised (no entry in the cost table),
 * the scheduler falls back to CPU and records the reason as
 * "unknown-op: no cost estimate available, defaulting to CPU".
 * This is NEVER a silent fallback — the reason is always set.
 *
 * Policy / execution separation
 * ==============================
 * SchedulingPolicy is a pure function of OperatorProfile + DeviceContext.
 * It can be unit-tested without a GPU or any tensor data.
 * ScheduledGraph wires it into graph execution.
 *
 * Usage
 * =====
 * @code
 *   ScheduledGraph sg(graph);
 *   // optional overrides:
 *   sg.setVramBudgetBytes(4ULL * 1024 * 1024 * 1024);
 *   sg.setTransferBandwidthBytesPerNs(6.0);
 *
 *   auto results = sg.execute(inputs);   // CPU or GPU per node, auto-decided
 *   sg.printSchedule();                  // explain every decision
 * @endcode
 */
#pragma once

#include "forgert/graph/graph.h"
#include "forgert/memory/memory_stats.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <cstddef>

namespace forgert {

// ---------------------------------------------------------------------------
// OperatorProfile
// ---------------------------------------------------------------------------

/**
 * @brief Static cost profile for a named operator.
 *
 * Units: nanoseconds.  All costs are per output element.
 * "output element" = total elements produced by the operator.
 */
struct OperatorProfile {
    std::string op_name;

    /// Estimated CPU execution time (ns) per output element.
    double cpu_ns_per_elem = 0.0;

    /// GPU speedup factor relative to CPU (dimensionless, > 1.0 means GPU is faster).
    /// gpu_ns_per_elem = cpu_ns_per_elem / gpu_speedup.
    double gpu_speedup = 1.0;

    /// True if the operator has a known CUDA implementation in ForgeRT.
    bool cuda_supported = false;
};

// ---------------------------------------------------------------------------
// DeviceContext
// ---------------------------------------------------------------------------

/**
 * @brief Snapshot of device resource state used by the scheduler.
 *
 * Decoupled from live hardware queries so tests can inject synthetic values.
 */
struct DeviceContext {
    /// Bytes currently allocated on CUDA device (from MemoryStats).
    size_t cuda_allocated_bytes = 0;

    /// Maximum allowable CUDA allocation (scheduler will not schedule on GPU
    /// if doing so would exceed this).  Default: 3.8 GB (GTX 1050 4 GB minus
    /// 200 MB headroom).
    size_t cuda_vram_budget_bytes = static_cast<size_t>(3.8 * 1024 * 1024 * 1024);

    /// Effective PCIe transfer bandwidth: bytes per nanosecond.
    /// PCIe 3.0 x16 practical H2D+D2H: ~6 GB/s = 6 bytes/ns.
    double transfer_bandwidth_bytes_per_ns = 6.0;

    /**
     * @brief Build a live context from MemoryStats singleton.
     *
     * This reads the real current CUDA allocation.  Inject a synthetic
     * DeviceContext in unit tests to avoid GPU dependency.
     */
    static DeviceContext fromMemoryStats() {
        DeviceContext ctx;
        ctx.cuda_allocated_bytes =
            MemoryStats::instance().getAllocatedBytes(Device::CUDA);
        return ctx;
    }
};

// ---------------------------------------------------------------------------
// ScheduleDecision
// ---------------------------------------------------------------------------

/**
 * @brief The scheduler's verdict for a single graph node.
 */
struct ScheduleDecision {
    Backend backend = Backend::CPU;

    /// Human-readable explanation of the decision.
    std::string reason;

    /// Estimated CPU cost (ns).
    double estimated_cost_cpu_ns = 0.0;

    /// Estimated total GPU cost including transfers (ns).
    /// 0 if GPU was ruled out before cost computation.
    double estimated_cost_gpu_ns = 0.0;

    bool operator_name_known = true;

    std::string summary() const {
        return "[" + std::string(backend == Backend::CPU ? "CPU" : "GPU") +
               "] " + reason +
               " (cpu=" + std::to_string(static_cast<long long>(estimated_cost_cpu_ns)) +
               "ns gpu=" + std::to_string(static_cast<long long>(estimated_cost_gpu_ns)) +
               "ns)";
    }
};

// ---------------------------------------------------------------------------
// SchedulingPolicy
// ---------------------------------------------------------------------------

/**
 * @brief Pure scheduling policy: CPU vs GPU decision logic.
 *
 * This class has NO side effects and requires NO GPU.  It operates entirely
 * on cost numbers derived from OperatorProfile + DeviceContext + tensor sizes.
 * Unit-test it with synthetic inputs.
 */
class SchedulingPolicy {
public:
    /**
     * @brief Default cost table calibrated from Phase 3 benchmarks on i5-9300H
     *        and GTX 1050 (sm_61).
     */
    static std::unordered_map<std::string, OperatorProfile> defaultProfiles();

    /**
     * @brief Decide CPU or GPU for a single operator node.
     *
     * @param op_name        Operator name (as returned by Operator::name()).
     * @param num_output_elems  Total output elements produced.
     * @param input_bytes    Total bytes of all input tensors (for H2D cost).
     * @param output_bytes   Total bytes of all output tensors (for D2H cost).
     * @param inputs_already_on_gpu  True if ALL inputs are already on CUDA device
     *                               (skip H2D transfer cost for those inputs).
     * @param ctx            Device resource snapshot.
     * @param profiles       Cost table (use defaultProfiles() for production).
     * @return ScheduleDecision with backend and explanation.
     */
    static ScheduleDecision decide(
        const std::string& op_name,
        size_t num_output_elems,
        size_t input_bytes,
        size_t output_bytes,
        bool inputs_already_on_gpu,
        const DeviceContext& ctx,
        const std::unordered_map<std::string, OperatorProfile>& profiles);
};

// ---------------------------------------------------------------------------
// NodeSchedule
// ---------------------------------------------------------------------------

/**
 * @brief Schedule for all nodes in a graph: one decision per node.
 */
struct NodeSchedule {
    size_t node_index = 0;
    std::string node_name;
    std::string op_name;
    ScheduleDecision decision;
};

// ---------------------------------------------------------------------------
// ScheduledGraph
// ---------------------------------------------------------------------------

/**
 * @brief Wraps a Graph and adds per-node CPU/GPU scheduling.
 *
 * ScheduledGraph does NOT modify the underlying Graph.  It computes a
 * schedule at execute() time (or on demand via computeSchedule()), then
 * executes each node on its assigned backend.
 *
 * Inputs are assumed to be CPU tensors (as in the standard Graph API).
 * The scheduler handles H2D/D2H transfers transparently when a GPU node
 * follows a CPU node or vice versa.  Transfers are made explicit and
 * tracked by MemoryStats automatically.
 *
 * Output tensors are returned on CPU (D2H for any GPU-executed outputs).
 */
class ScheduledGraph {
public:
    explicit ScheduledGraph(
        const Graph& graph,
        std::unordered_map<std::string, OperatorProfile> profiles =
            SchedulingPolicy::defaultProfiles());

    // Configuration
    void setVramBudgetBytes(size_t bytes);
    void setTransferBandwidthBytesPerNs(double bw);

    /**
     * @brief Compute scheduling decisions without executing.
     *
     * Useful for testing the policy independently.
     * Assumes all inputs are on CPU.
     *
     * @return One NodeSchedule per graph node, in topological order.
     */
    std::vector<NodeSchedule> computeSchedule() const;

    /**
     * @brief Execute the graph with automatic per-node CPU/GPU placement.
     *
     * Inputs must be CPU tensors.  Outputs are returned as CPU tensors.
     * Transfers between nodes on different devices are handled automatically.
     *
     * @param inputs  Input tensors (CPU, in addInput order).
     * @return Output tensors (CPU, in markOutput order).
     */
    std::vector<std::unique_ptr<Tensor>> execute(
        const std::vector<const Tensor*>& inputs);

    /**
     * @brief Print the schedule computed for the last execute() call.
     *
     * Shows op name, assigned backend, estimated costs, and reason.
     */
    void printSchedule() const;

    const std::vector<NodeSchedule>& lastSchedule() const { return last_schedule_; }

private:
    const Graph& graph_;
    std::unordered_map<std::string, OperatorProfile> profiles_;
    size_t vram_budget_bytes_ = static_cast<size_t>(3.8 * 1024 * 1024 * 1024);
    double transfer_bw_bytes_per_ns_ = 6.0;
    std::vector<NodeSchedule> last_schedule_;

    /// Transfer a tensor to the target device, returning the same pointer if
    /// it is already on the correct device, or a new tensor if a copy is needed.
    static std::unique_ptr<Tensor> ensureDevice(
        const Tensor* src,
        Device target,
        std::vector<std::unique_ptr<Tensor>>& owned);
};

} // namespace forgert

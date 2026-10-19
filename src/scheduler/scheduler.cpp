/**
 * @file scheduler.cpp
 * @brief Phase 6: Resource-aware CPU/GPU scheduler — implementation.
 */

#include "forgert/scheduler/scheduler.h"
#include "forgert/tensor/tensor.h"

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cstring>

namespace forgert {

// ---------------------------------------------------------------------------
// SchedulingPolicy
// ---------------------------------------------------------------------------

std::unordered_map<std::string, OperatorProfile>
SchedulingPolicy::defaultProfiles() {
    // ns/elem measured on i5-9300H, Release build, 100 iterations.
    // (benchmark_operators.exe, Phase 3 — second run after cache warm-up)
    //
    //  ReLU:      avg of 0.159 and 0.178 ns/elem → 0.18
    //  Add:       element-wise like ReLU          → 0.18
    //  Softmax:   avg of 5.33  and 6.09  ns/elem → 6.0
    //  LayerNorm: avg of 2.33  and 2.38  ns/elem → 2.4
    //  MatMul:    large case dominates; use 600   → 600
    //             (range: 117 – 1114 ns/elem; conservative mid-high value)
    //
    // GPU speedup factors are conservative estimates for GTX 1050 (sm_61).
    // They are NOT measured (CUDA end-to-end test is blocked by LNK4098).
    // They are labeled as estimates throughout; do not claim measurements.
    // Measured: benchmarks/benchmark_cuda.cu on GTX 1050 sm_61
    // i5-9300H, CUDA 12.6, MSVC 14.39, Release build
    // cpu_ns_per_elem: benchmark CPU column at largest size
    //   ReLU  1M: 0.37 ns/elem  -> 0.40
    //   Add   1M: 0.58 ns/elem  -> 0.60
    //   Softmax [512,1024]: 6.1 ns/elem  -> 6.0  (unchanged)
    //   LayerNorm [512,768]: 2.4 ns/elem -> 2.4  (unchanged)
    //   MatMul [512x512]: 846 ns/elem    -> 850
    // gpu_speedup: kernel-only cpu/kernel ratio (H2D+D2H modelled separately)
    //   ReLU  1M:      3.2x measured -> 3.0
    //   Add   1M:      4.1x measured -> 4.0
    //   Softmax:      10.4x measured -> 10.0  (unchanged)
    //   LayerNorm:     3.8x measured -> 4.0   (was 8x estimate)
    //   MatMul [512x512]: 127x measured -> 120 (was 40x estimate)
    return {
        {"ReLU",      {"ReLU",      0.40,   3.0, true}},
        {"Add",       {"Add",       0.60,   4.0, true}},
        {"Softmax",   {"Softmax",   6.0,   10.0, true}},
        {"LayerNorm", {"LayerNorm", 2.4,    4.0, true}},
        {"MatMul",    {"MatMul",  850.0,  120.0, true}},
        // Conv2D: measured GTX 1050 sm_61, benchmarks/benchmark_conv2d.cu
        // CPU 400 ns/elem (conservative mid; range 55-3200 by layer).
        // GPU speedup 100x (conservative floor; measured 74-328x).
        {"Conv2D",    {"Conv2D",  400.0,  100.0, true}},
    };
}

ScheduleDecision SchedulingPolicy::decide(
    const std::string& op_name,
    size_t num_output_elems,
    size_t input_bytes,
    size_t output_bytes,
    bool   inputs_already_on_gpu,
    const DeviceContext& ctx,
    const std::unordered_map<std::string, OperatorProfile>& profiles)
{
    ScheduleDecision decision;

    // ------------------------------------------------------------------
    // Step 1: Look up the operator profile.
    // ------------------------------------------------------------------
    auto it = profiles.find(op_name);
    if (it == profiles.end()) {
        decision.backend              = Backend::CPU;
        decision.operator_name_known  = false;
        decision.reason = "unknown-op: no cost estimate available for \"" +
                          op_name + "\", defaulting to CPU";
        return decision;
    }
    const OperatorProfile& profile = it->second;
    decision.operator_name_known = true;

    // ------------------------------------------------------------------
    // Step 2: Check whether the operator has a CUDA implementation.
    // ------------------------------------------------------------------
    if (!profile.cuda_supported) {
        decision.backend = Backend::CPU;
        decision.reason  = "no-cuda-impl: operator \"" + op_name +
                           "\" has no CUDA implementation";
        return decision;
    }

    // ------------------------------------------------------------------
    // Step 3: Estimate costs.
    // ------------------------------------------------------------------
    const double bw = ctx.transfer_bandwidth_bytes_per_ns;  // bytes / ns

    // CPU cost: pure compute, no transfers.
    const double cost_cpu_ns =
        profile.cpu_ns_per_elem * static_cast<double>(num_output_elems);

    // GPU cost breakdown:
    //   H2D:    only if inputs are NOT already resident on the GPU device.
    //   launch: fixed overhead per kernel dispatch, measured on GTX 1050 sm_61.
    //           benchmarks/measure_launch_overhead.cu: null_kernel (256 threads)
    //           = 20.2 us mean over 1000 iterations. Rounded to 20000 ns.
    //           This constant dominates for small tensors and is the main reason
    //           GPU is not beneficial below ~76K elements for ReLU with residency.
    //   kernel: cpu_ns_per_elem / gpu_speedup * elements (compute portion only).
    //   D2H:    always needed to return results to CPU.
    const double h2d_ns    = inputs_already_on_gpu
                               ? 0.0
                               : static_cast<double>(input_bytes) / bw;
    constexpr double CUDA_LAUNCH_OVERHEAD_NS = 20000.0;  // measured: GTX 1050 sm_61
    const double kernel_ns = CUDA_LAUNCH_OVERHEAD_NS
                             + (profile.cpu_ns_per_elem / profile.gpu_speedup)
                               * static_cast<double>(num_output_elems);
    const double d2h_ns    = static_cast<double>(output_bytes) / bw;
    const double cost_gpu_ns = h2d_ns + kernel_ns + d2h_ns;

    decision.estimated_cost_cpu_ns = cost_cpu_ns;
    decision.estimated_cost_gpu_ns = cost_gpu_ns;

    // ------------------------------------------------------------------
    // Step 4: VRAM check — prevent OOM on the GPU.
    // ------------------------------------------------------------------
    // Estimate peak additional allocation: output + un-transferred inputs.
    const size_t new_alloc_bytes =
        output_bytes +
        (inputs_already_on_gpu ? 0 : input_bytes);

    if (ctx.cuda_allocated_bytes + new_alloc_bytes > ctx.cuda_vram_budget_bytes) {
        decision.backend = Backend::CPU;
        decision.reason  = "vram-limit: projected peak (" +
                           std::to_string((ctx.cuda_allocated_bytes + new_alloc_bytes) / (1024 * 1024)) +
                           " MiB) exceeds budget (" +
                           std::to_string(ctx.cuda_vram_budget_bytes / (1024 * 1024)) +
                           " MiB), falling back to CPU";
        return decision;
    }

    // ------------------------------------------------------------------
    // Step 5: Cost comparison.
    // ------------------------------------------------------------------
    if (cost_gpu_ns < cost_cpu_ns) {
        decision.backend = Backend::CUDA;
        decision.reason  = "gpu-faster: estimated GPU cost (" +
                           std::to_string(static_cast<long long>(cost_gpu_ns)) +
                           " ns) < CPU cost (" +
                           std::to_string(static_cast<long long>(cost_cpu_ns)) +
                           " ns); h2d=" +
                           std::to_string(static_cast<long long>(h2d_ns)) +
                           " ns kernel=" +
                           std::to_string(static_cast<long long>(kernel_ns)) +
                           " ns d2h=" +
                           std::to_string(static_cast<long long>(d2h_ns)) + " ns";
    } else {
        decision.backend = Backend::CPU;
        decision.reason  = "cpu-faster: estimated CPU cost (" +
                           std::to_string(static_cast<long long>(cost_cpu_ns)) +
                           " ns) <= GPU cost (" +
                           std::to_string(static_cast<long long>(cost_gpu_ns)) +
                           " ns); transfer overhead dominates";
    }

    return decision;
}

// ---------------------------------------------------------------------------
// ScheduledGraph — helpers
// ---------------------------------------------------------------------------

// Transfer `src` to `target` device.
// If src is already on target, returns nullptr (caller uses src directly).
// Otherwise allocates a new tensor, copies, and appends to `owned`.
// Returns raw pointer to the tensor on `target` (either src itself or the copy).
static Tensor* ensureDeviceImpl(
    const Tensor* src,
    Device target,
    std::vector<std::unique_ptr<Tensor>>& owned)
{
    if (src->device() == target) {
        return const_cast<Tensor*>(src);
    }
    // Allocate on target device and copy.
    auto dst = std::make_unique<Tensor>(src->shape(), src->dtype(), target);
    src->copyTo(*dst);
    Tensor* ptr = dst.get();
    owned.push_back(std::move(dst));
    return ptr;
}

// ---------------------------------------------------------------------------
// ScheduledGraph — public API
// ---------------------------------------------------------------------------

ScheduledGraph::ScheduledGraph(
    const Graph& graph,
    std::unordered_map<std::string, OperatorProfile> profiles)
    : graph_(graph)
    , profiles_(std::move(profiles))
{}

void ScheduledGraph::setVramBudgetBytes(size_t bytes) {
    vram_budget_bytes_ = bytes;
}

void ScheduledGraph::setTransferBandwidthBytesPerNs(double bw) {
    transfer_bw_bytes_per_ns_ = bw;
}

std::vector<NodeSchedule> ScheduledGraph::computeSchedule() const {
    // Build a device context: read live CUDA allocation from MemoryStats,
    // then apply our configured budget and bandwidth.
    DeviceContext ctx = DeviceContext::fromMemoryStats();
    ctx.cuda_vram_budget_bytes          = vram_budget_bytes_;
    ctx.transfer_bandwidth_bytes_per_ns = transfer_bw_bytes_per_ns_;

    const auto& nodes      = graph_.getNodes();
    const auto  exec_order = graph_.getExecutionOrder();

    std::vector<NodeSchedule> schedule;
    schedule.reserve(exec_order.size());

    for (size_t i = 0; i < exec_order.size(); ++i) {
        size_t node_idx = exec_order[i];
        const auto& node = nodes[node_idx];

        // Compute total output bytes for this node.
        size_t output_bytes    = 0;
        size_t num_output_elems = 0;
        for (size_t oid : node->output_ids) {
            TensorShape sh = graph_.getValueShape(oid);
            DataType    dt = graph_.getValueDtype(oid);
            size_t elems   = sh.numElements();
            num_output_elems += elems;
            output_bytes += elems * sizeOf(dt);
        }

        // Compute total input bytes.
        size_t input_bytes = 0;
        for (size_t iid : node->input_ids) {
            TensorShape sh = graph_.getValueShape(iid);
            DataType    dt = graph_.getValueDtype(iid);
            input_bytes += sh.numElements() * sizeOf(dt);
        }

        // For `computeSchedule()`, assume all inputs are on CPU (no prior GPU node).
        const bool inputs_already_on_gpu = false;

        ScheduleDecision decision = SchedulingPolicy::decide(
            node->op->name(),
            num_output_elems,
            input_bytes,
            output_bytes,
            inputs_already_on_gpu,
            ctx,
            profiles_);

        // Update the tracked allocated bytes for downstream nodes.
        // (Only do this if GPU was chosen — the output will live on CUDA briefly.)
        if (decision.backend == Backend::CUDA) {
            ctx.cuda_allocated_bytes += output_bytes;
        }

        NodeSchedule ns;
        ns.node_index = node_idx;
        ns.node_name  = node->name;
        ns.op_name    = node->op->name();
        ns.decision   = decision;
        schedule.push_back(std::move(ns));
    }

    return schedule;
}

std::vector<std::unique_ptr<Tensor>> ScheduledGraph::execute(
    const std::vector<const Tensor*>& inputs)
{
    // Validate input count.
    if (inputs.size() != graph_.numInputs()) {
        throw std::invalid_argument(
            "ScheduledGraph::execute: expected " +
            std::to_string(graph_.numInputs()) +
            " inputs, got " + std::to_string(inputs.size()));
    }

    // Validate input shapes.
    const auto& input_ids = graph_.getInputIds();
    for (size_t i = 0; i < inputs.size(); ++i) {
        size_t vid = input_ids[i];
        if (inputs[i]->shape().numElements() !=
            graph_.getValueShape(vid).numElements()) {
            throw std::invalid_argument(
                "ScheduledGraph::execute: input " + std::to_string(i) +
                " shape mismatch");
        }
    }

    // Build a fresh device context for this execution.
    DeviceContext ctx = DeviceContext::fromMemoryStats();
    ctx.cuda_vram_budget_bytes          = vram_budget_bytes_;
    ctx.transfer_bandwidth_bytes_per_ns = transfer_bw_bytes_per_ns_;

    const auto& nodes      = graph_.getNodes();
    const auto  exec_order = graph_.getExecutionOrder();

    // value_storage maps value_id → raw Tensor* (never null).
    std::unordered_map<size_t, Tensor*> value_storage;

    // All heap-allocated tensors (intermediates, transferred copies, outputs).
    std::vector<std::unique_ptr<Tensor>> owned_tensors;

    // Map graph inputs.
    for (size_t i = 0; i < inputs.size(); ++i) {
        value_storage[input_ids[i]] = const_cast<Tensor*>(inputs[i]);
    }

    last_schedule_.clear();
    last_schedule_.reserve(exec_order.size());

    // ------------ main execution loop ----------------------------------------
    for (size_t node_idx : exec_order) {
        const auto& node = nodes[node_idx];

        // --- gather inputs for this node -------------------------------------
        std::vector<const Tensor*> raw_inputs;
        for (size_t iid : node->input_ids) {
            auto it = value_storage.find(iid);
            if (it == value_storage.end()) {
                throw std::runtime_error(
                    "ScheduledGraph::execute: value " + std::to_string(iid) +
                    " not yet computed");
            }
            raw_inputs.push_back(it->second);
        }

        // --- compute sizes for the scheduler ---------------------------------
        size_t output_bytes     = 0;
        size_t num_output_elems = 0;
        for (size_t oid : node->output_ids) {
            TensorShape sh = graph_.getValueShape(oid);
            DataType    dt = graph_.getValueDtype(oid);
            size_t elems   = sh.numElements();
            num_output_elems += elems;
            output_bytes += elems * sizeOf(dt);
        }

        size_t input_bytes = 0;
        for (const Tensor* t : raw_inputs) {
            input_bytes += t->numBytes();
        }

        // Check whether all inputs are already on CUDA (e.g. from a prior GPU node).
        bool inputs_on_gpu = !raw_inputs.empty();
        for (const Tensor* t : raw_inputs) {
            if (t->device() != Device::CUDA) { inputs_on_gpu = false; break; }
        }

        // --- scheduling decision ---------------------------------------------
        ScheduleDecision decision = SchedulingPolicy::decide(
            node->op->name(),
            num_output_elems,
            input_bytes,
            output_bytes,
            inputs_on_gpu,
            ctx,
            profiles_);

        // Record schedule entry.
        NodeSchedule ns;
        ns.node_index = node_idx;
        ns.node_name  = node->name;
        ns.op_name    = node->op->name();
        ns.decision   = decision;
        last_schedule_.push_back(ns);

        // --- choose target device --------------------------------------------
        Device target_device = (decision.backend == Backend::CUDA)
                                ? Device::CUDA
                                : Device::CPU;

        // --- transfer inputs if necessary ------------------------------------
        std::vector<const Tensor*> node_inputs;
        node_inputs.reserve(raw_inputs.size());
        for (const Tensor* t : raw_inputs) {
            Tensor* transferred = ensureDeviceImpl(t, target_device, owned_tensors);
            node_inputs.push_back(transferred);
        }

        // --- allocate outputs on target device --------------------------------
        std::vector<Tensor*> node_outputs;
        for (size_t oid : node->output_ids) {
            TensorShape sh = graph_.getValueShape(oid);
            DataType    dt = graph_.getValueDtype(oid);
            owned_tensors.emplace_back(
                std::make_unique<Tensor>(sh, dt, target_device));
            Tensor* out_ptr = owned_tensors.back().get();
            value_storage[oid] = out_ptr;
            node_outputs.push_back(out_ptr);
        }

        // --- execute ---------------------------------------------------------
        node->op->execute(decision.backend, node_inputs, node_outputs);

        // Update VRAM tracking for downstream decisions.
        if (decision.backend == Backend::CUDA) {
            ctx.cuda_allocated_bytes += output_bytes;
        }
    }

    // ------------ collect final outputs (always on CPU) ----------------------
    const auto& output_ids = graph_.getOutputIds();
    std::vector<std::unique_ptr<Tensor>> results;
    results.reserve(output_ids.size());

    for (size_t oid : output_ids) {
        Tensor* t = value_storage.at(oid);

        if (t->device() == Device::CPU) {
            // Find in owned_tensors and move it out.
            for (auto it = owned_tensors.begin(); it != owned_tensors.end(); ++it) {
                if (it->get() == t) {
                    results.push_back(std::move(*it));
                    owned_tensors.erase(it);
                    break;
                }
            }
        } else {
            // GPU output — transfer to CPU and return.
            auto cpu_tensor = std::make_unique<Tensor>(
                t->shape(), t->dtype(), Device::CPU);
            t->copyTo(*cpu_tensor);
            results.push_back(std::move(cpu_tensor));
        }
    }

    return results;
}

void ScheduledGraph::printSchedule() const {
    if (last_schedule_.empty()) {
        std::cout << "[ScheduledGraph] No schedule computed yet. "
                     "Call execute() or computeSchedule() first.\n";
        return;
    }
    std::cout << "[ScheduledGraph] Schedule (" << last_schedule_.size()
              << " nodes):\n";
    for (const auto& ns : last_schedule_) {
        std::cout << "  [" << ns.node_index << "] "
                  << (ns.node_name.empty() ? ns.op_name : ns.node_name)
                  << " → " << ns.decision.summary() << "\n";
    }
}

} // namespace forgert

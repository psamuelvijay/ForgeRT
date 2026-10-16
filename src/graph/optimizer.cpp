/**
 * @file optimizer.cpp
 * @brief Phase 5: GraphOptimizer implementation — dead-node elimination.
 *
 * Algorithm (dead-node elimination)
 * ==================================
 * 1. Obtain the topological execution order from the original graph
 *    (reuses the same Kahn's-algorithm implementation used by execute()).
 * 2. Seed a "live values" set with all graph output IDs.
 * 3. Walk nodes in *reverse* topological order.  A node is live if at
 *    least one of its output IDs is in the live-values set.  When a
 *    node is live, all of its input IDs are added to live-values so
 *    that the nodes producing those inputs are also retained.
 * 4. Collect live node indices in *forward* topological order.
 * 5. Rebuild a new Graph by replaying only the live nodes:
 *    a. Re-add graph inputs (preserve shape, dtype, name).
 *    b. Re-add live nodes (clone operator, remap value IDs).
 *    c. Mark original output IDs (remapped) as graph outputs.
 *
 * ID remapping
 * ============
 * The original graph has value IDs from 0..K.  After removing dead nodes
 * some IDs are unused.  Rather than carrying forward sparse IDs, we let
 * the new Graph::addInput / addNode assign fresh contiguous IDs and
 * maintain an old_id → new_id map throughout the rebuild.
 *
 * This keeps the new graph internally consistent and identical in
 * structure to the original minus the dead nodes.
 */

#include "forgert/graph/optimizer.h"

#include <stdexcept>

namespace forgert {

// ---------------------------------------------------------------------------
// GraphOptimizer::optimize
// ---------------------------------------------------------------------------

std::pair<Graph, OptimizationReport> GraphOptimizer::optimize(
    const Graph& original) const
{
    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();

    OptimizationReport report;
    report.nodes_before = original.numNodes();

    // Obtain topological order from the original graph.
    // Throws if the graph has a cycle.
    std::vector<size_t> exec_order = original.getExecutionOrder();

    // Determine which nodes are live.
    std::vector<size_t> live_idxs = computeLiveNodes(
        original.getNodes(), original.getOutputIds(), exec_order);

    // Rebuild a new graph from the live nodes only.
    Graph optimized = rebuildGraph(original, live_idxs);

    auto t1 = Clock::now();

    report.nodes_after  = optimized.numNodes();
    report.nodes_removed = report.nodes_before - report.nodes_after;
    report.elapsed_us   = std::chrono::duration_cast<
        std::chrono::microseconds>(t1 - t0).count();

    return {std::move(optimized), report};
}

// ---------------------------------------------------------------------------
// GraphOptimizer::computeLiveNodes  (static)
// ---------------------------------------------------------------------------

std::vector<size_t> GraphOptimizer::computeLiveNodes(
    const std::vector<std::unique_ptr<GraphNode>>& nodes,
    const std::vector<size_t>& output_ids,
    const std::vector<size_t>& exec_order)
{
    // Seed live-values set with the declared graph outputs.
    std::unordered_set<size_t> live_values(output_ids.begin(), output_ids.end());

    // Walk in REVERSE topological order so that by the time we inspect a
    // node, we already know whether its output is needed.
    std::unordered_set<size_t> live_node_set;

    for (int i = static_cast<int>(exec_order.size()) - 1; i >= 0; --i) {
        size_t node_idx = exec_order[static_cast<size_t>(i)];
        const auto& node = nodes[node_idx];

        // A node is live if any of its output values are needed.
        bool is_live = false;
        for (size_t out_id : node->output_ids) {
            if (live_values.count(out_id)) {
                is_live = true;
                break;
            }
        }

        if (is_live) {
            live_node_set.insert(node_idx);
            // All inputs of a live node must also be live.
            for (size_t in_id : node->input_ids) {
                live_values.insert(in_id);
            }
        }
    }

    // Return live node indices in FORWARD topological order.
    std::vector<size_t> result;
    result.reserve(live_node_set.size());
    for (size_t node_idx : exec_order) {
        if (live_node_set.count(node_idx)) {
            result.push_back(node_idx);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// GraphOptimizer::rebuildGraph  (static)
// ---------------------------------------------------------------------------

Graph GraphOptimizer::rebuildGraph(
    const Graph& original,
    const std::vector<size_t>& live_node_idxs)
{
    Graph rebuilt;

    // Maps original value ID → new value ID in the rebuilt graph.
    std::unordered_map<size_t, size_t> id_map;

    // Re-register graph inputs in their original order.
    for (size_t orig_input_id : original.getInputIds()) {
        TensorShape shape = original.getValueShape(orig_input_id);
        DataType    dtype = original.getValueDtype(orig_input_id);
        std::string name  = original.getValueName(orig_input_id);

        size_t new_id = rebuilt.addInput(shape, dtype, name);
        id_map[orig_input_id] = new_id;
    }

    // Replay only the live nodes, cloning each operator.
    const auto& orig_nodes = original.getNodes();
    for (size_t node_idx : live_node_idxs) {
        const auto& node = orig_nodes[node_idx];

        // Map original input IDs to new IDs (must already be present).
        std::vector<size_t> new_inputs;
        new_inputs.reserve(node->input_ids.size());
        for (size_t orig_in : node->input_ids) {
            auto it = id_map.find(orig_in);
            if (it == id_map.end()) {
                throw std::runtime_error(
                    "GraphOptimizer::rebuildGraph: input value " +
                    std::to_string(orig_in) +
                    " not yet mapped — topological order violation");
            }
            new_inputs.push_back(it->second);
        }

        // Clone the operator and add the node.
        auto cloned_op = node->op->clone();
        auto new_output_ids = rebuilt.addNode(
            std::move(cloned_op), new_inputs, node->name);

        // Map original output IDs to new IDs.
        for (size_t k = 0; k < node->output_ids.size(); ++k) {
            id_map[node->output_ids[k]] = new_output_ids[k];
        }
    }

    // Mark graph outputs (map original IDs to new IDs).
    for (size_t orig_out : original.getOutputIds()) {
        auto it = id_map.find(orig_out);
        if (it == id_map.end()) {
            throw std::runtime_error(
                "GraphOptimizer::rebuildGraph: output value " +
                std::to_string(orig_out) +
                " not found in rebuilt graph — it may have been a"
                " graph input rather than a computed value");
        }
        rebuilt.markOutput(it->second);
    }

    return rebuilt;
}

} // namespace forgert

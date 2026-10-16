/**
 * @file optimizer.h
 * @brief Phase 5: Graph optimization passes for ForgeRT computation graphs.
 *
 * Provides a GraphOptimizer that applies a small set of well-defined
 * optimization passes to a Graph and returns a new, optimized Graph.
 * The original graph is never modified so callers can compare optimized
 * vs. unoptimized execution at will.
 *
 * Implemented passes
 * ==================
 * 1. Dead-node elimination (DNE)
 *    Removes every GraphNode whose output values cannot reach any declared
 *    graph output.  The algorithm is a single backward reachability pass
 *    starting from the output value IDs:
 *      a) Seed a live-value set with all output_ids.
 *      b) Walk nodes in reverse topological order; if any output_id of a
 *         node is in the live-value set, mark all of the node's input_ids
 *         as live too and keep the node.
 *      c) Nodes with no live output remain dead and are dropped.
 *    Complexity: O(N + E) where N = nodes, E = edges (value dependencies).
 *
 * Rejected passes (and why)
 * =========================
 * - Constant folding: the Graph has no mechanism to embed constant tensors
 *   at construction time.  All values are external inputs supplied at
 *   execute() time.  Adding a constant-tensor API would require a
 *   non-trivial redesign of addInput() and the execute() contract.
 * - Common-subexpression elimination: useful only when the same (op-type,
 *   input-ids) pair appears more than once.  The current API makes this
 *   impossible by construction — addNode() always allocates fresh output
 *   value IDs, so users cannot accidentally create structural duplicates.
 *   The pass would be a no-op on every graph that can currently be built.
 *
 * Usage
 * =====
 * @code
 *   GraphOptimizer opt;
 *   auto [optimized_graph, report] = opt.optimize(original_graph);
 *   // report.nodes_before / report.nodes_after / report.elapsed_us
 * @endcode
 *
 * The returned graph is fully independent of the original.  Both can be
 * executed and compared.
 */
#pragma once

#include "forgert/graph/graph.h"

#include <chrono>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>

namespace forgert {

/**
 * @brief Summary of a single optimization run.
 */
struct OptimizationReport {
    size_t nodes_before   = 0;  ///< Node count in the original graph
    size_t nodes_after    = 0;  ///< Node count in the optimized graph
    size_t nodes_removed  = 0;  ///< nodes_before - nodes_after
    long long elapsed_us  = 0;  ///< Wall-clock time spent optimizing (microseconds)

    /**
     * @brief Human-readable summary for logging / test output.
     */
    std::string summary() const {
        return "Optimizer: " + std::to_string(nodes_before) + " nodes -> "
             + std::to_string(nodes_after) + " nodes ("
             + std::to_string(nodes_removed) + " removed in "
             + std::to_string(elapsed_us) + " us)";
    }
};

/**
 * @brief Graph optimization pass manager (Phase 5).
 *
 * Applies optimization passes to a Graph and returns a new optimized Graph
 * together with a report.  The original Graph is left unchanged.
 *
 * The optimizer is stateless; a single instance may be reused across
 * multiple graphs.
 */
class GraphOptimizer {
public:
    GraphOptimizer() = default;

    /**
     * @brief Run all enabled passes on @p original and return the result.
     *
     * Currently applies dead-node elimination only (see file-level docs).
     *
     * @param original  Source graph.  Must be valid (validated or not — the
     *                  optimizer re-validates internally for cycle detection).
     * @return A pair {optimized_graph, report}.  The optimized graph is
     *         fully constructed and ready for validate() / execute().
     * @throws std::runtime_error if the graph has a cycle (propagated from
     *         the internal topological sort).
     */
    std::pair<Graph, OptimizationReport> optimize(const Graph& original) const;

private:
    /**
     * @brief Dead-node elimination pass.
     *
     * Performs a backward reachability analysis from the graph's declared
     * outputs and returns the set of *node indices* (into the original
     * node list) that are reachable and must be kept.
     *
     * @param nodes      Ordered list of graph nodes (as stored in Graph).
     * @param output_ids Graph-level output value IDs.
     * @param exec_order Topologically sorted node indices (produced by the
     *                   same algorithm as Graph::computeExecutionOrder).
     * @return Ordered vector of live node indices, preserving topological
     *         order.
     */
    static std::vector<size_t> computeLiveNodes(
        const std::vector<std::unique_ptr<GraphNode>>& nodes,
        const std::vector<size_t>& output_ids,
        const std::vector<size_t>& exec_order);

    /**
     * @brief Rebuild a new Graph containing only the live nodes.
     *
     * Copies operators (via clone), preserves shapes, dtypes, input IDs,
     * and output IDs.  Value IDs are remapped so the new graph has a clean
     * contiguous ID space, but the logical structure is identical.
     *
     * @param original       Source graph (read-only).
     * @param live_node_idxs Ordered live node indices from computeLiveNodes.
     * @return A fully-constructed optimized Graph.
     */
    static Graph rebuildGraph(
        const Graph& original,
        const std::vector<size_t>& live_node_idxs);
};

} // namespace forgert

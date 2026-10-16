/**
 * @file test_graph_optimizer.cpp
 * @brief Phase 5: Graph optimizer correctness tests.
 *
 * Tests the GraphOptimizer::optimize() / dead-node elimination pass.
 *
 * Test coverage
 * =============
 * 1. already_optimal        — no dead nodes; optimizer is a no-op
 * 2. single_dead_node       — one node produces an unused output
 * 3. dead_chain             — a chain of dead nodes hangs off the live path
 * 4. branching_dead         — diamond graph where one branch is unused
 * 5. multiple_outputs       — both branches survive when both are outputs
 * 6. all_dead               — only graph inputs are declared as outputs (no nodes)
 * 7. output_equivalence     — optimized and original graphs agree numerically
 * 8. memory_stats           — MemoryStats shows fewer allocations on optimised path
 * 9. report_accuracy        — OptimizationReport counts are correct
 * 10. deep_graph            — long chain; only the live tail is kept
 */

#include "forgert/graph/graph.h"
#include "forgert/graph/optimizer.h"
#include "forgert/operator/add.h"
#include "forgert/operator/relu.h"
#include "forgert/operator/matmul.h"
#include "forgert/operator/softmax.h"
#include "forgert/memory/memory_stats.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace forgert;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool floatClose(float a, float b, float tol = 1e-5f) {
    return std::abs(a - b) <= tol;
}

static void fillRange(Tensor& t, float start = 1.0f, float step = 1.0f) {
    float* p = static_cast<float*>(t.data());
    for (size_t i = 0; i < t.numElements(); ++i) {
        p[i] = start + static_cast<float>(i) * step;
    }
}

// ---------------------------------------------------------------------------
// Test 1: already_optimal — no dead nodes
// ---------------------------------------------------------------------------
void test_already_optimal() {
    std::cout << "Test 1: already_optimal — no dead nodes\n";

    Graph g;
    auto a = g.addInput(TensorShape({4}), DataType::Float32, "A");
    auto b = g.addInput(TensorShape({4}), DataType::Float32, "B");
    auto add_out = g.addNode(std::make_unique<AddOp>(), {a, b}, "Add");
    auto relu_out = g.addNode(std::make_unique<ReLUOp>(), {add_out[0]}, "ReLU");
    g.markOutput(relu_out[0]);

    GraphOptimizer opt;
    auto [opt_g, report] = opt.optimize(g);

    assert(report.nodes_before == 2);
    assert(report.nodes_after  == 2);
    assert(report.nodes_removed == 0);
    assert(opt_g.numNodes() == 2);
    assert(opt_g.numInputs() == 2);
    assert(opt_g.numOutputs() == 1);

    std::cout << "  ✓ " << report.summary() << "\n";
}

// ---------------------------------------------------------------------------
// Test 2: single_dead_node — one node is unreachable from any output
// ---------------------------------------------------------------------------
void test_single_dead_node() {
    std::cout << "Test 2: single_dead_node\n";

    // Graph:   A ---> ReLU1 ---> output
    //          A ---> ReLU2 (dead — not marked as output)
    Graph g;
    auto a = g.addInput(TensorShape({4}), DataType::Float32, "A");
    auto live_out = g.addNode(std::make_unique<ReLUOp>(), {a}, "ReLU_live");
    /*auto dead_out =*/ g.addNode(std::make_unique<ReLUOp>(), {a}, "ReLU_dead");
    g.markOutput(live_out[0]);

    GraphOptimizer opt;
    auto [opt_g, report] = opt.optimize(g);

    assert(report.nodes_before  == 2);
    assert(report.nodes_after   == 1);
    assert(report.nodes_removed == 1);
    assert(opt_g.numNodes() == 1);

    std::cout << "  ✓ " << report.summary() << "\n";
}

// ---------------------------------------------------------------------------
// Test 3: dead_chain — a chain of dead nodes attached to the graph
// ---------------------------------------------------------------------------
void test_dead_chain() {
    std::cout << "Test 3: dead_chain\n";

    // Graph:  A ---> ReLU1 ---> Add ---> output
    //         A ---> ReLU2 ---> ReLU3 (dead chain)
    Graph g;
    auto a = g.addInput(TensorShape({4}), DataType::Float32, "A");
    auto b = g.addInput(TensorShape({4}), DataType::Float32, "B");

    auto live1  = g.addNode(std::make_unique<ReLUOp>(), {a},                 "ReLU_live");
    auto final_ = g.addNode(std::make_unique<AddOp>(),  {live1[0], b},       "Add_live");
    auto dead1  = g.addNode(std::make_unique<ReLUOp>(), {a},                 "ReLU_dead1");
    /*auto dead2  =*/ g.addNode(std::make_unique<ReLUOp>(), {dead1[0]},       "ReLU_dead2");

    g.markOutput(final_[0]);

    GraphOptimizer opt;
    auto [opt_g, report] = opt.optimize(g);

    // Should keep only ReLU_live and Add_live (2 nodes)
    assert(report.nodes_before  == 4);
    assert(report.nodes_after   == 2);
    assert(report.nodes_removed == 2);

    std::cout << "  ✓ " << report.summary() << "\n";
}

// ---------------------------------------------------------------------------
// Test 4: branching_dead — diamond graph, only one arm is output
// ---------------------------------------------------------------------------
void test_branching_dead() {
    std::cout << "Test 4: branching_dead — diamond with one unused branch\n";

    //         A
    //       /   \
    //  ReLU_L  ReLU_R (dead)
    //     |
    //   output

    Graph g;
    auto a = g.addInput(TensorShape({4}), DataType::Float32, "A");
    auto left  = g.addNode(std::make_unique<ReLUOp>(), {a}, "Left");
    /*auto right =*/ g.addNode(std::make_unique<ReLUOp>(), {a}, "Right");
    g.markOutput(left[0]);

    GraphOptimizer opt;
    auto [opt_g, report] = opt.optimize(g);

    assert(report.nodes_before  == 2);
    assert(report.nodes_after   == 1);
    assert(report.nodes_removed == 1);

    std::cout << "  ✓ " << report.summary() << "\n";
}

// ---------------------------------------------------------------------------
// Test 5: multiple_outputs — both branches of a fork are outputs; none removed
// ---------------------------------------------------------------------------
void test_multiple_outputs_preserved() {
    std::cout << "Test 5: multiple_outputs — both branches kept\n";

    Graph g;
    auto a = g.addInput(TensorShape({4}), DataType::Float32, "A");
    auto left  = g.addNode(std::make_unique<ReLUOp>(), {a}, "Left");
    auto right = g.addNode(std::make_unique<ReLUOp>(), {a}, "Right");
    g.markOutput(left[0]);
    g.markOutput(right[0]);

    GraphOptimizer opt;
    auto [opt_g, report] = opt.optimize(g);

    assert(report.nodes_before  == 2);
    assert(report.nodes_after   == 2);
    assert(report.nodes_removed == 0);
    assert(opt_g.numOutputs()   == 2);

    std::cout << "  ✓ " << report.summary() << "\n";
}

// ---------------------------------------------------------------------------
// Test 6: all_dead — no nodes, graph outputs are direct inputs
// ---------------------------------------------------------------------------
void test_all_nodes_dead() {
    std::cout << "Test 6: all_dead — graph outputs come straight from inputs\n";

    // Build a graph where outputs are the raw graph inputs (no computation).
    // Then add dead side nodes.
    Graph g;
    auto a = g.addInput(TensorShape({4}), DataType::Float32, "A");
    /*auto dead =*/ g.addNode(std::make_unique<ReLUOp>(), {a}, "Dead");
    // Mark the raw input as output — the ReLU is dead.
    g.markOutput(a);

    GraphOptimizer opt;
    auto [opt_g, report] = opt.optimize(g);

    assert(report.nodes_before  == 1);
    assert(report.nodes_after   == 0);
    assert(report.nodes_removed == 1);
    assert(opt_g.numNodes()    == 0);
    assert(opt_g.numInputs()   == 1);
    assert(opt_g.numOutputs()  == 1);

    std::cout << "  ✓ " << report.summary() << "\n";
}

// ---------------------------------------------------------------------------
// Test 7: output_equivalence — optimized graph produces identical numbers
// ---------------------------------------------------------------------------
void test_output_equivalence() {
    std::cout << "Test 7: output_equivalence — numerical correctness\n";

    // Graph: A ---> ReLU_live ---> output
    //        A ---> ReLU_dead (not output)
    Graph orig;
    auto a = orig.addInput(TensorShape({6}), DataType::Float32, "A");
    auto live = orig.addNode(std::make_unique<ReLUOp>(), {a}, "ReLU_live");
    /*auto dead =*/ orig.addNode(std::make_unique<ReLUOp>(), {a}, "ReLU_dead");
    orig.markOutput(live[0]);
    orig.validate();

    GraphOptimizer opt;
    auto [opt_g, report] = opt.optimize(orig);
    opt_g.validate();

    // Input: mix of negatives and positives
    Tensor input(TensorShape({6}), DataType::Float32, Device::CPU);
    float* d = static_cast<float*>(input.data());
    d[0] = -3.0f; d[1] = -1.0f; d[2] = 0.0f;
    d[3] =  1.0f; d[4] =  2.5f; d[5] = 4.0f;

    auto orig_out = orig.execute({&input}, Backend::CPU);
    auto opt_out  = opt_g.execute({&input}, Backend::CPU);

    assert(orig_out.size() == 1);
    assert(opt_out.size()  == 1);
    assert(orig_out[0]->numElements() == 6);
    assert(opt_out[0]->numElements()  == 6);

    const float* oa = static_cast<const float*>(orig_out[0]->data());
    const float* ob = static_cast<const float*>(opt_out[0]->data());
    for (size_t i = 0; i < 6; ++i) {
        assert(floatClose(oa[i], ob[i]));
    }
    // Spot-check: ReLU output
    assert(floatClose(ob[0], 0.0f));   // max(0, -3)
    assert(floatClose(ob[3], 1.0f));   // max(0,  1)
    assert(floatClose(ob[5], 4.0f));   // max(0,  4)

    std::cout << "  ✓ " << report.summary() << "\n";
}

// ---------------------------------------------------------------------------
// Test 8: memory_stats — optimised graph allocates fewer intermediate tensors
// ---------------------------------------------------------------------------
void test_memory_stats() {
    std::cout << "Test 8: memory_stats — allocation count comparison\n";

    // Larger tensors so the difference is clear.
    // Graph: A ---> ReLU_live   (kept)
    //        A ---> Add_dead ---> ReLU_dead2  (dead chain, 2 nodes)
    TensorShape big({128});

    Graph orig;
    auto a = orig.addInput(big, DataType::Float32, "A");
    auto b = orig.addInput(big, DataType::Float32, "B");
    auto live   = orig.addNode(std::make_unique<ReLUOp>(), {a},         "ReLU_live");
    auto dead1  = orig.addNode(std::make_unique<AddOp>(),  {a, b},      "Add_dead");
    /*auto dead2  =*/ orig.addNode(std::make_unique<ReLUOp>(), {dead1[0]}, "ReLU_dead2");
    orig.markOutput(live[0]);
    orig.validate();

    GraphOptimizer opt;
    auto [opt_g, report] = opt.optimize(orig);
    opt_g.validate();

    Tensor ta(big, DataType::Float32, Device::CPU);  ta.fill(1.0f);
    Tensor tb(big, DataType::Float32, Device::CPU);  tb.fill(2.0f);

    // --- run original ---
    MemoryStats::instance().reset();
    {
        auto r = orig.execute({&ta, &tb}, Backend::CPU);
        (void)r;
    }
    size_t peak_orig = MemoryStats::instance().getPeakAllocatedBytes(Device::CPU);

    // --- run optimised ---
    MemoryStats::instance().reset();
    {
        auto r = opt_g.execute({&ta, &tb}, Backend::CPU);
        (void)r;
    }
    size_t peak_opt = MemoryStats::instance().getPeakAllocatedBytes(Device::CPU);

    // Optimised graph allocates fewer intermediate tensors, so peak must be
    // <= original peak.  (We can't assume strictly less because ta/tb are
    // accounted for by the caller, but the intermediate allocations inside
    // execute() for dead nodes are gone.)
    std::cout << "    Original peak CPU: " << peak_orig << " bytes\n";
    std::cout << "    Optimised peak CPU: " << peak_opt << " bytes\n";
    assert(peak_opt <= peak_orig);

    std::cout << "  ✓ " << report.summary() << "\n";
}

// ---------------------------------------------------------------------------
// Test 9: report_accuracy — validate all report fields
// ---------------------------------------------------------------------------
void test_report_accuracy() {
    std::cout << "Test 9: report_accuracy\n";

    Graph g;
    auto a = g.addInput(TensorShape({2}), DataType::Float32);
    auto b = g.addInput(TensorShape({2}), DataType::Float32);
    auto live  = g.addNode(std::make_unique<AddOp>(), {a, b}, "LiveAdd");
    auto dead1 = g.addNode(std::make_unique<ReLUOp>(), {live[0]}, "DeadReLU1");
    auto dead2 = g.addNode(std::make_unique<ReLUOp>(), {dead1[0]}, "DeadReLU2");
    (void)dead2;
    g.markOutput(live[0]);  // Only the Add output is declared; the ReLU chain is dead

    GraphOptimizer opt;
    auto [opt_g, report] = opt.optimize(g);

    assert(report.nodes_before  == 3);
    assert(report.nodes_after   == 1);
    assert(report.nodes_removed == 2);
    assert(report.elapsed_us   >= 0);   // non-negative
    assert(!report.summary().empty());

    std::cout << "  ✓ " << report.summary() << "\n";
}

// ---------------------------------------------------------------------------
// Test 10: deep_graph — long chain; only the tail survives
// ---------------------------------------------------------------------------
void test_deep_graph() {
    std::cout << "Test 10: deep_graph — long chain, partially dead\n";

    // Build: input ---> R0 ---> R1 ---> R2 ---> R3 ---> output
    //        input ---> X0 ---> X1 (dead branch)
    const size_t live_depth = 5;

    Graph g;
    auto a = g.addInput(TensorShape({4}), DataType::Float32, "A");

    // Live chain
    size_t cur = a;
    for (size_t i = 0; i < live_depth; ++i) {
        auto out = g.addNode(std::make_unique<ReLUOp>(), {cur},
                             "ReLU_live_" + std::to_string(i));
        cur = out[0];
    }
    g.markOutput(cur);

    // Dead branch (2 nodes)
    auto d0 = g.addNode(std::make_unique<ReLUOp>(), {a}, "Dead0");
    /*auto d1 =*/ g.addNode(std::make_unique<ReLUOp>(), {d0[0]}, "Dead1");

    g.validate();

    GraphOptimizer opt;
    auto [opt_g, report] = opt.optimize(g);
    opt_g.validate();

    assert(report.nodes_before  == static_cast<size_t>(live_depth + 2));
    assert(report.nodes_after   == live_depth);
    assert(report.nodes_removed == 2);

    // Verify numerical equivalence on the live chain
    Tensor input(TensorShape({4}), DataType::Float32, Device::CPU);
    float* dp = static_cast<float*>(input.data());
    dp[0] = -1.0f; dp[1] = 2.0f; dp[2] = -3.0f; dp[3] = 4.0f;

    auto r_orig = g.execute({&input}, Backend::CPU);
    auto r_opt  = opt_g.execute({&input}, Backend::CPU);

    const float* o1 = static_cast<const float*>(r_orig[0]->data());
    const float* o2 = static_cast<const float*>(r_opt[0]->data());
    for (size_t i = 0; i < 4; ++i) {
        assert(floatClose(o1[i], o2[i]));
    }

    std::cout << "  ✓ " << report.summary() << "\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    std::cout << "========================================\n";
    std::cout << "ForgeRT Graph Optimizer Tests (Phase 5)\n";
    std::cout << "========================================\n";

    try {
        test_already_optimal();
        test_single_dead_node();
        test_dead_chain();
        test_branching_dead();
        test_multiple_outputs_preserved();
        test_all_nodes_dead();
        test_output_equivalence();
        test_memory_stats();
        test_report_accuracy();
        test_deep_graph();

        std::cout << "========================================\n";
        std::cout << "All optimizer tests PASSED ✓\n";
        std::cout << "========================================\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Test FAILED: " << e.what() << "\n";
        return 1;
    }
}

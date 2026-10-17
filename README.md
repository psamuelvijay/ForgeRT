# ForgeRT

**Resource-Aware Neural Inference Runtime**

ForgeRT is an educational/research-oriented neural network inference runtime written in modern C++ and CUDA. Unlike traditional frameworks that simply offload everything to GPU, ForgeRT intelligently decides where operations should execute based on computation cost, memory pressure, and CPU↔GPU transfer overhead.

## Project Goals

The central research question: *"Given a neural network graph and limited heterogeneous hardware, can a lightweight runtime automatically make better execution decisions by considering computation cost, memory pressure, and transfer overhead?"*

### Key Features

- **Resource-Aware Scheduling**: Adaptive execution placement (CPU vs GPU) based on measured costs
- **Custom CUDA Kernels**: Hand-optimized kernels targeting Pascal architecture (sm_61)
- **Graph-Based Execution**: DAG computation graph with automatic dependency resolution
- **Operator Abstraction**: Clean separation between operations and execution backends
- **Memory Management**: Explicit lifetime tracking, pooling, and reuse strategies
- **Performance Profiling**: Detailed metrics separating kernel time, transfer time, and overhead
- **Reproducible Benchmarks**: Comprehensive comparison against established runtimes

## Execution Pipeline

ForgeRT implements a complete inference pipeline:

```
Graph Construction
    ↓
Shape Inference (validates compatibility)
    ↓
Topological Sort (determines execution order)
    ↓
Memory Allocation (intermediate tensors)
    ↓
Operator Execution (CPU/CUDA backends)
    ↓
Result Tensors
```

## Implemented Operators

### Phase 1 & 2: CPU Backend

| Operator | Description | Shape Example | Status |
|----------|-------------|---------------|--------|
| **Add** | Element-wise addition with broadcasting | `[3,4] + [4] → [3,4]` | ✅ Complete |
| **ReLU** | Rectified Linear Unit activation | `[M,N] → [M,N]` | ✅ Complete |
| **MatMul** | 2D Matrix multiplication | `[M,K] @ [K,N] → [M,N]` | ✅ Complete |
| **Softmax** | Softmax activation (last dimension) | `[Batch,Classes] → [Batch,Classes]` | ✅ Complete |
| **LayerNorm** | Layer normalization (last dimension) | `[Batch,Features] → [Batch,Features]` | ✅ Complete |

### Phase 3: CUDA Backend

| Operator | CUDA Kernel | Backend Integration | CPU vs CUDA Tests | Status |
|----------|-------------|---------------------|-------------------|--------|
| **ReLU** | ✅ Grid-stride, sm_61 optimized | ✅ `executeCUDA()` | ✅ Numerical equivalence | ✅ Complete |
| **MatMul** | ✅ 16×16 tiled shared memory | ✅ `executeCUDA()` | ✅ Numerical equivalence | ✅ Complete |
| **Softmax** | ✅ Multi-pass reduction | ✅ `executeCUDA()` | ✅ Numerical equivalence | ✅ Complete |
| **LayerNorm** | ✅ Dual-reduction (mean+var) | ✅ `executeCUDA()` | ✅ Numerical equivalence | ✅ Complete |
| **Add** | ⏳ Deferred | ⏳ Deferred | ⏳ Deferred | 🔄 Phase 4 |

**Example:**
```cpp
// Create computation graph
Graph graph;
auto input_a = graph.addInput(TensorShape({3, 4}), DataType::Float32);
auto input_b = graph.addInput(TensorShape({4}), DataType::Float32);

// Add operations
auto add_out = graph.addNode(std::make_unique<AddOp>(), {input_a, input_b});
auto relu_out = graph.addNode(std::make_unique<ReLUOp>(), {add_out[0]});

// Mark output and execute
graph.markOutput(relu_out[0]);
graph.validate();

auto results = graph.execute({&tensor_a, &tensor_b}, Backend::CPU);
```

## Design Philosophy

ForgeRT does NOT:
- Try to replace PyTorch or ONNX Runtime
- Maximize GPU utilization blindly
- Support every possible neural network operator
- Hide complexity behind abstractions

ForgeRT DOES:
- Optimize for latency, throughput, and memory efficiency
- Make execution decisions transparent and measurable
- Demonstrate that GPU isn't always faster when transfer overhead dominates
- Provide deep insight into heterogeneous execution tradeoffs

## Target Hardware

- **CPU**: Intel Core i5-9300H (4 physical cores)
- **RAM**: 32 GB
- **GPU**: NVIDIA GeForce GTX 1050 (4 GB VRAM, Pascal, compute capability 6.1)
- **Storage**: SSD (primary) + HDD (datasets)

## Technology Stack

- C++17/20
- CUDA (NVIDIA CUDA Toolkit 12.x)
- CMake
- Python (tooling, visualization, benchmarks)
- MSVC x64

## Project Structure

```
forgert/
├── include/forgert/     # Public API headers
├── src/                 # Implementation
├── cuda/                # CUDA kernels and device code
├── tests/               # Unit tests
├── benchmarks/          # Performance benchmarks
├── tools/               # Utilities and scripts
├── scripts/             # Build and automation scripts
├── examples/            # Usage examples
└── docs/                # Documentation
```

## Development Phases

### Phase 1: Foundation ✅ Complete
- Tensor abstraction, Graph representation, CMake build system, unit test harness

### Phase 2: CPU Backend ✅ Complete
- CPU implementations: Add, ReLU, MatMul, Softmax, LayerNorm, benchmark harness
- Conv2D explicitly deferred

### Phase 3: CUDA Backend 🔧 In Progress
- ✅ Build system fixed for CMake 4.x + MSVC 14.39 + Ninja + CUDA 12.6 + sm_61
- ✅ ReLU CUDA kernel (grid-stride, Pascal sm_61)
- ✅ ReLUOp CUDA backend dispatch (`Backend::CUDA` → GPU kernel)
- ⏳ CUDA kernels for remaining operators
- ⏳ Tensor CUDA memory allocation (Phase 4 prerequisite for graph-level CUDA)

### Phase 3: CUDA Backend
- Custom CUDA kernels for Pascal (sm_61)
- Explicit compute capability targeting

### Phase 4: Memory System ✅ Complete
- ✅ `Device` enum extracted to `include/forgert/tensor/device.h` — clean separation of concerns
- ✅ `MemoryStats` singleton: tracks CPU/CUDA byte allocations, peak usage, and transfer counts
- ✅ `Tensor` instrumented: constructor/destructor/copyTo() automatically record to `MemoryStats`
- ✅ CUDA memory query: `forgert_cuda_mem_info()` wraps `cudaMemGetInfo()` 
- ✅ All existing CPU operator headers fixed: `FORGERT_CUDA_AVAILABLE` guards on all `extern "C"` CUDA declarations
- ✅ `forgert` core library: `memory_stats.cpp` + `tensor.cpp` (CPU, no CUDA dependency)
- ✅ `forgert.h` convenience header added
- ✅ `test_memory_stats` — 6 tests: allocation tracking, peak tracking, transfer tracking, multi-alloc, reset, print

**Test suite: 10/10 CPU tests passing** ✅

### Phase 5: Graph Optimization ✅ Complete
- **Dead-node elimination** — backward reachability pass from declared outputs removes every operator whose result can never reach a graph output
- **Constant folding / CSE** — deliberately not implemented (see design notes in `include/forgert/graph/optimizer.h`)
- `GraphOptimizer` class: `optimize(graph)` returns `{optimized_graph, OptimizationReport}`
- `OptimizationReport`: `nodes_before`, `nodes_after`, `nodes_removed`, `elapsed_us`, `summary()`
- Operator `clone()` added to all five operators (required for graph rebuild)
- Graph introspection accessors added (`getNodes()`, `getValueShape()`, `getValueDtype()`, `getExecutionOrder()`, …)
- **Test suite: 10 focused optimizer tests** (`test_graph_optimizer`) — all passing
- **Measured memory reduction:** 3-node graph with 2 dead nodes → optimised graph allocates **3× less peak CPU memory** (1536 B → 512 B)
- **Optimizer overhead:** 2–28 µs for all test graphs (O(N+E) backward pass)

### Phase 6: Resource-Aware Scheduler
- CPU vs CUDA execution decisions
- Cost model based on profiling data

### Phase 7: Autotuning
- Kernel configuration search
- Block size and tiling optimization

### Phase 8: Model Import
- Restricted ONNX operator subset support

### Phase 9: Benchmarking
- Comparison vs PyTorch/ONNX Runtime
- Comprehensive metrics collection

### Phase 10: Visualization
- Execution decision visualization
- Performance report generation

## Building

### Windows (MSVC + CUDA 12.6)

**Quick Build:**
```batch
scripts\build.bat
```

**Build Options:**
```batch
scripts\build.bat          # Build with existing configuration
scripts\build.bat clean    # Clean and reconfigure
scripts\build.bat config   # Reconfigure only
scripts\build.bat test     # Build and run tests
```

**Manual Build:**
```batch
# Set up MSVC 14.39 environment
call "D:\Development\VSBuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.39

# Configure
cmake -S . -B build -G Ninja ^
  -DCMAKE_CXX_COMPILER="D:/Development/VSBuildTools/VC/Tools/MSVC/14.39.33519/bin/Hostx64/x64/cl.exe" ^
  -DCMAKE_CUDA_COMPILER="D:/Development/CUDA/v12.6/bin/nvcc.exe" ^
  -DCMAKE_CUDA_HOST_COMPILER="D:/Development/VSBuildTools/VC/Tools/MSVC/14.39.33519/bin/Hostx64/x64/cl.exe" ^
  -DCMAKE_CUDA_ARCHITECTURES=61

# Build
cmake --build build --parallel
```

**Requirements:**
- CUDA Toolkit 12.6 (sm_61 support - CUDA 13.x dropped Pascal)
- MSVC 14.39 (Visual Studio 2022 Build Tools)
- CMake 3.18+
- Ninja build system

## Running Tests

```batch
scripts\build.bat test
```

Or manually:
```batch
call "D:\Development\VSBuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.39
ctest --test-dir build --output-on-failure
```

## Current Status

**Phase 1: Foundation** ✅ Complete
- ✅ Tensor shape/stride representation
- ✅ CPU tensor allocation with RAII
- ✅ DataType abstraction (Float32, Int32)
- ✅ Operator abstraction framework
- ✅ Add operator with NumPy-style broadcasting
- ✅ Graph representation and execution
- ✅ DAG validation with cycle detection
- ✅ Topological execution ordering
- ✅ CPU backend execution pipeline

**Phase 2: CPU Backend** ✅ Complete
- ✅ ReLU operator
- ✅ MatMul operator
- ✅ Softmax operator (numerically stable)
- ✅ LayerNorm operator (two-pass algorithm)
- ✅ CPU benchmark harness
- ⏳ Conv2D — explicitly deferred

**Phase 3: CUDA Backend** ✅ Complete
- ✅ Build system: CMake 4.x + MSVC 14.39 + Ninja + CUDA 12.6 + sm_61 working
- ✅ ReLU CUDA kernel (`cuda/kernels/relu_kernel.cu`) — grid-stride loop, sm_61, tested at 1K and 1M elements
- ✅ `ReLUOp::executeCUDA()` — CUDA backend wired into operator dispatch (`Backend::CUDA` routes to the GPU kernel)
- ✅ `ReLUOp::supportsBackend()` reports CUDA when built with `FORGERT_CUDA_AVAILABLE`
- ✅ MatMul CUDA kernel (`cuda/kernels/matmul_kernel.cu`) — tiled multiplication with 16x16 shared memory tiles, sm_61 optimized
- ✅ `MatMulOp::executeCUDA()` — CUDA backend integration with comprehensive testing
- ✅ Softmax CUDA kernel (`cuda/kernels/softmax_kernel.cu`) — multi-pass reduction approach, numerically stable
- ✅ `SoftmaxOp::executeCUDA()` — CUDA backend integration with comprehensive testing
- ✅ LayerNorm CUDA kernel (`cuda/kernels/layernorm_kernel.cu`) — dual-reduction approach (mean + variance), block-level shared memory
- ✅ `LayerNormOp::executeCUDA()` — CUDA backend integration with comprehensive testing
- ⏳ Add CUDA kernel — deferred to Phase 4 (simple element-wise operation, lower priority)

**Test suite: 11/11 CPU tests passing** ✅ (Phase 5 state)
- 11 CPU test suites: Tensor, Operator, ReLU, MatMul, Softmax, LayerNorm, Graph ×3, MemoryStats, GraphOptimizer
- CUDA tests (Phase 3): `test_graph_cuda` and `test_relu_cuda` link and compile; `test_graph_cuda.exe` successfully built with Phase 5 changes

**CUDA configuration:** CUDA 12.6.85 · sm_61 (Pascal) · MSVC 14.39 · Ninja · CMake 4.3

**Phase 4: Memory System** ✅ Complete
- ✅ `Device` enum in `include/forgert/tensor/device.h` (clean separation)
- ✅ `MemoryStats` singleton: CPU/CUDA allocated bytes, peak bytes, transfer bytes/counts
- ✅ `Tensor` instrumented: constructor/destructor/copyTo() auto-report to stats
- ✅ `forgert_cuda_mem_info()`: queries `cudaMemGetInfo()` for available VRAM
- ✅ All operator headers fixed: CUDA `extern "C"` declarations guarded with `FORGERT_CUDA_AVAILABLE`
- ✅ `forgert` core library: CPU-only, independent of CUDA link
- ✅ `test_memory_stats`: 6 focused tests (allocation, peak, transfer, multi-alloc, reset, print)

**Phase 5: Graph Optimization** ✅ Complete
- ✅ `GraphOptimizer` class: dead-node elimination (backward reachability, O(N+E))
- ✅ `OptimizationReport`: nodes_before/after/removed, elapsed_us, summary()
- ✅ `Operator::clone()` pure virtual added; all 5 operators implement it
- ✅ Graph introspection accessors added (getNodes, getValueShape/Dtype/Name, getExecutionOrder)
- ✅ `test_graph_optimizer`: 10 tests covering no-op, single/chain/branch dead nodes, multi-output, all-dead, numerical equivalence, memory stats, report accuracy, deep graph
- ✅ **Memory**: 3-node graph with 2 dead → 3× lower peak CPU allocation (1536 → 512 bytes, measured)
- ✅ **Overhead**: 2–28 µs optimizer pass (all test graphs)
- ✅ `test_graph_cuda.exe` compiled and linked successfully with Phase 5 changes

**Phase 6: Resource-Aware Scheduler MVP** ✅ Complete

### Scheduling Policy

`SchedulingPolicy` is a **pure function** (no side effects, no GPU dependency). It answers one question per operator node: *"Should this run on CPU or GPU?"*

Cost model:
```
cost_cpu  = cpu_ns_per_elem[op] × num_output_elements

cost_gpu  = H2D_transfer_ns(input_bytes)      ← 0 if inputs already on GPU
          + kernel_ns_per_elem[op] × elements  ← cpu_ns / gpu_speedup
          + D2H_transfer_ns(output_bytes)       ← always needed

decision  = argmin(cost_cpu, cost_gpu)
          subject to: op.cuda_supported AND projected_vram ≤ budget
```

### CPU Cost Table (measured, i5-9300H, Release, 100 iterations)

| Operator   | ns/elem (measured) | GPU speedup (estimate) | CUDA supported |
|------------|-------------------|------------------------|---------------|
| ReLU       | 0.18              | 5×                     | ✅            |
| Add        | 0.18              | 5×                     | ✅            |
| Softmax    | 6.0               | 10×                    | ✅            |
| LayerNorm  | 2.4               | 8×                     | ✅            |
| MatMul     | 600               | 40×                    | ✅            |

Transfer bandwidth: 6 bytes/ns (PCIe 3.0 x16 practical H2D+D2H combined).  
VRAM budget: 3.8 GB (GTX 1050 4 GB − 200 MB headroom).  
GPU speedup factors are **conservative estimates** — CUDA end-to-end benchmarks are blocked by a pre-existing LNK4098 LIBCMT linker conflict (see below).

### Scheduler Decision Examples (default BW = 6 bytes/ns)

| Scenario | Decision | Why |
|----------|----------|-----|
| ReLU [1024] | **CPU** | GPU cost ~1400 ns > CPU cost ~184 ns (transfer dominates) |
| MatMul [512×512] | **GPU** | CPU cost ~157 ms >> GPU ~4.5 ms (compute savings dominate) |
| MatMul with inputs on GPU | **GPU** | H2D skipped → even lower GPU cost |
| Any op, 1 KB VRAM budget | **CPU** | VRAM limit prevents GPU allocation |
| Unknown op | **CPU** | `operator_name_known=false`, reason="unknown-op:..." |

### Key Design Decisions

- **Policy / execution separation**: `SchedulingPolicy::decide()` is unit-testable with synthetic `DeviceContext` — no GPU needed for scheduler tests.
- **Explicit reasons**: Every `ScheduleDecision` has a non-empty `reason` string. No silent fallbacks.
- **`ScheduledGraph` wraps `Graph`**: Does not modify `Graph::execute()` or break the existing explicit-backend API. Scheduler is opt-in.
- **Per-node device transfers**: `ScheduledGraph::execute()` handles H2D/D2H between nodes assigned to different backends transparently, using `Tensor::copyTo()`.
- **Outputs always on CPU**: Matches the standard `Graph::execute()` contract.

### Test Results

```
=== Phase 6 Scheduler Tests ===
PASS: Test 1: ReLU [1024] → CPU preferred (transfer overhead > savings)
PASS: Test 2: MatMul [512×512] → GPU preferred (compute savings > transfer)
PASS: Test 3: GPU cost with residency <= GPU cost without residency
PASS: Test 3: slow-BW + no residency → CPU preferred (H2D dominates)
PASS: Test 3: slow-BW + residency → GPU preferred (H2D skipped, kernel+D2H < CPU)
PASS: Test 4: VRAM limit → CPU fallback even for large MatMul
PASS: Test 5: cuda_supported=false → always CPU
PASS: Test 6: unknown op → CPU fallback (operator_name_known=false)
PASS: Test 7: CPU cost matches hand calculation
PASS: Test 7: GPU cost matches hand calculation (h2d+kernel+d2h)
PASS: Test 8: ScheduledGraph single-node output matches Graph::execute()
PASS: Test 9: multi-node ScheduledGraph output matches Graph::execute()
=== All Phase 6 tests passed ===
```

**Test suite: 12/12 CPU tests passing** ✅ (Phase 6 state)
- 12 CPU test suites: Tensor, Operator, ReLU, MatMul, Softmax, LayerNorm, Graph ×3, MemoryStats, GraphOptimizer, Scheduler
- `test_graph_cuda.exe` (7 CUDA tests): built and **all 7 pass** on GTX 1050

### CUDA Build Status (Phase 6)

- `test_graph_cuda.exe` — **compiles, links, 7/7 CUDA tests pass** on GTX 1050 sm_61
- Standalone CUDA kernel tests (`test_relu_cuda`, `test_matmul_cuda`, `test_softmax_cuda`, `test_layernorm_cuda`, `test_relu_op_cuda`) — link fails with pre-existing `LNK4098: LIBCMT / MSVCRT conflict` (MSVC 14.39 + CUDA 12.6 CRT mismatch, unrelated to Phase 6 changes)
- `ScheduledGraph::execute()` on GPU path requires CUDA runtime; CPU-only build correctly falls back to CPU for all nodes (VRAM budget = 0 MiB when CUDA unavailable)

## Engineering Principles

1. **Correctness before optimization**
2. **Benchmark before claiming improvements**
3. **Keep CPU and CUDA implementations independently testable**
4. **Avoid unnecessary dependencies**
5. **Never hide CUDA errors**
6. **Keep code interview-explainable**

## Non-Goals

- ❌ Full PyTorch replacement
- ❌ Support every neural network operator
- ❌ Immediate transformer/LLM support
- ❌ Artificial complexity for repository size
- ❌ Optimization without measurements

## License

MIT License - See LICENSE file for details

## Author

Educational/research project demonstrating systems programming, GPU architecture, and performance engineering.

---

*"A GPU operation is NOT automatically better than a CPU operation."*

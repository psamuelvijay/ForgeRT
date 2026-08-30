# ForgeRT Development Roadmap

## Current Status: Phase 1 - Foundation (In Progress)

---

## Phase 1: Foundation ✓ (Started)

**Goal**: Establish core abstractions and build system

### Tasks
- [x] Project structure
- [x] CMake build system configured for CUDA sm_61
- [x] Documentation framework
- [ ] Tensor shape/stride representation
- [ ] Basic tensor memory allocation (CPU)
- [ ] Tensor data type abstraction (float32, int32)
- [ ] Graph node structure
- [ ] Operator base class
- [ ] Basic unit test framework
- [ ] Simple benchmark harness

**Deliverable**: Compiles on Windows with MSVC + CUDA, basic tensor operations work on CPU

---

## Phase 2: CPU Backend

**Goal**: Implement correct CPU versions of core operators

### Operators to Implement
- [ ] MatMul (matrix multiplication)
- [ ] Add (element-wise addition)
- [ ] ReLU (activation function)
- [ ] Softmax
- [ ] LayerNorm
- [ ] Conv2D (basic 2D convolution)

### Requirements
- Correctness validated against reference implementations
- Unit tests for each operator
- Multiple input shapes tested
- Edge cases handled (empty tensors, dimension mismatches)
- Performance baseline measurements collected

**Deliverable**: All operators pass correctness tests, baseline CPU performance documented

---

## Phase 3: CUDA Backend

**Goal**: Implement custom CUDA kernels for Pascal (sm_61)

### Kernels to Implement
- [ ] Vector addition
- [ ] Vector reduction (sum, max)
- [ ] Matrix multiplication (tiled)
- [ ] ReLU
- [ ] Softmax

### Requirements
- Explicit sm_61 targeting
- Error checking after every CUDA call
- Kernel launch parameter documentation
- Shared memory usage justified
- Correctness validated against CPU implementations
- Performance profiled with Nsight Compute

**Deliverable**: CUDA kernels work on GTX 1050, performance characterized

---

## Phase 4: Memory System

**Goal**: Explicit memory management with lifetime tracking

### Components
- [ ] Device memory allocator
- [ ] Host-to-device transfer with timing
- [ ] Device-to-host transfer with timing
- [ ] Tensor lifetime analysis
- [ ] Basic memory pool
- [ ] Buffer reuse logic

### Requirements
- Transfer time separated from kernel time
- Memory leaks detected and fixed
- Peak memory usage tracked
- Clear ownership semantics

**Deliverable**: Memory transfers are explicit, measurable, and efficient

---

## Phase 5: Graph Optimization

**Goal**: Basic graph-level transformations

### Optimizations
- [ ] Constant folding
- [ ] Dead node elimination
- [ ] Operator fusion (select patterns)
  - [ ] ReLU after MatMul
  - [ ] Add after MatMul
- [ ] Memory lifetime analysis for buffer reuse

### Requirements
- Each optimization has before/after tests
- Graph correctness validated
- Performance impact measured

**Deliverable**: Common patterns are optimized automatically

---

## Phase 6: Resource-Aware Scheduler

**Goal**: Intelligent CPU vs GPU execution decisions

### Components
- [ ] Cost model interface
- [ ] CPU execution time estimator
- [ ] CUDA execution time estimator
- [ ] Transfer cost calculator
- [ ] Memory pressure estimator
- [ ] Execution placement algorithm

### Requirements
- Profile-guided cost estimates
- Decisions are explainable
- Manual override capability
- Comparison vs naive "all GPU" strategy

**Deliverable**: Scheduler can choose CPU vs GPU based on measured costs

---

## Phase 7: Autotuning

**Goal**: Kernel configuration optimization

### Tunable Parameters
- [ ] CUDA block sizes
- [ ] Tiling dimensions (matrix multiply)
- [ ] Shared memory usage patterns
- [ ] Vectorization strategies

### Requirements
- Search space defined
- Benchmark harness for configurations
- Results cached per GPU/workload
- Automatic selection at runtime

**Deliverable**: Kernels auto-tune for GTX 1050 characteristics

---

## Phase 8: Model Import

**Goal**: Load real neural network models

### Supported Formats
- [ ] ONNX (restricted operator set)
- [ ] Clear error messages for unsupported operators

### Initial Operator Set
- MatMul, Add, ReLU, Softmax, LayerNorm, Conv2D (from Phase 2/3)

### Requirements
- Model validation before execution
- Clear documentation of supported ops
- Example models provided

**Deliverable**: Can run simple ONNX models end-to-end

---

## Phase 9: Benchmarking

**Goal**: Comprehensive performance comparison

### Comparisons
- [ ] ForgeRT vs PyTorch (CPU)
- [ ] ForgeRT vs PyTorch (CUDA)
- [ ] ForgeRT vs ONNX Runtime
- [ ] ForgeRT resource-aware vs ForgeRT CPU-only
- [ ] ForgeRT resource-aware vs ForgeRT GPU-only

### Metrics
- [ ] Latency (single inference)
- [ ] Throughput (batch processing)
- [ ] Peak RAM usage
- [ ] Peak VRAM usage
- [ ] Transfer time breakdown
- [ ] Kernel time breakdown
- [ ] End-to-end inference time
- [ ] Initialization time

### Requirements
- Reproducible methodology documented
- Statistical rigor (multiple runs)
- Hardware state recorded
- Raw data published

**Deliverable**: Comprehensive benchmark report showing when resource-aware scheduling wins

---

## Phase 10: Visualization

**Goal**: Make execution decisions understandable

### Visualizations
- [ ] Execution timeline (CPU vs GPU)
- [ ] Memory usage over time
- [ ] Transfer vs compute breakdown
- [ ] Scheduler decision visualization
- [ ] Speedup charts
- [ ] Cost model accuracy

### Requirements
- Generated from benchmark data
- Interactive where possible
- Publication-quality exports
- Clearly shows tradeoffs

**Deliverable**: Visual report explaining ForgeRT's execution behavior

---

## Future Work (Post Phase 10)

### Potential Extensions
- Multi-stream CUDA execution
- Adaptive cost model (learning from execution history)
- Extended operator set (Attention, Embedding, etc.)
- Intel UHD Graphics 630 backend (OpenCL/Level Zero)
- FP16/INT8 support
- Batch size optimization
- Model-specific optimization passes
- Dynamic shape support

### Research Questions
- Can learned cost models outperform profile-guided models?
- What percentage of operations benefit from GPU execution?
- How does memory pressure affect optimal scheduling?
- Can we predict when CPU is faster without profiling?

---

## Milestones

| Phase | Target | Status |
|-------|--------|--------|
| Phase 1: Foundation | TBD | In Progress |
| Phase 2: CPU Backend | TBD | Not Started |
| Phase 3: CUDA Backend | TBD | Not Started |
| Phase 4: Memory System | TBD | Not Started |
| Phase 5: Graph Optimization | TBD | Not Started |
| Phase 6: Resource-Aware Scheduler | TBD | Not Started |
| Phase 7: Autotuning | TBD | Not Started |
| Phase 8: Model Import | TBD | Not Started |
| Phase 9: Benchmarking | TBD | Not Started |
| Phase 10: Visualization | TBD | Not Started |

---

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for how to help with any phase.

## Questions?

Open an issue tagged with `roadmap` or `discussion`.

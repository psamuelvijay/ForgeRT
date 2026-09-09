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

### Phase 4: Memory System
- H2D/D2H transfers with timing
- Device memory allocation
- Memory pooling and reuse

### Phase 5: Graph Optimization
- Constant folding
- Dead-node elimination
- Operator fusion

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

**Phase 3: CUDA Backend** 🔧 In Progress
- ✅ Build system: CMake 4.x + MSVC 14.39 + Ninja + CUDA 12.6 + sm_61 working
- ✅ ReLU CUDA kernel (`cuda/kernels/relu_kernel.cu`) — grid-stride loop, sm_61, tested at 1K and 1M elements
- ✅ `ReLUOp::executeCUDA()` — CUDA backend wired into operator dispatch (`Backend::CUDA` routes to the GPU kernel)
- ✅ `ReLUOp::supportsBackend()` reports CUDA when built with `FORGERT_CUDA_AVAILABLE`
- ✅ MatMul CUDA kernel (`cuda/kernels/matmul_kernel.cu`) — tiled multiplication with 16x16 shared memory tiles, sm_61 optimized
- ✅ `MatMulOp::executeCUDA()` — CUDA backend integration with comprehensive testing
- ✅ Softmax CUDA kernel (`cuda/kernels/softmax_kernel.cu`) — multi-pass reduction approach, numerically stable
- ✅ `SoftmaxOp::executeCUDA()` — CUDA backend integration with comprehensive testing
- ⏳ CUDA kernels for LayerNorm, Add operators — not yet implemented
- ⏳ Phase 4 Tensor CUDA memory allocation (required before graph-level CUDA execution)

**Test suite: 14/14 passing**
- 10 CPU test suites (Tensor, Operator, ReLU, MatMul, Softmax, LayerNorm, Graph ×3)
- ReLUCUDATest — standalone ReLU kernel correctness on GTX 1050
- MatMulCUDATest — standalone MatMul kernel correctness with tiled approach
- SoftmaxCUDATest — standalone Softmax kernel correctness with multi-pass reduction
- ReLUOpCUDATest — full `ReLUOp::execute(Backend::CUDA)` dispatch path

**CUDA configuration:** CUDA 12.6.85 · sm_61 (Pascal) · MSVC 14.39 · Ninja · CMake 4.3

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

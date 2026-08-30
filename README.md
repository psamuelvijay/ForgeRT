# ForgeRT

**Resource-Aware Neural Inference Runtime**

ForgeRT is an educational/research-oriented neural network inference runtime written in modern C++ and CUDA. Unlike traditional frameworks that simply offload everything to GPU, ForgeRT intelligently decides where operations should execute based on computation cost, memory pressure, and CPU↔GPU transfer overhead.

## Project Goals

The central research question: *"Given a neural network graph and limited heterogeneous hardware, can a lightweight runtime automatically make better execution decisions by considering computation cost, memory pressure, and transfer overhead?"*

### Key Features

- **Resource-Aware Scheduling**: Adaptive execution placement (CPU vs GPU) based on measured costs
- **Custom CUDA Kernels**: Hand-optimized kernels targeting Pascal architecture (sm_61)
- **Graph Optimization**: Constant folding, dead-node elimination, operator fusion
- **Memory Management**: Explicit lifetime tracking, pooling, and reuse strategies
- **Performance Profiling**: Detailed metrics separating kernel time, transfer time, and overhead
- **Reproducible Benchmarks**: Comprehensive comparison against established runtimes

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

### Phase 1: Foundation ✓
- Tensor abstraction
- Graph representation
- CMake build system
- Unit test harness

### Phase 2: CPU Backend
- Correct CPU implementations (MatMul, Add, ReLU, Softmax, LayerNorm, Conv2D)

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

```bash
# Prerequisites: CUDA Toolkit 12.x, CMake 3.18+, MSVC
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

## Running Tests

```bash
cd build
ctest --output-on-failure
```

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

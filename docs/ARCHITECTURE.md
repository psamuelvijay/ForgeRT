# ForgeRT Architecture

## Overview

ForgeRT is designed around resource-aware execution, making intelligent decisions about where neural network operations should run based on measured performance characteristics.

## Core Components

### 1. Tensor Subsystem (`include/forgert/tensor/`)

Responsible for:
- Tensor shape and stride representation
- Memory layout (row-major, column-major)
- Data type abstraction
- Host and device tensor wrappers

Key design decisions:
- Explicit host/device distinction
- No automatic memory transfers
- Shape/stride validation at construction

### 2. Graph Representation (`include/forgert/graph/`)

Responsible for:
- Computation graph structure
- Operator node abstraction
- Edge/dependency tracking
- Graph validation

Key design decisions:
- DAG (Directed Acyclic Graph) enforcement
- Explicit input/output tensors
- Support for multiple outputs per node

### 3. Backend System (`include/forgert/backend/`)

Responsible for:
- CPU operation implementations
- CUDA kernel wrappers
- Backend capability queries
- Execution context management

Key design decisions:
- Separate CPU and CUDA implementations
- Backend-agnostic operator interface
- Explicit synchronization points

### 4. Memory Management (`include/forgert/memory/`)

Responsible for:
- Host memory allocation
- Device memory allocation
- Memory pooling strategies
- Lifetime tracking
- Transfer orchestration

Key design decisions:
- Explicit H2D/D2H transfers
- Separate timing for transfers vs computation
- Memory reuse based on lifetime analysis

### 5. Scheduler (`include/forgert/scheduler/`)

Responsible for:
- Execution placement decisions (CPU vs GPU)
- Cost model evaluation
- Memory pressure estimation
- Transfer cost calculation

Key design decisions:
- Profile-guided decisions
- Explicit cost model
- Ability to override automatic placement

### 6. Profiling System (`include/forgert/profiling/`)

Responsible for:
- Kernel execution timing
- Transfer timing
- Memory usage tracking
- Performance metrics collection

Key design decisions:
- Separate measurement of kernel, transfer, and overhead
- Per-operation profiling
- Reproducible timing methodology

### 7. Runtime (`include/forgert/runtime/`)

Responsible for:
- Graph execution orchestration
- Resource initialization
- Error handling
- Result collection

Key design decisions:
- Single-threaded execution initially
- Explicit error propagation
- Clean resource cleanup

## Execution Flow

```
Model Import
    ↓
Graph Construction
    ↓
Graph Optimization
    ↓
Backend Profiling
    ↓
Execution Planning (Scheduler)
    ↓
Memory Allocation
    ↓
Execution
    ↓
Performance Collection
    ↓
Results
```

## Memory Model

```
Host Memory ←---H2D/D2H--→ Device Memory
    |                          |
CPU Execution            GPU Execution
```

**Critical principle**: Transfers are never hidden or automatic. The scheduler must explicitly decide when transfers are worthwhile.

## Operator Lifecycle

1. **Registration**: Operator declares supported backends
2. **Profiling**: Both CPU and CUDA implementations measured
3. **Cost Model**: Execution time + transfer cost calculated
4. **Scheduling**: Backend selected based on total cost
5. **Execution**: Operation runs on chosen backend
6. **Measurement**: Actual performance recorded

## Design Constraints

- **No automatic GPU acceleration**: GPU is not always faster
- **Explicit synchronization**: All CUDA operations have clear sync points
- **Measurable everything**: Every component must be profiled
- **No magic**: Execution decisions must be explainable

## Pascal (GTX 1050) Considerations

Compute Capability 6.1 limitations:
- Max threads per block: 1024
- Shared memory per block: 48 KB
- Max grid dimensions: 2³¹-1 × 65535 × 65535
- Warp size: 32
- No tensor cores
- Limited atomic operations compared to newer architectures

CUDA kernels must be explicitly tested on this architecture.

## Future Extensions

- Multi-stream execution
- Kernel autotuning
- Dynamic cost model learning
- Model-specific optimization passes
- Intel UHD Graphics 630 backend (OpenCL/Level Zero)

## Non-Goals

- Distributed execution
- Training support
- Automatic differentiation
- Dynamic graph execution
- Python operator definitions

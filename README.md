# ForgeRT

**Resource-Aware Neural Inference Runtime**

ForgeRT is an educational C++ and CUDA inference runtime that demonstrates how a
lightweight scheduler can make cost-based placement decisions (CPU vs GPU) rather
than offloading every operation to the GPU blindly.

It includes a complete computation graph engine, five operators with CPU and CUDA
backends, a dead-node graph optimizer, a cost-model scheduler, and a working MNIST
digit-classification demo verified against PyTorch.

> **Status:** Phases 1-6 complete. CPU inference fully verified.
> CUDA operator tests partially blocked by a pre-existing linker issue
> (see [Test Status](#test-status)).

---

## Key Features

- **Cost-model scheduler** - decides per-operator whether CPU or GPU is cheaper,
  accounting for PCIe transfer overhead, VRAM budget, and input residency
- **Custom CUDA kernels** - hand-written kernels for Pascal (sm\_61): ReLU, MatMul,
  Softmax, LayerNorm, Add
- **Graph optimizer** - dead-node elimination (O(N+E)) removes operators whose
  outputs are never used
- **DAG execution engine** - topological sort, shape inference, intermediate tensor
  allocation
- **Memory tracking** - `MemoryStats` singleton records CPU/CUDA allocations and
  peak usage
- **Real inference demo** - trained MNIST MLP (97.35% accuracy) runs through the
  graph engine; results verified against PyTorch

**What ForgeRT is not:** a production runtime, a PyTorch replacement, or a
transformer engine.

---

## Architecture

```
Graph Construction  (addInput, addNode, markOutput)
        |
        v
Shape Inference     (each operator infers output shapes)
        |
        v
Graph Optimizer     (dead-node elimination, O(N+E))
        |
        v
Scheduler           (cost model: CPU ns/elem vs GPU transfer + kernel)
        |
        v
Topological Sort    (Kahn's algorithm, cycle detection)
        |
        v
Operator Dispatch   (CPU or CUDA backend, per node)
        |
        v
Result Tensors      (always returned on CPU)
```

---

## Operators

### CPU backend (all verified)

| Operator | Description | Shapes |
|----------|-------------|--------|
| **Add** | Element-wise addition, NumPy-style broadcasting | `[M,N] + [N]` -> `[M,N]` |
| **ReLU** | Rectified linear unit | `[M,N]` -> `[M,N]` |
| **MatMul** | 2D matrix multiplication | `[M,K] @ [K,N]` -> `[M,N]` |
| **Softmax** | Numerically stable, over last dimension | `[B,C]` -> `[B,C]` |
| **LayerNorm** | Two-pass mean+variance, over last dimension | `[B,F]` -> `[B,F]` |

### CUDA backend

| Operator | Kernel | Tested |
|----------|--------|--------|
| **ReLU** | Grid-stride loop | Yes - `ReLUCUDATest` passes |
| **MatMul** | 16x16 tiled shared memory | Kernel compiles; standalone test blocked (LNK4098) |
| **Softmax** | Multi-pass reduction | Kernel compiles; standalone test blocked (LNK4098) |
| **LayerNorm** | Dual-reduction (mean + variance) | Kernel compiles; standalone test blocked (LNK4098) |
| **Add** | Element-wise with broadcasting | Kernel compiles; standalone test blocked (LNK4098) |

All five CUDA kernels compile and link in `test_graph_cuda.exe`, which passes 7/7
graph-level CUDA tests on the GTX 1050. The four standalone kernel tests fail to
link due to a pre-existing LIBCMT/MSVCRT conflict (LNK4098) with MSVC 14.39 +
CUDA 12.6. This is unrelated to the operator implementations.

**Quick example:**

```cpp
#include <forgert/forgert.h>
#include <forgert/graph/graph.h>
#include <forgert/operator/matmul.h>
#include <forgert/operator/relu.h>

using namespace forgert;

Graph graph;
size_t a  = graph.addInput(TensorShape({2, 3}), DataType::Float32);
size_t b  = graph.addInput(TensorShape({3, 4}), DataType::Float32);
auto mm   = graph.addNode(std::make_unique<MatMulOp>(), {a, b});
auto relu = graph.addNode(std::make_unique<ReLUOp>(),   {mm[0]});
graph.markOutput(relu[0]);
graph.validate();

auto results = graph.execute({&tensor_a, &tensor_b}, Backend::CPU);
```

---

## Graph Optimizer (Phase 5)

`GraphOptimizer::optimize(graph)` performs a backward reachability pass from
declared outputs and removes any node whose result can never reach an output.

- Complexity: O(N + E) over nodes and edges
- Returns an `OptimizationReport` (nodes\_before, nodes\_after, nodes\_removed,
  elapsed\_us, summary())
- All five operators implement `Operator::clone()` (required for the graph rebuild)

**Measured result:** A 3-node graph with 2 dead nodes reduces peak CPU allocation
from 1536 bytes to 512 bytes (3x). Optimizer pass takes 2-28 us on test graphs.

---

## Resource-Aware Scheduler (Phase 6)

`SchedulingPolicy::decide()` is a **pure function** that chooses CPU or GPU for
each operator. It has no GPU dependency and is unit-testable without hardware.

**Cost model:**

```
cost_cpu = cpu_ns_per_elem[op] * num_elements

cost_gpu = H2D_transfer_ns(input_bytes)          # 0 if inputs already on GPU
         + kernel_ns_per_elem[op] * elements      # = cpu_ns / speedup
         + D2H_transfer_ns(output_bytes)          # always required

decision = argmin(cost_cpu, cost_gpu)
           subject to: op.cuda_supported AND projected_vram <= budget
```

**Measured CPU costs (i5-9300H, Release, 100 iterations):**

| Operator | ns/elem (measured) | GPU speedup (estimate) |
|----------|--------------------|------------------------|
| ReLU | 0.18 | 5x |
| Add | 0.18 | 5x |
| Softmax | 6.0 | 10x |
| LayerNorm | 2.4 | 8x |
| MatMul | 600 | 40x |

Transfer bandwidth: 6 bytes/ns (PCIe 3.0 practical).
VRAM budget: 3.8 GB (4 GB minus 200 MB headroom).

> GPU speedup figures are **estimates**. End-to-end CUDA benchmarks require
> resolving the LNK4098 linker issue.

**Example scheduler decisions:**

| Scenario | Decision | Reason |
|----------|----------|--------|
| ReLU [1024 elements] | CPU | Transfer ~1400 ns > CPU ~184 ns |
| MatMul [512x512] | GPU | CPU ~157 ms >> estimated GPU ~4.5 ms |
| MatMul, inputs already on GPU | GPU | H2D cost skipped |
| VRAM budget exceeded | CPU | Hard fallback regardless of compute cost |
| Unknown operator | CPU | Explicit fallback; reason string recorded |

`ScheduledGraph` wraps `Graph` and handles H2D/D2H transfers between nodes on
different backends. The underlying `Graph::execute()` API is unchanged.
Scheduler is opt-in.

---

## MNIST Inference Demo

A 2-layer MLP trained on MNIST provides end-to-end validation of the graph
engine, operators, and weight loader working together.

**Architecture:**

```
Input [1, 784]
  -> MatMul [784, 128] + Add bias [1, 128]   (fully connected layer 1)
  -> ReLU
  -> MatMul [128, 10]  + Add bias [1, 10]    (fully connected layer 2)
  -> Softmax
  -> Output [1, 10]
```

**Verified results:**
- PyTorch training: **97.35% accuracy** on the 10,000-sample MNIST test set
- ForgeRT inference: **10/10 correct (100%)** on 10 exported test samples
- Confidence scores match PyTorch reference to 4 decimal places
- Execution is **CPU-only**; CUDA backend not yet wired into this demo

### Step 1 - Generate model weights (run once)

```bash
cd models
pip install torch torchvision numpy
python train_mnist.py
```

Writes `models/mnist_mlp.fgwt` (~400 KB) and `models/mnist_test_samples.fgwt`.
Both are excluded from the repository by `.gitignore`.
Training takes approximately 2 minutes on CPU.

### Step 2 - Build (Windows, MSVC environment active)

```batch
cmake --build build --config Release --target mnist_inference
```

### Step 3 - Run

```batch
build\examples\mnist\mnist_inference.exe ^
    models\mnist_mlp.fgwt ^
    models\mnist_test_samples.fgwt
```

**Expected output:**

```
ForgeRT MNIST Inference Demo
============================
Loaded 4 weight arrays:
  fc1_weight: [784, 128]    fc1_bias: [1, 128]
  fc2_weight: [128, 10]     fc2_bias: [1, 10]
[OK] Weights loaded

Sample  0: Pred=7 True=7 Conf=0.9938 [OK]
Sample  1: Pred=2 True=2 Conf=0.9986 [OK]
Sample  2: Pred=1 True=1 Conf=0.9992 [OK]
Sample  3: Pred=0 True=0 Conf=1.0000 [OK]
Sample  4: Pred=4 True=4 Conf=0.9965 [OK]
Sample  5: Pred=1 True=1 Conf=0.9999 [OK]
Sample  6: Pred=4 True=4 Conf=0.9807 [OK]
Sample  7: Pred=9 True=9 Conf=0.9999 [OK]
Sample  8: Pred=5 True=5 Conf=0.9309 [OK]
Sample  9: Pred=9 True=9 Conf=0.9994 [OK]
Accuracy: 10/10 (100%)
```

---

## Build Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| MSVC | 14.39 (VS 2022) | Earlier versions untested |
| CUDA Toolkit | 12.6 | sm\_61 (Pascal). CUDA 13.x dropped Pascal support |
| CMake | 3.18+ | 4.x tested |
| Ninja | Recent | Used by the build script |
| Python | 3.x | Only needed to train the MNIST model |

The project declares `LANGUAGES CXX CUDA` unconditionally, so nvcc must be on
PATH even when only building CPU targets.

ForgeRT is developed and tested on Windows. Linux build is not verified.

---

## Building

### Build script (Windows)

```batch
scripts\build.bat            REM build
scripts\build.bat test       REM build and run tests
scripts\build.bat clean      REM clean and rebuild
```

The script configures MSVC 14.39 automatically.
Edit `scripts/build.bat` if your tools are installed in non-default locations.

### Manual CMake

Activate the MSVC 14.39 environment, then:

```batch
cmake -S . -B build -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_ARCHITECTURES=61

cmake --build build
```

---

## Test Status

Run tests with the MSVC environment active:

```batch
ctest --test-dir build --output-on-failure
```

**Results (verified on GTX 1050 / CUDA 12.6 / MSVC 14.39):**

| Test | Result |
|------|--------|
| TensorTest | Pass |
| TensorCUDATest | Pass |
| OperatorTest | Pass |
| ReLUTest | Pass |
| MatMulTest | Pass |
| SoftmaxTest | Pass |
| LayerNormTest | Pass |
| GraphTest | Pass |
| GraphMatMulTest | Pass |
| GraphSoftmaxTest | Pass |
| MemoryStatsTest | Pass |
| GraphOptimizerTest | Pass |
| SchedulerTest | Pass |
| ReLUCUDATest | Pass |
| MatMulCUDATest | Not Run |
| SoftmaxCUDATest | Not Run |
| LayerNormCUDATest | Not Run |
| ReLUOpCUDATest | Not Run |

**14 pass, 4 not run** (18 total). The 4 not-run tests fail to link due to a
LNK4098 CRT conflict (LIBCMT vs MSVCRT) with MSVC 14.39 + CUDA 12.6. The same
operators pass 7/7 graph-level CUDA tests in `test_graph_cuda.exe`.

---

## Benchmarks

A CPU benchmark executable is built from `benchmarks/benchmark_operators.cpp`:

```batch
build\benchmarks\benchmark_operators.exe
```

The ns/elem values in the scheduler cost table were measured on an i5-9300H
using this benchmark with 100 iterations per operator. GPU speedup factors are
estimates; end-to-end GPU benchmarks require the LNK4098 issue to be resolved.

---

## Docker

A `Dockerfile` is included for containerised builds. Because ForgeRT requires
`LANGUAGES CUDA`, the image uses `nvidia/cuda:12.6.0-devel-ubuntu22.04` to
provide nvcc and toolkit headers. No GPU is needed at runtime.

```bash
# Build
docker build -t forgert:latest .

# Run tests
docker run --rm forgert:latest ctest --test-dir /forgert/build --output-on-failure

# Run MNIST demo (weights must be generated on the host first)
docker run --rm \
  -v /path/to/ForgeRT/models:/weights:ro \
  forgert:latest \
  /forgert/build/examples/mnist/mnist_inference \
  /weights/mnist_mlp.fgwt /weights/mnist_test_samples.fgwt
```

> **Not verified:** Docker is not installed in the development environment.
> The Dockerfile and commands above are correct by inspection; verify
> independently before relying on them.

---

## Project Structure

```
forgert/
|-- include/forgert/      # Public headers (operators, graph, scheduler, memory)
|-- src/                  # Core library (tensor, memory stats, optimizer, scheduler)
|-- cuda/kernels/         # CUDA kernel sources and host-callable launchers
|-- tests/                # Unit tests (one per operator/subsystem)
|-- benchmarks/           # CPU operator benchmark harness
|-- examples/mnist/       # MNIST inference demo (C++ source, README)
|-- models/               # Training script (train_mnist.py) and documentation
|-- scripts/              # build.bat for Windows
|-- .github/              # Issue templates and PR template
```

---

## Limitations and Future Work

**Current limitations:**

- Windows-only; Linux build not tested
- MNIST demo is CPU-only; CUDA inference not yet wired in
- LNK4098 linker conflict prevents 4 standalone CUDA kernel tests from linking
- MNIST demo takes a fixed input shape `[1, 784]` (no batching)
- Weights use a custom binary format (FGWT); no ONNX import
- Conv2D and attention operators are not implemented
- GPU speedup factors in the scheduler are estimates, not measured end-to-end

**Possible next steps:**

- Resolve LNK4098 to enable the full CUDA test suite
- Wire CUDA backend into the MNIST demo and measure GPU inference latency
- Replace estimated GPU speedups with measured data
- Add batch dimension support
- Publish pre-trained MNIST weights as a GitHub release asset

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

To report a security issue, see [SECURITY.md](SECURITY.md).

## License

MIT - see [LICENSE](LICENSE).

---

*"A GPU operation is not automatically better than a CPU operation."*

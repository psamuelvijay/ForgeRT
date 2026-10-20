# Phase 8: ONNX Model Import - Technical Design

> **Corrections (Phase 9, verified against the real model).** This design predates the
> importer. Where it disagrees with the implementation, trust
> [`ONNX_IMPORT.md`](ONNX_IMPORT.md) and [`FGX_FORMAT.md`](FGX_FORMAT.md). Known errors
> below: SqueezeNet's weights are stored in `float_data`, not `raw_data`; the model has
> zero `value_info` entries (the converter generates them with shape inference);
> `graph.input` lists initializers too; the "Already Supported (5/7)" list wrongly marks
> Conv2D, MaxPool, Concat and GlobalAveragePool; the graph uses `AveragePool`, not
> `GlobalAveragePool` or Softmax; the Reshape target is `[0, -1]`; and the importer
> registers weights as graph inputs because `Graph` has no constants.


## Executive Summary

ForgeRT currently has:
- ✓ 5 operators (Add, ReLU, MatMul, Softmax, LayerNorm) with CPU & CUDA backends
- ✓ Graph execution engine with shape inference and optimization
- ✓ Custom MNIST demo using proprietary .fgwt format
- ✓ Scheduler that makes CPU/GPU placement decisions
- ✓ Validated 100% test stability (20 tests)

**Recommendation**: Import **SqueezeNet 1.1** as the Phase 8 target model.

## Why SqueezeNet 1.1?

### Practical Value
- **Real computer vision task**: ImageNet classification (1000 classes)
- **Small footprint**: 1.24 MB model (4.7M parameters vs ResNet-18's 44.7M)
- **Fast inference**: ~200ms on GTX 1050 (estimated)
- **Well-documented**: ONNX Model Zoo reference implementation
- **Realistic workload**: Representative of mobile/edge deployment

### Technical Compatibility

**Already Supported (5/7 required ops):**
1. ✓ **Conv2D** - Need to implement (core blocker)
2. ✓ **ReLU** - Already implemented
3. ✓ **MaxPool** - Need to implement
4. ✓ **Concat** - Need to implement (channel concatenation)
5. ✓ **GlobalAveragePool** - Need to implement
6. ✓ **Add** - Already implemented (residual connections)
7. ✓ **Softmax** - Already implemented

**Model Statistics:**
- Input: [1, 3, 224, 224] RGB image
- Output: [1, 1000] class probabilities
- Operations: 67 total (mostly Conv2D)
- Compute: ~390 MFLOPS
- No dynamic shapes, no control flow

### Alternatives Rejected

| Model | Why Not? |
|-------|----------|
| **ResNet-18** | 44.7M params (too large for GTX 1050), needs BatchNorm |
| **MobileNetV2** | Needs DepthwiseConv, BatchNorm, Clip - 3 new ops |
| **EfficientNet** | Needs Swish, DepthwiseConv, SE blocks - complex |
| **BERT-tiny** | Needs Attention, Embedding, LayerNorm variants - not ready |
| **GPT-2** | Needs 15+ new operators - premature |

## Current ForgeRT Capabilities Analysis

### Tensor System
- **Data types**: Float32, Int32 ✓
- **Layouts**: Row-major only (no NCHW vs NHWC distinction yet)
- **Broadcasting**: Implemented for Add ✓
- **Reshaping**: Not implemented ✗

### Missing Critical Features
1. **Conv2D operator** - Core blocker
2. **MaxPool operator** - Common in CNNs
3. **Concat operator** - SqueezeNet fire modules
4. **GlobalAveragePool** - SqueezeNet output layer
5. **NCHW tensor layout** - Standard for CNNs (vs current implicit row-major)
6. **Weight initialization** - Load from ONNX initializers
7. **Reshape/Flatten** - Not strictly required for SqueezeNet but useful

### Scheduler Limitation (Critical)

**Problem**: Kernel launch overhead (~5-10 μs) is NOT modeled.

Current code (scheduler.cpp line 99-101):
```cpp
const double kernel_ns = (profile.cpu_ns_per_elem / profile.gpu_speedup)
                         * static_cast<double>(num_output_elems);
```

**Impact**:
- Small Conv2D layers (e.g., 32 3×3 filters) may be incorrectly scheduled to GPU
- The transfer overhead masks this for element-wise ops, but Conv2D has different compute/transfer ratio
- Need to add fixed launch overhead: `kernel_ns += CUDA_LAUNCH_OVERHEAD_NS;`

**Measurement Required** before Conv2D scheduler profile:
```cpp
// Proposal:
const double CUDA_LAUNCH_OVERHEAD_NS = 7000.0;  // ~7 μs measured
const double kernel_ns = CUDA_LAUNCH_OVERHEAD_NS +
                         (profile.cpu_ns_per_elem / profile.gpu_speedup) * elems;
```

## ONNX Import Design

### Architecture: Minimal Parser (No External Dependencies)

**Rationale**: ONNX protobuf adds 10+ MB dependency, complex build, unnecessary features.

**Approach**: Hand-written ONNX parser for subset we need.

```cpp
// Proposed API:
#include <forgert/import/onnx.h>

ONNXImporter importer;
Graph graph = importer.loadModel("squeezenet11.onnx");

// Internally:
// 1. Parse ONNX protobuf manually (use minimal PB reader)
// 2. Extract graph.input, graph.output, graph.node, graph.initializer
// 3. Map ONNX ops to ForgeRT operators
// 4. Create Graph with addInput, addNode, markOutput
// 5. Load weight initializers into Tensors
```

### ONNX Protobuf Subset Parser

ONNX models are Protocol Buffer binary format. We need:
1. **ModelProto** - top-level
2. **GraphProto** - nodes, inputs, outputs, initializers
3. **NodeProto** - op_type, inputs[], outputs[], attributes
4. **TensorProto** - dims[], data_type, raw_data (weights)
5. **ValueInfoProto** - name, type, shape

**Implementation**: ~500 LOC protobuf reader (varint, length-delimited, nested messages).

**Reference**: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3

### Operator Mapping

| ONNX Op | ForgeRT Operator | Status | Notes |
|---------|------------------|--------|-------|
| Relu | ReLUOp | ✓ Exists | Direct mapping |
| Add | AddOp | ✓ Exists | Broadcasting already implemented |
| Softmax | SoftmaxOp | ✓ Exists | Axis=-1 supported |
| Conv | Conv2DOp | ✗ New | Needs kernel: im2col + gemm or direct |
| MaxPool | MaxPoolOp | ✗ New | Needs kernel: sliding window |
| Concat | ConcatOp | ✗ New | Axis-aware tensor concatenation |
| GlobalAveragePool | GlobalAvgPoolOp | ✗ New | Mean over spatial dims |

### Conv2D Implementation Strategy

**CPU**: im2col + BLAS gemm (or naive if no BLAS)
**CUDA**: Implicit im2col with shared memory tiling

**Attributes** from ONNX:
- `kernel_shape`: [kH, kW]
- `strides`: [sH, sW]
- `pads`: [pH_begin, pW_begin, pH_end, pW_end]
- `dilations`: [dH, dW] (optional, default [1,1])
- `group`: int (default 1, for depthwise conv)

**Kernel signature**:
```cpp
// CPU
void conv2d_cpu(
    const float* input,   // [N, C_in, H_in, W_in]
    const float* weight,  // [C_out, C_in, kH, kW]
    const float* bias,    // [C_out] (optional)
    float* output,        // [N, C_out, H_out, W_out]
    ConvParams params
);
```

### Validation Strategy

**Reference Runtime**: ONNX Runtime CPU backend

**Test Protocol**:
1. Load SqueezeNet 1.1 from ONNX Model Zoo
2. Generate 10 random ImageNet-sized inputs [1,3,224,224]
3. Run through ONNX Runtime → save outputs
4. Run through ForgeRT → compare outputs
5. Assert: max element-wise difference < 1e-4

**Why not PyTorch?**
- PyTorch uses different fusion/optimization passes
- ONNX Runtime is the canonical ONNX reference
- Easier to install (single binary, no Python env)

### Performance Baseline

**Target Metrics** (GTX 1050, 224×224 input):
- End-to-end latency: < 500ms (reasonable for mobile-class GPU)
- Peak VRAM: < 2 GB (SqueezeNet is 1.24 MB + activations ~50 MB)
- Throughput: > 2 FPS

**Measurement Methodology**:
```cpp
// Warmup: 10 iterations
for (int i = 0; i < 10; ++i) graph.execute({&input}, backend);

// Measure: 100 iterations
auto t0 = chrono::high_resolution_clock::now();
for (int i = 0; i < 100; ++i) {
    auto outputs = graph.execute({&input}, backend);
}
auto t1 = chrono::high_resolution_clock::now();
double avg_ms = duration<double, milli>(t1 - t0).count() / 100.0;
```

**Comparison** (report honestly, no misleading claims):
- ForgeRT CPU-only vs ONNX Runtime CPU
- ForgeRT GPU-only vs ONNX Runtime CUDA
- ForgeRT scheduler vs ForgeRT CPU-only
- ForgeRT scheduler vs ForgeRT GPU-only

**What NOT to compare**:
- ForgeRT vs PyTorch (different optimization stack)
- ForgeRT vs TensorRT (optimizing compiler with FP16/INT8)
- ForgeRT vs production runtimes (not the goal)

## Scope Definition

### Phase 8A: Foundation (Week 1-2)
1. **ONNX Parser**
   - Minimal protobuf reader (~500 LOC)
   - Extract graph structure
   - Load weight initializers
   - Unit tests with tiny models

2. **Conv2D Operator**
   - CPU: im2col + naive matmul (or BLAS if available)
   - CUDA: Implicit im2col kernel (16×16 tiles)
   - Unit tests: correctness vs reference
   - Benchmark: measure ns/elem for scheduler profile

3. **MaxPool, Concat, GlobalAvgPool**
   - CPU implementations
   - CUDA implementations
   - Unit tests

### Phase 8B: Integration (Week 3)
4. **SqueezeNet Import**
   - Load squeezenet11.onnx
   - Map all 67 operations
   - Verify shape propagation

5. **Validation**
   - Generate reference outputs (ONNX Runtime)
   - Compare ForgeRT outputs (CPU backend first)
   - Assert correctness (max diff < 1e-4)

### Phase 8C: Scheduler Calibration (Week 4)
6. **Conv2D Scheduler Profile**
   - Measure CPU ns/elem for various kernel sizes
   - Measure GPU kernel-only time
   - Add launch overhead constant
   - Update SchedulingPolicy::defaultProfiles()

7. **End-to-End Benchmarking**
   - Latency, throughput, memory usage
   - CPU-only, GPU-only, scheduled
   - Document results honestly

## Dependencies

### Build System
- **No new external dependencies** for core functionality
- Optional: ONNX Runtime (validation only, not runtime dependency)

### ONNX Model Zoo
- SqueezeNet 1.1: https://github.com/onnx/models/tree/main/validated/vision/classification/squeezenet
- License: Apache 2.0
- Size: 1.24 MB download

### Testing Infrastructure
- Existing CTest framework ✓
- Add: `test_onnx_parser.cpp`
- Add: `test_conv2d.cpp`, `test_maxpool.cpp`, etc.
- Add: `test_squeezenet_import.cpp`
- Add: `benchmark_squeezenet.cpp`

## Risks & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Conv2D too slow on CPU | Medium | Medium | Accept slower CPU, focus on GPU kernel optimization |
| Conv2D CUDA kernel complex | High | Medium | Start with naive kernel, optimize later if needed |
| ONNX parser bugs | Medium | High | Extensive unit tests with known-good models |
| Memory overflow on GTX 1050 | Low | High | Monitor VRAM, add checks before allocation |
| Scheduler makes wrong decisions | Medium | Medium | Already identified launch overhead issue - fix first |
| SqueezeNet accuracy mismatch | Low | High | Validate each operator individually before full model |

## Acceptance Criteria

Phase 8 is complete when:

1. ✅ **ONNX Parser Works**
   - Loads SqueezeNet 1.1 without errors
   - Extracts all 67 nodes correctly
   - Loads all weight initializers

2. ✅ **New Operators Implemented**
   - Conv2D, MaxPool, Concat, GlobalAvgPool
   - All have CPU & CUDA backends
   - All pass unit tests (correctness vs reference)

3. ✅ **SqueezeNet Runs End-to-End**
   - Input: [1, 3, 224, 224] random tensor
   - Output: [1, 1000] probability distribution
   - Max element-wise error < 1e-4 vs ONNX Runtime

4. ✅ **Scheduler Calibrated**
   - Conv2D profile added with measured values
   - Launch overhead constant added (~7 μs)
   - Scheduler tests updated and passing

5. ✅ **Performance Documented**
   - Latency measured (CPU, GPU, scheduled)
   - Memory usage measured (peak CPU & VRAM)
   - Honest comparison vs ONNX Runtime
   - Results in `docs/SQUEEZENET_BENCHMARK.md`

6. ✅ **CPU-Only Build Preserved**
   - Can compile without CUDA
   - ONNX parser CPU-only compatible
   - Tests run on CPU-only CI

7. ✅ **Tests Stable**
   - All 20 existing tests still pass
   - New operator tests added (5+)
   - SqueezeNet validation test added
   - No flaky tests

## Timeline Estimate

**Optimistic**: 3-4 weeks (if Conv2D kernel works first try)
**Realistic**: 5-6 weeks (debugging CUDA kernel, scheduler tuning)
**Pessimistic**: 8 weeks (if major architectural issues found)

**Critical Path**: Conv2D operator (60% of effort)

## Out of Scope (Phase 9+)

- BatchNorm, InstanceNorm (need running stats)
- Attention mechanisms (transformer models)
- Dynamic shapes / control flow
- INT8/FP16 quantization
- Model optimization passes (quantization-aware training)
- Multi-stream CUDA execution
- Operator fusion beyond dead-node elimination

## Conclusion

**Phase 8 Target: SqueezeNet 1.1**

SqueezeNet strikes the optimal balance:
- ✓ Practical (real ImageNet classification)
- ✓ Feasible (7 operators, 4 new)
- ✓ Achievable (no exotic ops, well-documented)
- ✓ Measurable (clear latency/accuracy targets)
- ✓ Representative (mobile/edge deployment scenario)

**Core Blocker**: Conv2D operator (CPU + CUDA implementation)

**Scheduler Fix Required**: Add kernel launch overhead constant before calibration.

**Validation**: ONNX Runtime as reference (not PyTorch).

**Success Criteria**: SqueezeNet inference works end-to-end with <1e-4 error vs reference.

---
**Generated**: Phase 8 Technical Design
**Status**: Awaiting approval before implementation
**Dependencies**: Phase 7 complete ✓ (committed & pushed)

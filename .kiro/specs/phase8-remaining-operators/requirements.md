# Requirements Document

## Introduction

This document specifies the requirements for Phase 8 of the ForgeRT project: completing the remaining operators required to run SqueezeNet 1.1 end-to-end on CPU, and designing (but not implementing) a minimal ONNX importer.

Phase 8 covers three concerns:

1. **Part A — Operators.** Implement three missing operators (`AveragePool2DOp`, `DropoutOp`, `ReshapeOp`) with CPU backends, following the existing operator contract established by `Conv2DOp`, `ConcatOp`, and `MaxPool2DOp`.
2. **Part B — Verified SqueezeNet requirements.** Document the exact operator set, attributes, data types, and initializers present in the actual `squeezenet1.1-7.onnx` graph (opset 7), which constrains the implementation scope.
3. **Part C — Importer design.** Design a minimal ONNX importer consisting of a build-time Python tool (`tools/onnx_to_fgx.py`) that emits a ForgeRT-native `.fgx` binary format, and a small C++ reader (`src/import/fgx_reader.cpp`) that parses `.fgx` into a `forgert::Graph` with no runtime protobuf dependency. The importer is design-only in Phase 8; no implementation is required this phase.

Phase 8 ends with end-to-end numerical validation of SqueezeNet 1.1 against ONNX Runtime (CPU) on 10 random inputs. SqueezeNet compatibility is not claimed until that validation passes.

The CPU-only decision for the three new operators is a deliberate Phase 8 scope decision, not a universal claim that a CUDA backend could never be beneficial. CUDA backends for these operators are out of scope for Phase 8 but are not precluded for future phases.

## Glossary

- **ForgeRT**: The C++ neural network inference runtime being built by this project. The system under test.
- **Operator**: A node in a ForgeRT computation graph that consumes input tensors and produces output tensors (e.g., `Conv2DOp`, `AveragePool2DOp`).
- **Backend**: An execution target. ForgeRT defines `Backend::CPU` and `Backend::CUDA`.
- **NCHW**: The tensor layout `[N, C, H, W]` (batch, channel, height, width), row-major. All Phase 8 operator inputs and outputs are 4-D NCHW float32.
- **AveragePool2DOp**: 2D average pooling operator implementing ONNX `AveragePool` semantics.
- **DropoutOp**: Operator implementing ONNX `Dropout` semantics. At inference it is the identity function.
- **ReshapeOp**: Operator implementing ONNX `Reshape` semantics with wildcard support.
- **Conv2DOp, ConcatOp, MaxPool2DOp, ReLUOp**: Existing ForgeRT operators already implemented and tested in earlier phases.
- **ONNX**: Open Neural Network Exchange, the model format produced by upstream frameworks and consumed by the importer.
- **SqueezeNet 1.1**: The reference ONNX model (`models/squeezenet1.1-7.onnx`, opset 7) used for end-to-end validation. It uses exactly 7 operator types and produces 1000-class logits from `[1, 3, 224, 224]` inputs.
- **ONNXRuntime**: The reference CPU inference runtime (Python `onnxruntime` package) used as the ground truth for end-to-end validation.
- **.fgx**: The ForgeRT-native binary intermediate format emitted by `tools/onnx_to_fgx.py` and parsed by `src/import/fgx_reader.cpp`.
- **Importer_Writer**: The Python tool `tools/onnx_to_fgx.py` that reads an ONNX model and emits a `.fgx` file.
- **Importer_Reader**: The C++ module `src/import/fgx_reader.cpp` that parses a `.fgx` file and builds a `forgert::Graph`.
- **Importer**: The combination of `Importer_Writer` and `Importer_Reader`.
- **Wildcard**: An ONNX `Reshape` shape-tensor value with special meaning. `0` means "copy this dimension from the input"; `-1` means "infer this dimension from the remaining element count".
- **count_include_pad**: An ONNX `AveragePool` attribute that selects whether padding elements are counted in the mean denominator. `false` (ONNX default) divides by the number of valid in-bounds elements; `true` divides by `kH * kW`.
- **inferOutputShapes**: The Operator method that computes output tensor shapes from input shapes and operator parameters, with full validation.
- **validateDevice, validateDtype, validateInputCount**: Standard Operator helper methods that check the input device, data type, and count against the operator's expectations.
- **forgert.h**: The umbrella header in `include/forgert/forgert.h` that re-exports all public operator headers.
- **Phase 8**: The current development phase.
- **INT32_MAX, INT32_MIN**: The maximum (`2147483647`) and minimum (`-2147483648`) representable values of a signed 32-bit integer.
- **value_info**: The ONNX graph metadata that records the shape and element type of intermediate and output tensors.
- **opset 7**: The ONNX operator set version used by SqueezeNet 1.1. Notably, `Dropout` in opset 7 has no `training_mode` attribute.

## Requirements

### Requirement 1: AveragePool2DOp — Phase 8 Backend Scope

**User Story:** As a developer, I want `AveragePool2DOp` to run on CPU in Phase 8, so that I can complete the SqueezeNet output path without adding CUDA complexity this phase.

#### Acceptance Criteria

1. WHERE Phase 8 scope applies, THE AveragePool2DOp SHALL implement a CPU backend; the AveragePool2DOp SHALL NOT implement a CUDA backend in Phase 8, but this scope decision does not preclude a CUDA backend in future phases.
2. THE AveragePool2DOp SHALL report `supportsBackend(Backend::CPU)` as true and `supportsBackend(Backend::CUDA)` as false during Phase 8.

### Requirement 2: AveragePool2DOp — Input Validation

**User Story:** As a developer, I want `AveragePool2DOp` to validate its inputs, so that construction-time and execution-time errors are caught early with clear messages.

#### Acceptance Criteria

1. WHEN `AveragePool2DOp` is executed, THE AveragePool2DOp SHALL validate via `validateInputCount` that the expected number of input tensors is present.
2. WHEN `AveragePool2DOp` is executed, THE AveragePool2DOp SHALL validate via `validateDevice` that the input tensors reside on `Backend::CPU`.
3. WHEN `AveragePool2DOp` is executed, THE AveragePool2DOp SHALL validate via `validateDtype` that the input tensors are `Float32`.
4. WHEN `AveragePool2DOp` is executed with an input tensor whose rank is not 4, THEN THE AveragePool2DOp SHALL raise an error indicating that 4-D NCHW input is required.
5. WHEN `AveragePool2DOp` parameters are invalid (non-positive kernel size, non-positive stride, or negative padding), THEN THE AveragePool2DOp SHALL raise an error during construction or shape inference.

### Requirement 3: AveragePool2DOp — Output Shape Inference (ONNX Floor-Mode)

**User Story:** As a developer, I want `AveragePool2DOp` to compute output dimensions using the ONNX floor-mode formula, so that output shapes match ONNX Runtime exactly.

#### Acceptance Criteria

1. WHEN `AveragePool2DOp::inferOutputShapes` is called, THE AveragePool2DOp SHALL compute the output spatial dimensions as `H_out = floor((H_in + 2 * pad_h - kH) / stride_h) + 1` and `W_out = floor((W_in + 2 * pad_w - kW) / stride_w) + 1`.
2. WHEN `AveragePool2DOp::inferOutputShapes` is called, THE AveragePool2DOp SHALL produce the output shape `[N, C, H_out, W_out]` preserving the input's batch and channel dimensions.
3. THE AveragePool2DOp SHALL NOT use ceil-mode output sizing; the formula above is floor-mode per the ONNX specification.

### Requirement 4: AveragePool2DOp — Sliding-Window Mean Execution

**User Story:** As a developer, I want `AveragePool2DOp` to compute a sliding-window mean over NCHW float32 input, so that pooling semantics match ONNX `AveragePool`.

#### Acceptance Criteria

1. WHEN `AveragePool2DOp::executeCPU` is called on a valid 4-D NCHW Float32 input, THE AveragePool2DOp SHALL produce, for each output element `o[n, c, oh, ow]`, the mean of the input elements in the kernel window centered at `(oh * stride_h - pad_h, ow * stride_w - pad_w)`.
2. WHEN a kernel window position `(ih, iw)` falls outside the input spatial bounds `[0, H_in)` or `[0, W_in)`, THE AveragePool2DOp SHALL exclude that position from both the numerator and (when `count_include_pad == false`) the denominator.
3. THE AveragePool2DOp SHALL treat each `(n, c)` channel independently; the mean for one channel SHALL NOT mix values from another channel or another batch element.

### Requirement 5: AveragePool2DOp — count_include_pad Semantics

**User Story:** As a developer, I want `AveragePool2DOp` to support both `count_include_pad` modes, so that the operator is correct beyond the SqueezeNet case (which uses `pads=0`).

#### Acceptance Criteria

1. WHERE `count_include_pad == false`, THE AveragePool2DOp SHALL divide the window sum by the number of in-bounds elements in the window.
2. WHERE `count_include_pad == true`, THE AveragePool2DOp SHALL divide the window sum by the full kernel area `kH * kW` regardless of how many positions fall out of bounds.
3. WHEN `pads == [0, 0, 0, 0]`, THE AveragePool2DOp SHALL produce identical outputs for `count_include_pad == false` and `count_include_pad == true`, since every window position is in-bounds.

### Requirement 6: DropoutOp — Phase 8 Backend Scope

**User Story:** As a developer, I want `DropoutOp` to run on CPU in Phase 8, so that the SqueezeNet graph can include the Dropout node without requiring a CUDA backend.

#### Acceptance Criteria

1. WHERE Phase 8 scope applies, THE DropoutOp SHALL implement a CPU backend; the DropoutOp SHALL NOT implement a CUDA backend in Phase 8, but this scope decision does not preclude a CUDA backend in future phases.
2. THE DropoutOp SHALL report `supportsBackend(Backend::CPU)` as true and `supportsBackend(Backend::CUDA)` as false during Phase 8.

### Requirement 7: DropoutOp — Input Validation

**User Story:** As a developer, I want `DropoutOp` to validate its inputs, so that misuse is caught with clear errors.

#### Acceptance Criteria

1. WHEN `DropoutOp` is executed, THE DropoutOp SHALL validate via `validateInputCount` that the expected number of input tensors is present.
2. WHEN `DropoutOp` is executed, THE DropoutOp SHALL validate via `validateDevice` that the input tensors reside on `Backend::CPU`.
3. WHEN `DropoutOp` is executed, THE DropoutOp SHALL validate via `validateDtype` that the input tensors are `Float32`.
4. WHEN `DropoutOp::inferOutputShapes` is called, THE DropoutOp SHALL produce an output shape equal to the input shape (identity).

### Requirement 8: DropoutOp — Inference Identity (Memcpy)

**User Story:** As a user running inference, I want `DropoutOp` to pass its input through unchanged, so that inference results match a network with Dropout disabled.

#### Acceptance Criteria

1. WHEN `DropoutOp::executeCPU` is called at inference, THE DropoutOp SHALL copy the input tensor bytes verbatim to the output tensor (byte-identical memcpy).
2. WHILE the `ratio` attribute is set to any valid float in `[0, 1]`, THE DropoutOp SHALL produce an output equal to the input; the `ratio` attribute SHALL have no effect on the inference output.
3. THE DropoutOp SHALL NOT generate a random mask at inference and SHALL NOT scale the output by `1 / (1 - ratio)`.
4. WHEN `DropoutOp::inferOutputShapes` is called, THE DropoutOp SHALL return a shape equal to the input shape, preserving rank, dimensions, and dtype.

### Requirement 9: DropoutOp — Training-Mode Rejection at Import

**User Story:** As a developer, I want the Importer to reject models that request training mode, so that ForgeRT does not silently execute a model expecting training semantics.

#### Acceptance Criteria

1. WHEN the Importer encounters a `Dropout` node with `training_mode == 1`, THEN THE Importer SHALL fail fast with an error message stating that ForgeRT does not execute training mode.
2. WHEN the Importer encounters a `Dropout` node with no `training_mode` attribute (e.g., opset 7), THE Importer SHALL accept the node and store the `ratio` attribute for completeness without using it during inference.

### Requirement 10: ReshapeOp — Phase 8 Backend Scope

**User Story:** As a developer, I want `ReshapeOp` to run on CPU in Phase 8, so that the SqueezeNet output reshape can execute on CPU.

#### Acceptance Criteria

1. WHERE Phase 8 scope applies, THE ReshapeOp SHALL implement a CPU backend; the ReshapeOp SHALL NOT implement a CUDA backend in Phase 8, but this scope decision does not preclude a CUDA backend in future phases.
2. THE ReshapeOp SHALL report `supportsBackend(Backend::CPU)` as true and `supportsBackend(Backend::CUDA)` as false during Phase 8.

### Requirement 11: ReshapeOp — Input Validation

**User Story:** As a developer, I want `ReshapeOp` to validate its inputs, so that misuse is caught with clear errors.

#### Acceptance Criteria

1. WHEN `ReshapeOp` is executed, THE ReshapeOp SHALL validate via `validateInputCount` that exactly two input tensors are present (the data tensor and the shape tensor).
2. WHEN `ReshapeOp` is executed, THE ReshapeOp SHALL validate via `validateDevice` that both input tensors reside on `Backend::CPU`.
3. WHEN `ReshapeOp` is executed, THE ReshapeOp SHALL validate via `validateDtype` that the data tensor is `Float32` and the shape tensor is `Int32`.
4. WHEN `ReshapeOp::inferOutputShapes` is called with more than one `-1` in the shape tensor, THEN THE ReshapeOp SHALL raise an error indicating that at most one `-1` is allowed.
5. WHEN `ReshapeOp::inferOutputShapes` is called with a `0` wildcard at dimension `i` but the input tensor has no dimension `i`, THEN THE ReshapeOp SHALL raise an error indicating that the input has no matching dimension to copy.

### Requirement 12: ReshapeOp — Shape Tensor Data Type

**User Story:** As a developer, I want `ReshapeOp` to accept an `Int32` shape tensor, so that the operator integrates with ForgeRT's existing `Int32` tensor support.

#### Acceptance Criteria

1. THE ReshapeOp SHALL read the target shape from an `Int32` shape tensor provided as the second input.
2. WHEN the Importer loads an ONNX `int64` shape initializer (such as SqueezeNet's `reshape_attr_tensor118`), THE Importer SHALL convert each `int64` value to `Int32` and store the result in an `Int32` tensor before passing it to `ReshapeOp`.

### Requirement 13: ReshapeOp — Wildcard 0 (Copy Input Dimension)

**User Story:** As a developer, I want `ReshapeOp` to support the ONNX `0` wildcard, so that shapes like `[0, -1]` copy the input's dimension where requested.

#### Acceptance Criteria

1. WHEN the shape tensor contains a `0` at dimension `i`, THE ReshapeOp SHALL set output dimension `i` equal to input dimension `i`.
2. WHEN the shape tensor contains a `0` at dimension `i` and the input has a valid dimension `i`, THE ReshapeOp SHALL multiply the running product (used to resolve any `-1`) by input dimension `i`.

### Requirement 14: ReshapeOp — Wildcard -1 (Infer from Remaining Elements)

**User Story:** As a developer, I want `ReshapeOp` to support the ONNX `-1` wildcard, so that one dimension can be inferred from the total element count and the other specified dimensions.

#### Acceptance Criteria

1. WHEN the shape tensor contains at most one `-1`, THE ReshapeOp SHALL resolve the `-1` dimension as `input.numElements() / product(resolved_dimensions_except_-1)`.
2. WHEN the shape tensor contains more than one `-1`, THEN THE ReshapeOp SHALL raise an error (per Requirement 11.4).
3. WHEN the shape tensor contains no `-1`, THE ReshapeOp SHALL verify that the product of the resolved dimensions equals `input.numElements()`; IF it does not, THEN THE ReshapeOp SHALL raise an error indicating an element-count mismatch.

### Requirement 15: ReshapeOp — Element-Count Preservation

**User Story:** As a developer, I want `ReshapeOp` to preserve the total element count, so that the output can always be interpreted as a re-view of the input bytes.

#### Acceptance Criteria

1. WHEN `ReshapeOp::executeCPU` completes successfully, THE ReshapeOp SHALL produce an output tensor whose `numElements()` equals the input tensor's `numElements()`.
2. WHEN `ReshapeOp::executeCPU` completes successfully, THE ReshapeOp SHALL copy the input bytes verbatim to the output (memcpy preserving the row-major bit pattern).
3. THE ReshapeOp SHALL read the shape tensor's data at execute time and resolve wildcards against the input shape at that moment; THE ReshapeOp SHALL NOT bake the output shape at construction time.

### Requirement 16: Common Operator Contract — Umbrella Header Registration

**User Story:** As a developer, I want the three new operators to be reachable through the `forgert.h` umbrella header, so that downstream code can include them with a single `#include`.

#### Acceptance Criteria

1. THE `forgert.h` umbrella header SHALL include `include/forgert/operator/averagepool2d.h`, `include/forgert/operator/dropout.h`, and `include/forgert/operator/reshape.h`.
2. WHEN a translation unit includes `forgert.h`, THE ForgeRT headers SHALL expose the classes `AveragePool2DOp`, `DropoutOp`, and `ReshapeOp`.

### Requirement 17: Common Operator Contract — Operator Interface

**User Story:** As a developer, I want the three new operators to follow the same interface as existing operators, so that they integrate uniformly with the graph, scheduler, and executor.

#### Acceptance Criteria

1. THE AveragePool2DOp, DropoutOp, and ReshapeOp SHALL each inherit from `Operator` and implement `name()`, `clone()`, `supportsBackend()`, `inferOutputShapes()`, and `executeCPU()`.
2. THE AveragePool2DOp SHALL return the name `"AveragePool2D"`, THE DropoutOp SHALL return the name `"Dropout"`, and THE ReshapeOp SHALL return the name `"Reshape"` from their respective `name()` methods.
3. THE AveragePool2DOp, DropoutOp, and ReshapeOp SHALL each provide a `clone()` method returning a `std::unique_ptr<Operator>` that owns a deep copy of the operator and its parameters.

### Requirement 18: SqueezeNet — Operator Set Completeness

**User Story:** As a developer, I want the implementation to support exactly the 7 operator types that SqueezeNet 1.1 uses, so that the model can be imported and executed without missing-operator failures.

#### Acceptance Criteria

1. WHEN the SqueezeNet 1.1 ONNX graph (opset 7) is loaded by the Importer, THE Importer SHALL successfully map every node to one of: `Conv`, `Relu`, `Concat`, `MaxPool`, `Dropout`, `AveragePool`, or `Reshape`.
2. THE ForgeRT operator set used by SqueezeNet 1.1 SHALL consist of exactly 7 operator types: `Conv2DOp`, `ReLUOp`, `ConcatOp`, `MaxPool2DOp`, `DropoutOp`, `AveragePool2DOp`, and `ReshapeOp`.
3. WHEN the SqueezeNet 1.1 graph is inspected, THE graph SHALL contain no operators outside the 7 listed types (no `Flatten`, `GlobalAveragePool`, `Transpose`, `Gemm`, or other shape ops beyond `Reshape`).

### Requirement 19: SqueezeNet — Reshape Shape Tensor Is Wildcard

**User Story:** As a developer, I want the implementation to handle the actual SqueezeNet Reshape shape tensor `[0, -1]`, so that the output `[1, 1000]` is produced correctly for any input `[1, 1000, 1, 1]`.

#### Acceptance Criteria

1. WHEN the SqueezeNet 1.1 graph is inspected, THE Reshape node's shape initializer SHALL be the 2-element tensor `[0, -1]`, not the literal `[1, 1000]`.
2. WHEN `ReshapeOp` is executed on input shape `[1, 1000, 1, 1]` with shape tensor `[0, -1]`, THE ReshapeOp SHALL produce output shape `[1, 1000]` (dimension 0 copies input dim 0 = 1; dimension 1 infers `1000` from the remaining elements).

### Requirement 20: SqueezeNet — Dropout Attribute

**User Story:** As a developer, I want the implementation to handle the SqueezeNet Dropout attribute correctly, so that the inference output is the identity of the input.

#### Acceptance Criteria

1. WHEN the SqueezeNet 1.1 Dropout node is inspected, THE Dropout node SHALL carry a `ratio = 0.5` attribute and no `training_mode` attribute (opset 7).
2. WHEN the Importer loads the SqueezeNet 1.1 Dropout node, THE Importer SHALL accept the `ratio = 0.5` attribute and SHALL NOT raise an error related to `training_mode`.

### Requirement 21: SqueezeNet — AveragePool Attributes

**User Story:** As a developer, I want the implementation to handle the SqueezeNet AveragePool attributes exactly, so that the pooling output matches ONNX Runtime.

#### Acceptance Criteria

1. WHEN the SqueezeNet 1.1 AveragePool node is inspected, THE AveragePool node SHALL carry `kernel_shape = [13, 13]`, `strides = [13, 13]`, and `pads = [0, 0, 0, 0]`.
2. WHEN `AveragePool2DOp` is executed on a SqueezeNet-shaped input `[1, 1000, 13, 13]` with `kernel = [13, 13]`, `stride = [13, 13]`, `pads = [0, 0, 0, 0]`, THE AveragePool2DOp SHALL produce output shape `[1, 1000, 1, 1]`.

### Requirement 22: SqueezeNet — Initializer Data Types and Sizes

**User Story:** As a developer, I want the implementation to handle the verified SqueezeNet initializers, so that weights and the reshape shape tensor are loaded correctly.

#### Acceptance Criteria

1. WHEN the SqueezeNet 1.1 graph is inspected, THE graph SHALL contain exactly 52 `float32` weight and bias initializers and exactly 1 `int64` initializer (the Reshape shape tensor).
2. WHEN the Importer loads the single `int64` initializer `reshape_attr_tensor118`, THE Importer SHALL convert its two `int64` values `[0, -1]` to `Int32` and load them into an `Int32` tensor of shape `[2]`.
3. WHEN the SqueezeNet 1.1 model file is inspected, THE total size SHALL be approximately 4.95 MB, comprising 4.71 MB of `float32` weights and 16 bytes of `int64` shape data.

### Requirement 23: Importer — Tool Architecture (Design Only)

**User Story:** As a developer, I want the Importer to be split into a build-time Python writer and a small C++ reader, so that the ForgeRT runtime has no protobuf dependency.

#### Acceptance Criteria

1. WHERE Phase 8 scope applies, THE Importer design SHALL consist of a Python tool `tools/onnx_to_fgx.py` that uses the `onnx` package to read an ONNX model and emit a `.fgx` file, and a C++ reader `src/import/fgx_reader.cpp` that parses a `.fgx` file and builds a `forgert::Graph`.
2. THE `Importer_Writer` SHALL be a Python script located at `tools/onnx_to_fgx.py` and SHALL use the `onnx` Python package to inspect the input model.
3. THE `Importer_Reader` SHALL be a C++ module located at `src/import/fgx_reader.cpp` and SHALL NOT link against any protobuf or ONNX library at runtime.
4. THE `Importer_Reader` SHALL be approximately 200 lines of C++ code, simple enough to audit by hand.
5. WHERE Phase 8 scope applies, THE Importer SHALL be design-only; no implementation of `tools/onnx_to_fgx.py` or `src/import/fgx_reader.cpp` is required in Phase 8.

### Requirement 24: Importer — FGX Format Contract

**User Story:** As a developer, I want the Python writer and C++ reader to agree exactly on the `.fgx` binary layout, so that a `.fgx` file produced by the writer is parseable by the reader without ambiguity.

#### Acceptance Criteria

1. THE `Importer_Writer` and `Importer_Reader` SHALL agree exactly on the `.fgx` binary layout, including the magic header, section ordering, field widths, endianness, tensor names, shapes, data types, and weight bytes.
2. THE `.fgx` format SHALL begin with a magic header that the `Importer_Reader` validates on open; IF the magic does not match, THEN THE `Importer_Reader` SHALL fail fast with an "incompatible .fgx file" error.
3. THE `.fgx` format SHALL include a `GRAPH` section (inputs, outputs, nodes, attributes) and an `INITIALIZERS` section (named tensors with shapes, data types, and raw bytes).
4. WHEN the `Importer_Writer` emits a tensor and the `Importer_Reader` parses that same tensor, THE two sides SHALL produce identical tensor names, shapes, data types, and byte-for-byte identical weight data.

### Requirement 25: Importer — Unsupported Operator Fail-Fast

**User Story:** As a developer, I want the Importer to fail fast on unsupported operators, so that I know exactly which node and which operator type caused the failure.

#### Acceptance Criteria

1. WHEN the Importer encounters an ONNX node whose `op_type` is not in the supported set (`Conv`, `Relu`, `Concat`, `MaxPool`, `Dropout`, `AveragePool`, `Reshape`), THEN THE Importer SHALL fail fast with an error message that lists the supported op set and identifies the offending node by its index in the ONNX graph.
2. THE Importer SHALL NOT silently skip unsupported operators and SHALL NOT fall back to a partial graph execution.

### Requirement 26: Importer — int64 to int32 Conversion with Bounds Check

**User Story:** As a developer, I want the Importer to convert `int64` initializers to `Int32` safely, so that overflow cannot silently corrupt shape or weight data.

#### Acceptance Criteria

1. WHEN the Importer loads an `int64` initializer, THE Importer SHALL convert each value to `Int32`.
2. WHEN any `int64` value being converted exceeds `INT32_MAX` or is less than `INT32_MIN`, THEN THE Importer SHALL fail fast with an error message indicating that an `int64` value overflowed on conversion to `int32`.
3. WHEN the Importer converts the SqueezeNet `reshape_attr_tensor118` values `[0, -1]` from `int64` to `Int32`, THE Importer SHALL succeed without error.

### Requirement 27: Importer — Shape Validation Against ONNX value_info

**User Story:** As a developer, I want the Importer to cross-check `inferOutputShapes` against ONNX `value_info`, so that attribute-extraction bugs are caught early rather than producing silent numerical errors.

#### Acceptance Criteria

1. WHEN the Importer loads an ONNX node whose outputs have a `value_info` entry, THE Importer SHALL compare the shape produced by the corresponding ForgeRT operator's `inferOutputShapes()` against the shape recorded in the ONNX `value_info`.
2. WHEN the `inferOutputShapes()` result does not match the ONNX `value_info` shape, THEN THE Importer SHALL fail fast with an error message reporting both shapes and identifying the offending node.
3. WHEN an ONNX node's output has no `value_info` entry, THE Importer SHALL fall back to the ForgeRT operator's `inferOutputShapes()` result and SHALL log a warning indicating the missing `value_info`.

### Requirement 28: End-to-End Validation — Numerical Equivalence

**User Story:** As a user, I want ForgeRT's SqueezeNet 1.1 inference to match ONNX Runtime numerically, so that I can trust the ForgeRT implementation is correct.

#### Acceptance Criteria

1. WHEN 10 random inputs `x_i ~ N(0, 1)` of shape `[1, 3, 224, 224]` are fed through both ForgeRT and ONNXRuntime (CPU) on the SqueezeNet 1.1 model, THE ForgeRT output and ONNXRuntime output SHALL satisfy `max |ForgeRT(x_i) - ONNXRuntime(x_i)| < 1e-3` for each of the 10 inputs.
2. THE end-to-end validation SHALL require all 10 inputs to satisfy the `< 1e-3` bound; passing on average or on a subset SHALL NOT be sufficient.
3. THE end-to-end validation SHALL compare numerical correctness only and SHALL NOT compare wall-clock latency, memory usage, or throughput against ONNX Runtime (performance comparison is Phase 9).

### Requirement 29: End-to-End Validation — argmax Match

**User Story:** As a user, I want ForgeRT to predict the same class as ONNX Runtime for the same input, so that the model's behavior is preserved end-to-end.

#### Acceptance Criteria

1. WHEN each of the 10 random inputs is fed through both ForgeRT and ONNXRuntime on SqueezeNet 1.1, THE `argmax(ForgeRT(x_i))` SHALL equal `argmax(ONNXRuntime(x_i))` for each of the 10 inputs.
2. THE end-to-end validation SHALL require all 10 argmax comparisons to match; matching on a subset SHALL NOT be sufficient.

### Requirement 30: End-to-End Validation — Reference Runtime

**User Story:** As a developer, I want the validation to use ONNX Runtime CPU as the reference, so that the comparison is against a well-established, deterministic inference runtime.

#### Acceptance Criteria

1. THE end-to-end validation SHALL use the Python `onnxruntime` package running on CPU as the reference runtime.
2. THE reference outputs SHALL be the 1000-class logits produced by `onnxruntime` for each of the 10 random inputs.
3. THE random inputs SHALL be drawn from `N(0, 1)` with shape `[1, 3, 224, 224]` and SHALL be identical between the ForgeRT run and the ONNXRuntime run.

### Requirement 31: Non-Goal — Preserve Existing Tests

**User Story:** As a developer, I want Phase 8 to preserve all 24 existing tests, so that no regression is introduced.

#### Acceptance Criteria

1. WHERE Phase 8 scope applies, THE ForgeRT test suite SHALL preserve all 24 existing tests with no weakening of their assertions.
2. WHEN the Phase 8 test suite is executed, THE suite SHALL include the 24 existing tests plus the new operator tests, and all SHALL pass.

### Requirement 32: Non-Goal — Preserve CPU-Only Build Path

**User Story:** As a developer, I want SqueezeNet CPU inference to remain buildable without a CUDA toolchain, so that the CPU-only build path is preserved.

#### Acceptance Criteria

1. WHERE Phase 8 scope applies, THE ForgeRT build SHALL remain buildable on a CPU-only machine without requiring a CUDA toolchain to run SqueezeNet CPU inference.
2. WHEN the three new operators are added with CPU-only backends, THE build SHALL NOT introduce a new CUDA requirement for SqueezeNet CPU inference.

### Requirement 33: Non-Goal — No Importer Implementation in Phase 8

**User Story:** As a developer, I want Phase 8 to scope the Importer to design only, so that implementation can proceed in a focused follow-up phase.

#### Acceptance Criteria

1. WHERE Phase 8 scope applies, THE Importer SHALL be design-only; Phase 8 SHALL NOT deliver an implementation of `tools/onnx_to_fgx.py` or `src/import/fgx_reader.cpp`.
2. WHEN Phase 8 is complete, THE design for the Importer SHALL be documented in `design.md` and the Importer requirements SHALL be captured in this `requirements.md`.

### Requirement 34: Non-Goal — No LLM/Ollama Integration

**User Story:** As a developer, I want Phase 8 to exclude LLM and Ollama integration, so that the scope remains focused on classical CNN inference.

#### Acceptance Criteria

1. WHERE Phase 8 scope applies, THE ForgeRT codebase SHALL NOT integrate any LLM or Ollama component.
2. THE Phase 8 requirements SHALL NOT include any feature depending on LLM or Ollama services.

### Requirement 35: Non-Goal — No New Runtime Dependencies

**User Story:** As a developer, I want the ForgeRT runtime to remain dependency-free, so that the binary stays easy to build and audit.

#### Acceptance Criteria

1. WHERE Phase 8 scope applies, THE ForgeRT runtime SHALL NOT add any new runtime dependency as a result of the three new operators or the Importer design.
2. THE Python `onnx` and `onnxruntime` packages SHALL be build-time and test-time only; THE ForgeRT runtime SHALL NOT link against either at runtime.
3. THE `Importer_Reader` SHALL NOT link against protobuf or any ONNX library at runtime.

### Requirement 36: Non-Goal — SqueezeNet Compatibility Claim Gated on Validation

**User Story:** As a developer, I want SqueezeNet compatibility to be claimed only after end-to-end validation passes, so that the claim is backed by evidence.

#### Acceptance Criteria

1. THE ForgeRT project SHALL NOT claim SqueezeNet 1.1 compatibility until the end-to-end validation described in Requirements 28 and 29 passes on all 10 random inputs.
2. WHEN the end-to-end validation has not yet been executed, THE project documentation SHALL state that SqueezeNet compatibility is unverified.

# ONNX Import and SqueezeNet 1.1 Validation

ForgeRT runs a restricted subset of ONNX through a two-step path:

```
model.onnx --tools/onnx_to_fgx.py--> model.fgx --forgert_import (C++)--> Graph --> CPU execution
```

- **Converter** (`tools/onnx_to_fgx.py`): uses the `onnx` package; build-time only.
- **Reader** (`src/import/fgx_reader.cpp`): strict, dependency-free parser for the
  binary format in [`FGX_FORMAT.md`](FGX_FORMAT.md). Never returns a partial model.
- **Importer** (`src/import/importer.cpp`): operator/attribute allowlist, ONNX opset 7
  only, builds a `forgert::Graph`.
- **Runner** (`apps/forgert_run`): `forgert_run model.fgx input.bin output.bin`
  (raw little-endian float32 in and out).

## Supported subset

| ONNX op (opset 7) | ForgeRT operator | Accepted attributes and limits |
|---|---|---|
| Conv | `Conv2DOp` | `kernel_shape` (must match weight dims), `strides`, `pads` (symmetric only), `group` = 1, `dilations` = 1 |
| Relu | `ReLUOp` | none |
| Concat | `ConcatOp` | `axis` required (negative normalized), at least 2 inputs |
| MaxPool | `MaxPool2DOp` | `kernel_shape` required, `strides`, `pads` (symmetric) |
| AveragePool | `AveragePool2DOp` | as MaxPool, plus `count_include_pad` (0 or 1) |
| Dropout | `DropoutOp` | `ratio` (ignored); exactly one output, so a training mask is rejected |
| Reshape | `ReshapeOp` (fixed target) | shape must be a constant int64 initializer; values are range-checked to int32 |

Anything else fails with the node index, node name, op type, and the list of
supported operators. There is no fallback and no partial graph. Other rejections:
opset other than 7, `auto_pad` other than `NOTSET`, rank-0 tensors, symbolic
dimensions, external tensor data, non-default operator domains, non-float32
graph inputs/outputs.

## Design notes

- **Weights are graph inputs.** `Graph` has no constants. Each float32 initializer is
  registered as an extra graph input after the model's own inputs; `ImportedModel`
  owns the tensors and appends them on every `run()`. The existing optimizer and
  scheduler work unchanged (tested). A real constant node would need changes to
  `Graph::execute`, the optimizer and the scheduler. The weights are `freeze()`d at
  import, so a `ScheduledGraph` kept alive across runs uploads them to the GPU once and
  keeps them resident (`ImportedModel::graphInputs()` builds the input vector); see
  the *Persistent GPU weights* section of the README. `ImportedModel::run()` itself is
  CPU-only and unchanged.
- **Reshape.** `Tensor` shapes are immutable and `Graph` allocates exactly what
  `inferOutputShapes()` returns, so the importer uses `ReshapeOp`'s fixed-target mode.
  SqueezeNet's `[0, -1]` target resolves to exactly `[1, 1000]`.
- **Shape cross-check.** The converter stores ONNX shape-inference results (the source
  model has none). The importer requires every node output to match them exactly.
- **int64.** Only a Reshape shape initializer may be int64; it is narrowed to int32 with
  a range check (`narrowInt64ToInt32`). SqueezeNet's only int64 tensor is `[0, -1]`.

## Validation protocol (`tools/validate_squeezenet.py`)

- Model: `models/squeezenet1.1-7.onnx`, SHA-256 `1eeff551...4088`
  (`python tools/fetch_squeezenet.py` downloads and verifies it; the file is git-ignored).
- Reference: ONNX Runtime CPU, default graph optimization.
- Inputs: 10 tensors of `[1,3,224,224]` float32 drawn from N(0,1) with
  `numpy.random.default_rng(20261020)`, identical for both runtimes.
- Pass criteria, **per input** (not averaged): `max |ForgeRT - ORT| < 1e-3` and equal argmax.
- Relative error and ORT's top-1/top-2 margin are printed as diagnostics only.
- A missing package or model exits 77, which CTest reports as *Skipped*, never *Passed*.

### Result (2026-10-20, i5-9300H, onnx 1.23.2, onnxruntime 1.29.0, numpy 2.4.3)

| Input | max abs diff | argmax (ForgeRT / ORT) | ORT top-1 margin |
|---|---|---|---|
| 0 | 3.34e-06 | 539 / 539 | 0.170 |
| 1 | 3.10e-06 | 539 / 539 | 0.054 |
| 2 | 4.77e-06 | 630 / 630 | 0.173 |
| 3 | 3.34e-06 | 630 / 630 | 0.255 |
| 4 | 3.82e-06 | 735 / 735 | 0.042 |
| 5 | 5.72e-06 | 539 / 539 | 0.445 |
| 6 | 5.25e-06 | 539 / 539 | 0.089 |
| 7 | 3.82e-06 | 539 / 539 | 0.231 |
| 8 | 3.82e-06 | 735 / 735 | 0.133 |
| 9 | 3.82e-06 | 539 / 539 | 0.023 |

10/10 inputs pass; worst difference 5.7e-06, about 175x below the limit. A negative
control (`--atol 1e-7`) fails as expected.

### What this does and does not show

- It shows that, for these 10 random inputs, ForgeRT's CPU execution of the imported
  SqueezeNet 1.1 matches ONNX Runtime to float32 round-off and predicts the same class.
- Random noise inputs are not natural images, and the predicted classes are arbitrary.
  No accuracy on a real dataset was measured.
- Only the CPU path was validated. The CUDA path was not run on the imported model.
- Only SqueezeNet 1.1 opset 7 was tested. Other models, opsets, and batch sizes
  other than 1 are untested.
- Timing is not a reliable baseline. An earlier, uncontrolled observation was about 1.3 s per
  inference (machine state not recorded); a later measured run on battery under load gave a 5.75 s
  median. Both are single-threaded naive convolution and depend heavily on machine state; neither is
  representative. See [`SQUEEZENET_CPU_LATENCY.md`](SQUEEZENET_CPU_LATENCY.md). ONNX Runtime timing
  was not measured and no speed comparison is claimed.

## Tests

| CTest name | What it covers |
|---|---|
| `ReshapeTest` | fixed-target and shape-tensor modes |
| `OnnxToFgxConverterTest` | converter on synthetic models (python) |
| `FgxReaderTest` | valid parse (20 checks) plus 56 malformed-file cases |
| `ImporterTest` | 7-op model vs hand reference, 25 error paths, optimizer/scheduler |
| `FgxCrossLanguageSyntheticTest` / `...SqueezeNetTest` | Python manifest == C++ `fgx_dump` manifest |
| `SqueezeNetE2ETest` | the validation above (label `slow;e2e`, about 15 s) |

Malformed-file cases not exercised: a file over 1 GiB (needs a 1 GiB buffer) and a
big-endian host (cannot run on x86); both are single comparisons at the top of `parse()`.
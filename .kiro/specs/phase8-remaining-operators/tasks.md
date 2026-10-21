# Implementation Plan: Phase 8 — Remaining Operators (AveragePool, Dropout, Reshape)

## Overview

This plan implements the three remaining CPU operators required to complete the SqueezeNet 1.1 output path: `AveragePool2DOp`, `DropoutOp`, and `ReshapeOp`. Each operator follows the existing `Conv2DOp` / `ConcatOp` / `MaxPool2DOp` contract:

- Header in `include/forgert/operator/`
- Inherit from `Operator`; implement `name()`, `clone()`, `supportsBackend()`, `inferOutputShapes()`, `executeCPU()`
- CPU-only backend (no CUDA this phase, per design justification)
- Use the `validateDevice`, `validateDtype`, `validateInputCount` helpers
- Registered in `tests/CMakeLists.txt` (`add_executable` + `add_test`)
- Exposed via the `include/forgert/forgert.h` umbrella header
- One test file per operator in `tests/`

Implementation order is AveragePool → Dropout → Reshape. The three operators have no logical dependency on each other and could be written in parallel, but they share two files (`tests/CMakeLists.txt` and `include/forgert/forgert.h`), so their registration steps are serialized in the dependency graph and their commits are made sequentially for hygiene.

Each operator task includes a build + run verification step on its own test executable. The final regression task verifies all 27 tests (24 existing + 3 new) pass together.

Phase 8 Part C (ONNX importer: `tools/onnx_to_fgx.py` and `src/import/fgx_reader.cpp`), end-to-end SqueezeNet validation, and any CUDA backend work are explicitly OUT OF SCOPE for this task list — no such tasks appear below.

All code is C++ (the design specifies C++ headers and the existing operators are C++). Testing uses the project's existing plain-C++ `ok()` helper convention (no Google Test); property tests are implemented as loops over random inputs checked against a brute-force reference, consistent with that style.

## Tasks

- [x] 1. Implement AveragePool2DOp (CPU backend)
  - [x] 1.1 Write `include/forgert/operator/averagepool2d.h`
    - Follow the `MaxPool2DOp` pattern: define `AvgPool2DParams { kH, kW, stride_h, stride_w, pad_h, pad_w, count_include_pad }`; `AvgPool2DOp` inherits `Operator`.
    - Implement `name()` → `"AveragePool2D"`, `clone()` (deep copy of params), `supportsBackend()` → CPU only, `inferOutputShapes()` (ONNX floor-mode `H_out = floor((H_in + 2*pad_h - kH)/stride_h) + 1`, same for W; require 4-D NCHW input; preserve N and C), and `executeCPU()` sliding-window mean with `count_include_pad` handling and per-`(n, c)` independence.
    - Validate params at construction/shape-inference: kernel > 0, stride > 0, pad >= 0 (throw `std::invalid_argument` on invalid).
    - _Requirements: 1.1, 1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 5.1, 5.2, 5.3, 17.1, 17.2, 17.3_

  - [x] 1.2 Write `tests/test_averagepool2d.cpp` (focused numerical + error tests)
    - SqueezeNet shape `[1,1000,13,13]` 13x13/s13/p0 → `[1,1000,1,1]`; known-values 2x2 average on a 4x4 input hand-verified; `count_include_pad=true` vs `false` with padding > 0; multi-channel and multi-batch independence; output-size formula over 4 configs; error cases: 3-D input throws, wrong input count throws, non-positive kernel/stride throws.
    - Use the existing `ok()` helper convention from `tests/test_maxpool2d.cpp` (no external test framework).
    - _Requirements: 2.4, 2.5, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 5.1, 5.2, 5.3, 21.1, 21.2_

  - [x] 1.3 Register `test_averagepool2d` in `tests/CMakeLists.txt` and add to `include/forgert/forgert.h`
    - Add `add_executable(test_averagepool2d test_averagepool2d.cpp)` + `target_include_directories` + `target_link_libraries(test_averagepool2d PRIVATE forgert)` mirroring the `test_maxpool2d` block; add `add_test(NAME AveragePool2DTest COMMAND test_averagepool2d)`.
    - Add `#include "forgert/operator/averagepool2d.h"` to the Operators section of `forgert.h` (alongside `maxpool2d.h`).
    - _Requirements: 16.1, 16.2_

  - [x] 1.4 Build and run `test_averagepool2d`; confirm it passes
    - Reconfigure CMake, build the `test_averagepool2d` target, run `ctest -R AveragePool2DTest` (or the executable directly), and confirm PASS before proceeding.
    - _Requirements: 1.1, 1.2, 17.1_

  - [ ]* 1.5 Write property test for AveragePool2D mean correctness
    - **Property 1: AveragePool2D mean correctness** — for random NCHW float32 inputs and random valid `AvgPool2DParams`, assert each output element equals the window mean per the `count_include_pad` rule and that output dims satisfy the ONNX floor-mode formula; compare against a brute-force reference computed in the test.
    - **Validates: Requirements 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 5.1, 5.2, 5.3**
    - Add the property cases to `tests/test_averagepool2d.cpp`, rebuild, rerun `test_averagepool2d`, confirm PASS.

- [x] 2. Implement DropoutOp (CPU backend)
  - [x] 2.1 Write `include/forgert/operator/dropout.h`
    - Define `DropoutParams { float ratio = 0.5f }` (stored, unused at inference); `DropoutOp` inherits `Operator`.
    - Implement `name()` → `"Dropout"`, `clone()`, `supportsBackend()` → CPU only, `inferOutputShapes()` → identity (output shape == input shape, rank and dtype preserved), `executeCPU()` → `memcpy` of input bytes to output (no mask, no `1/(1-ratio)` scaling).
    - _Requirements: 6.1, 6.2, 7.1, 7.2, 7.3, 7.4, 8.1, 8.2, 8.3, 8.4, 17.1, 17.2, 17.3_

  - [x] 2.2 Write `tests/test_dropout.cpp` (focused numerical + error tests)
    - Byte-for-byte identity across 1-D, 2-D, and 4-D shapes; random non-trivial float values (assert no zeroing and no `1/(1-ratio)` scaling); `ratio` varied (0.0, 0.5, 0.9) with identical output; error cases: wrong input count, output shape mismatch.
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 8.1, 8.2, 8.3, 8.4, 20.1, 20.2_

  - [x] 2.3 Register `test_dropout` in `tests/CMakeLists.txt` and add to `include/forgert/forgert.h`
    - Add `add_executable(test_dropout ...)` + link `forgert`; add `add_test(NAME DropoutTest COMMAND test_dropout)`; add `#include "forgert/operator/dropout.h"` to `forgert.h`.
    - _Requirements: 16.1, 16.2_

  - [x] 2.4 Build and run `test_dropout`; confirm it passes
    - Reconfigure, build `test_dropout`, run `ctest -R DropoutTest`, confirm PASS.
    - _Requirements: 6.1, 6.2, 17.1_

  - [ ]* 2.5 Write property test for Dropout inference identity
    - **Property 2: Dropout inference identity** — for random shapes and random float data, assert `output[i] == input[i]` byte-for-byte for every element regardless of `ratio`, and that output shape equals input shape.
    - **Validates: Requirements 8.1, 8.2, 8.3, 8.4**
    - Add the property cases to `tests/test_dropout.cpp`, rebuild, rerun `test_dropout`, confirm PASS.

- [x] 3. Implement ReshapeOp (CPU backend)
  - [x] 3.1 Write `include/forgert/operator/reshape.h`
    - `ReshapeOp` inherits `Operator` (no params — target shape is the 2nd input tensor).
    - Implement `name()` → `"Reshape"`, `clone()`, `supportsBackend()` → CPU only, `inferOutputShapes()` (read the `Int32` shape tensor; resolve `0` = copy input dim `i`, `-1` = infer from remaining elements; enforce at most one `-1`; enforce that a `0` at dim `i` has a matching input dim; verify resolved product == input `numElements()`), `executeCPU()` → `memcpy` input bytes into the output tensor with the resolved shape (read shape tensor at execute time, not construction time).
    - Validate exactly two inputs (data Float32 + shape Int32) via `validateInputCount` / `validateDevice` / `validateDtype`.
    - _Requirements: 10.1, 10.2, 11.1, 11.2, 11.3, 11.4, 11.5, 12.1, 12.2, 13.1, 13.2, 14.1, 14.2, 14.3, 15.1, 15.2, 15.3, 17.1, 17.2, 17.3_

  - [x] 3.2 Write `tests/test_reshape.cpp` (focused numerical + error tests)
    - SqueezeNet case `[1,1000,1,1]` + shape `[0,-1]` → `[1,1000]`; wildcard `0`: `[2,3,4]` + `[0,2,6]` → `[2,2,6]`; wildcard `-1`: `[2,3,4]` + `[2,-1]` → `[2,12]`; both wildcards: `[2,3,4]` + `[0,-1]` → `[2,12]`; literal `[6]` + `[2,3]` → `[2,3]`; data bytes unchanged after reshape; error cases: total element-count mismatch, more than one `-1`, `0` with no matching input dim, wrong input count, wrong shape-tensor dtype.
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 13.1, 13.2, 14.1, 14.2, 14.3, 15.1, 15.2, 15.3, 19.1, 19.2_

  - [x] 3.3 Register `test_reshape` in `tests/CMakeLists.txt` and add to `include/forgert/forgert.h`
    - Add `add_executable(test_reshape ...)` + link `forgert`; add `add_test(NAME ReshapeTest COMMAND test_reshape)`; add `#include "forgert/operator/reshape.h"` to `forgert.h`.
    - _Requirements: 16.1, 16.2_

  - [x] 3.4 Build and run `test_reshape`; confirm it passes
    - Reconfigure, build `test_reshape`, run `ctest -R ReshapeTest`, confirm PASS.
    - _Requirements: 10.1, 10.2, 17.1_

  - [ ]* 3.5 Write property test for Reshape element-count preservation
    - **Property 3: Reshape element count preservation** — for random input shapes and valid shape tensors containing `0` / `-1` wildcards, assert `output.numElements() == input.numElements()`, that the data bytes are unchanged, and that the at-most-one-`-1` and element-count invariants hold; compare against a brute-force reference resolver.
    - **Validates: Requirements 13.1, 13.2, 14.1, 14.2, 14.3, 15.1, 15.2**
    - Add the property cases to `tests/test_reshape.cpp`, rebuild, rerun `test_reshape`, confirm PASS.

- [x] 4. Regression, commit, and documentation wrap-up
  - [x] 4.1 Run the full regression suite; confirm all 27 tests pass
    - Run `ctest` from the build directory; confirm the 24 existing tests plus `AveragePool2DTest`, `DropoutTest`, `ReshapeTest` all pass (27 total). Confirm no existing test was weakened or broken.
    - Ensure all tests pass, ask the user if questions arise.
    - _Requirements: 31.1, 31.2_

  - [x] 4.2 Commit the Phase 8 operator work as focused commits
    - Make one focused commit per operator, in order AveragePool → Dropout → Reshape (header + test + registration together), or group logically; the final commit message references the regression-suite confirmation. Do not commit importer files (none exist this phase).
    - _Requirements: 31.1, 31.2_

  - [x] 4.3 Update documentation (operator-support table + limitations)
    - Update the README operator-support table to list `AveragePool2D`, `Dropout`, `Reshape` as CPU-only.
    - Note Phase 8 scope: CUDA backends deferred; importer is design-only this phase.
    - State that SqueezeNet 1.1 compatibility is unverified until end-to-end validation runs.
    - _Requirements: 33.1, 33.2, 36.1, 36.2_

## Notes

- Tasks marked with `*` are optional property-test sub-tasks and may be skipped for a faster MVP; the required numerical + error test files (1.2, 2.2, 3.2) and per-operator verify steps (1.4, 2.4, 3.4) are mandatory.
- The three operators have no logical dependency on each other (headers and test files are written in parallel in waves 0–1), but they share `tests/CMakeLists.txt` and `include/forgert/forgert.h`, so their registration steps (1.3, 2.3, 3.3) are serialized in the dependency graph. Per-operator verify steps are also separated from any registration step to avoid CMake-configure / header-include races. Commits are made sequentially AveragePool → Dropout → Reshape for hygiene.
- Each operator's verify step (1.4, 2.4, 3.4) builds and runs its own test executable; the final regression task (4.1) runs all 27 tests together.
- Out of scope (per Phase 8 Part C design-only): `tools/onnx_to_fgx.py`, `src/import/fgx_reader.cpp`, end-to-end SqueezeNet validation, and any CUDA backend work. These are intentionally NOT tasks here.
- All code is C++ (the design specifies C++ headers and the existing operators are C++); no language selection was needed.
- Testing uses the project's existing plain-C++ `ok()` helper convention (no Google Test); property tests are implemented as loops over random inputs with a brute-force reference, consistent with that style.

## Completion Record

Checked against the repository during Phase 9 housekeeping (HEAD `81334c0`). The task text
above is unchanged; only checkboxes were updated, and this section was added.

| Task | Evidence |
|---|---|
| 1.1 - 1.4 | `averagepool2d.h` and `test_averagepool2d.cpp` added in `044fd39`; registered in `tests/CMakeLists.txt` and `forgert.h`; `AveragePool2DTest` passes in the current `ctest` run (33/33). |
| 2.1 - 2.4 | `dropout.h` and `test_dropout.cpp` added in `d1de781`; registered; `DropoutTest` passes. |
| 3.1 - 3.4 | `reshape.h` and `test_reshape.cpp` added in `507ade6`; registered; `ReshapeTest` passes. |
| 4.1 | 27/27 passed when the operators landed (recorded in the `507ade6` message); those 27 tests are all part of the current 33/33. |
| 4.2 | Three focused commits `044fd39`, `d1de781`, `507ade6`; the last one records the 27-test result. |
| 4.3 | `c882424` updated the README: the three operators as CPU-only, CUDA deferred, importer not built, SqueezeNet unverified. The README was later updated again once the importer shipped. |
| 1 - 4 (parents) | Checked because every non-optional sub-task is done. The optional `*` sub-tasks may be skipped per the Notes. |

**Deviation in 3.1.** In shape-tensor mode `inferOutputShapes()` cannot read the shape
tensor's values (it only receives shapes), so it returns a flat shape with the right element
count; the `0` / `-1` resolution and every validation run in `executeCPU()`. `d4ce41b` later
added a fixed-target mode whose `inferOutputShapes()` resolves the exact shape.

**Left unchecked: 1.5\*, 2.5\*, 3.5\*.** No randomized property loop against a brute-force
reference exists. `test_averagepool2d.cpp` and `test_reshape.cpp` generate no random inputs.
`test_dropout.cpp` has one seeded random-data case on a fixed `[1,1000]` shape, but nothing
over random shapes or ratios, so Property 2 as written is not covered.

**Out-of-scope items above have since been done** by later work: the importer, converter,
reader and SqueezeNet validation (`d4ce41b` to `81334c0`, see `docs/ONNX_IMPORT.md`).

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "2.1", "3.1"] },
    { "id": 1, "tasks": ["1.2", "2.2", "3.2"] },
    { "id": 2, "tasks": ["1.3"] },
    { "id": 3, "tasks": ["1.4"] },
    { "id": 4, "tasks": ["1.5"] },
    { "id": 5, "tasks": ["2.3"] },
    { "id": 6, "tasks": ["2.4"] },
    { "id": 7, "tasks": ["2.5"] },
    { "id": 8, "tasks": ["3.3"] },
    { "id": 9, "tasks": ["3.4"] },
    { "id": 10, "tasks": ["3.5"] },
    { "id": 11, "tasks": ["4.1"] },
    { "id": 12, "tasks": ["4.2"] },
    { "id": 13, "tasks": ["4.3"] }
  ]
}
```

# Implementation Plan: ONNX Importer Pilot

## Overview

Pilot task used to verify that spec-task-execution dispatch works. It is a review
task only: it reads the draft format and writes one findings file.

## Tasks

- [x] 1. Review the draft .fgx v1 format for binary safety
  - Read `.kiro/specs/onnx-importer-pilot/fgx-format-draft.md` and
    `include/forgert/tensor/tensor.h`, `include/forgert/tensor/dtype.h`.
  - Write findings to `.kiro/specs/onnx-importer-pilot/fgx-review.md`: ambiguities,
    missing bounds checks, overflow/truncation hazards, versioning gaps, and a list of
    malformed-file test cases the C++ reader must reject.
  - Do not create or modify any other file. Do not build, commit, or push.
  - **Outcome (recorded afterwards):** the `spec-task-execution` dispatch for this task did not complete and was abandoned (see `a1202f1`). The review was obtained from a `requirement-detailer` subagent run instead; its findings were saved as `fgx-review.md`. The deliverable exists and its findings were applied in `docs/FGX_FORMAT.md` and used to build the reader's malformed-file tests, so the task is marked done rather than failed.
  - **Deviation from the task text:** `fgx-review.md` has three sections (ambiguities, missing safety checks, malformed-file cases), not the five requested. Versioning gaps appear among the ambiguities, and there is no separate MUST/SHOULD list; resolutions are marked `[adopted]` instead.

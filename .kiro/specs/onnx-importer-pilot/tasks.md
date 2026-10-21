# Implementation Plan: ONNX Importer Pilot

## Overview

Pilot task used to verify that spec-task-execution dispatch works. It is a review
task only: it reads the draft format and writes one findings file.

## Tasks

- [-] 1. Review the draft .fgx v1 format for binary safety
  - Read `.kiro/specs/onnx-importer-pilot/fgx-format-draft.md` and
    `include/forgert/tensor/tensor.h`, `include/forgert/tensor/dtype.h`.
  - Write findings to `.kiro/specs/onnx-importer-pilot/fgx-review.md`: ambiguities,
    missing bounds checks, overflow/truncation hazards, versioning gaps, and a list of
    malformed-file test cases the C++ reader must reject.
  - Do not create or modify any other file. Do not build, commit, or push.
# Bugfix Requirements Document

> **Status: Resolved in `ee43eac`** ("test(phase7): validate scheduler calibration and establish CUDA test stability").
> Test 7 in `tests/test_scheduler.cpp` now uses the measured ReLU values (0.40 ns/elem, 3.0x speedup, matching `scheduler.cpp`), and `LayerNormStressTest` is registered with CTest.
> No design or task list was generated for this spec; the fix landed directly in that commit. `326ad7e` later added the measured 20 us CUDA launch overhead to Test 7's expected-cost helper.
>
> The text below is the original defect description, kept unchanged as a historical record.

## Introduction

Test 7 in `tests/test_scheduler.cpp` ("Cost model verification — numbers match hand calculation") is failing because it uses hardcoded cost model values that were replaced with measured values during Phase 6 cost model calibration. The scheduler implementation in `src/scheduler/scheduler.cpp` was updated with measured values from GTX 1050 benchmarks, but Test 7 still uses the old estimated values for its hand calculations, causing a mismatch.

Additionally, the `test_layernorm_stress` executable is built but not registered with CTest, preventing it from being run as part of the test suite.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN Test 7 calculates expected CPU cost for ReLU THEN the system uses 0.18 ns/elem (old estimate) instead of 0.40 ns/elem (measured)

1.2 WHEN Test 7 calculates expected GPU speedup for ReLU THEN the system uses 5.0× (old estimate) instead of 3.0× (measured)

1.3 WHEN the test suite is run via CTest THEN the system does not execute test_layernorm_stress because it is not registered

### Expected Behavior (Correct)

2.1 WHEN Test 7 calculates expected CPU cost for ReLU THEN the system SHALL use 0.40 ns/elem (matching the measured value in scheduler.cpp)

2.2 WHEN Test 7 calculates expected GPU speedup for ReLU THEN the system SHALL use 3.0× (matching the measured value in scheduler.cpp)

2.3 WHEN the test suite is run via CTest THEN the system SHALL execute test_layernorm_stress as part of the registered tests

### Unchanged Behavior (Regression Prevention)

3.1 WHEN Test 7 verifies cost model calculations THEN the system SHALL CONTINUE TO validate that reported costs match hand calculations within tolerance

3.2 WHEN other scheduler tests (1-6, 8-9) execute THEN the system SHALL CONTINUE TO pass without modification

3.3 WHEN CTest runs the existing CUDA tests THEN the system SHALL CONTINUE TO execute all previously registered tests

3.4 WHEN the scheduler's cost model is used at runtime THEN the system SHALL CONTINUE TO use the measured values from scheduler.cpp (no change to implementation)

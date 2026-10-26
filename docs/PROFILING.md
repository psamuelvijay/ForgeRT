# ForgeRT — Measured Profiling & Scheduler Calibration

The scheduler (`SchedulingPolicy`) decides CPU vs GPU per operator from a cost
model. Its default profiles are **estimates** calibrated once on a GTX 1050 /
i5-9300H and clearly labelled as such in `src/scheduler/scheduler.cpp`. This
document describes the reproducible path for replacing those estimates with
measurements — and the discipline required before any measured value is
promoted into the compiled defaults.

## Pipeline overview

```
benchmarks/benchmark_profile.cu   ->  raw JSON          (measurements only)
tools/calibrate_profiles.py       ->  derived JSON      (scheduler-ready values)
                                  ->  C++ snippet       (paste-ready, reviewed)
(manual, reviewed)                ->  defaultProfiles() (promotion step)
```

Raw measurements and derived profiles are kept in **separate** files. The
scheduler is never auto-updated; promotion is a deliberate, reviewed edit.

## Cost model & units (what calibration feeds)

`SchedulingPolicy::decide()` computes, in **nanoseconds**:

```
cost_cpu = cpu_ns_per_elem * num_output_elems
cost_gpu = H2D(input_bytes)            [0 if inputs already resident on GPU]
         + CUDA_LAUNCH_OVERHEAD_NS     [constant in scheduler.cpp]
         + (cpu_ns_per_elem / gpu_speedup) * num_output_elems   [kernel compute]
         + D2H(output_bytes)
transfer term = bytes / transfer_bandwidth_bytes_per_ns   (bytes/ns)
decision = argmin(cost_cpu, cost_gpu), subject to cuda_supported and VRAM budget
```

So calibration must produce, per operator:
- `cpu_ns_per_elem` — CPU steady-state latency per output element.
- `gpu_speedup` — `cpu_kernel_ns_per_elem / gpu_kernel_ns_per_elem` (kernel-only;
  transfers are modelled separately, so speedup must NOT include transfer time).

…and two globals:
- `transfer_bandwidth_bytes_per_ns` — effective PCIe bandwidth.
- `cuda_launch_overhead_ns` — fixed per-launch cost (currently a constant in
  `scheduler.cpp`, not a profile field).

`gpu_speedup` is dimensionless, derived from kernel-only times, so there is no
double-counting of transfer cost (verified by `ProfileCalibrationTest`).

## 1. Collect measurements

Build and run the profiler (requires a CUDA GPU at runtime):

```bash
cmake --build build --target benchmark_profile
build/benchmarks/benchmark_profile.exe build/benchmarks/profile_raw.json
```

Methodology (see the file header for details):
- Warm-up before every series; CUDA context warmed first.
- Each measurement averages `inner_iters` back-to-back ops; this is repeated for
  `samples` independent series. Reported stats are median / min / max / mean /
  stddev **across series** — a single run is never the result.
- GPU: `cudaEvent_t` + `cudaEventSynchronize`. CPU: `std::chrono`.
- Allocation happens once, outside the timed loop (steady-state, not setup).
- CPU, GPU-kernel-only, and end-to-end (H2D+kernel+D2H) are measured separately;
  H2D and D2H are isolated to derive effective bandwidth.

The output records provenance: GPU name, SM, VRAM, CUDA runtime/driver, iteration
counts, and an explicit note that results are **provisional unless the host was
on AC power, thermally stable, and idle** — the program cannot certify that.

## 2. Calibrate (derive scheduler values)

```bash
python tools/calibrate_profiles.py build/benchmarks/profile_raw.json \
    --out build/benchmarks/profile_derived.json \
    --snippet build/benchmarks/profile_snippet.txt
```

The calibrator:
- takes the **largest** shape per operator as the steady-state sample,
- derives `cpu_ns_per_elem` and `gpu_speedup`, plus the two globals,
- runs unit-consistency guards (finite, positive values; exit code 2 on failure),
- writes a derived JSON (schema `forgert.profile.derived/v1`) with a
  `provisional` flag and full provenance, and a paste-ready C++ snippet.

By default results are marked `provisional: true`. Pass `--authoritative` ONLY
when the host was demonstrably controlled; this clears the flag and is an
explicit assertion by the operator, recorded in the output.

## 3. Inspect provenance

Both JSON files carry the `provenance` block (hardware, toolchain, iteration
counts). The derived file adds `provisional` and `notes`. Never present a
provisional derivation as a measured default.

## 4. Promote to defaults (reviewed, manual)

Only when a derivation is **authoritative** and reproduced for agreement across
runs:
1. Review `profile_snippet.txt`; sanity-check each value against physics (e.g. a
   GPU "speedup" < 1 for a memory-bound op on a tiny tensor is a measurement
   artifact, not a default).
2. Hand-edit `SchedulingPolicy::defaultProfiles()` in
   `src/scheduler/scheduler.cpp`, updating the provenance comment with the host
   and date.
3. If the measured launch overhead or bandwidth differ materially, update
   `CUDA_LAUNCH_OVERHEAD_NS` / the default `transfer_bandwidth_bytes_per_ns`
   with a comment citing the run.
4. Re-run the full CTest suite; `ProfileCalibrationTest` guards unit consistency.

## Reproducing a run

Fix the host (AC power, no background load, allow thermal steady-state), then
repeat steps 1–2. Compare derived values across at least two independent runs;
promote only values that agree within a few percent. Record the host and date.

## Measurement validity checks (what the calibrator refuses)

`calibrate_profiles.py` exits **2** (derivation written for inspection, but NOT to
be used) when any of these hold. They exist because each one was a real defect:

| Check | Catches |
|---|---|
| CPU `ns/elem` below 0.01 | CPU work that was optimized away or never ran |
| Time grows < 5% of linear across a >=16x size range (CPU, and GPU kernels above ~2x launch cost) | A benchmark that did **not** run the size it reports |
| Bandwidth outside 0.1..1000 bytes/ns, missing/non-positive values | Mis-scaled units, failed transfers |

`tests/test_calibrate_profiles.py` (`CalibratorFailureModesTest`, standard-library
Python only) pins each failure mode and the no-false-rejection case.

### Postmortem: the "0.0006 ns/element" ReLU

The first provisional run reported ~0.0006 ns/elem for CPU ReLU, flat across
1K..1M elements. It was first attributed to compiler dead-code elimination and the
harness was hardened with a full-output `consume()` and an input `perturb()`. That
did not change the numbers, because **the diagnosis was wrong**: the per-size setup
lambdas declared their tensors `static`, which initialises once, so every larger
"size" silently re-measured the first (1,024-element) tensors while reporting the
larger element count. The harness now owns tensors per case, and the same
operators scale sensibly (see below). `consume()`/`perturb()` are kept as cheap,
standard hygiene but are not what fixed the numbers; note the consume pass adds one
linear read of the output to CPU timings, so memory-bound CPU figures are slightly
conservative (they over-count rather than under-count).

The harness also uses **cost-adaptive iteration counts** (one probe call sizes the
inner loop to ~20 ms per sample, capped at the configured maximum; heavy ops such
as a 512x512 CPU matmul run with `inner_iters = 1`). The actual `inner_iters` and
`warmup` per measurement are recorded in the raw JSON.

## Current status

The compiled-in defaults in `scheduler.cpp` are still the original GTX 1050 /
i5-9300H **estimates**. Nothing measured has been promoted.

One corrected, validated provisional run on the development laptop (AC power; the host was not verified idle or thermally stable, and a browser was
observed consuming CPU earlier in the session) gave,
for reference only:

| Op | Shape | CPU ns/elem | GPU kernel ns/elem | kernel-only speedup |
|---|---|---|---|---|
| ReLU | 1,048,576 | ~1.66 | ~0.136 | ~12x |
| MatMul | 512x512 | ~885 | ~4.0 | ~220x |

with ~5.7 bytes/ns H2D/D2H and ~9.0 us null-kernel launch overhead (vs the
hard-coded 20 us). These are **single-run, loaded-host figures**: spread across
samples was 4-7%, but they have not been reproduced across independent runs on an
idle machine, so they are **provisional** and are not promoted. They do suggest the
estimated launch overhead (20 us) and CPU ReLU cost (0.40 ns/elem) are worth
re-examining on a controlled host.

### Known limitation

`CUDA_LAUNCH_OVERHEAD_NS` is a constant inside `SchedulingPolicy::decide()`, not a
profile field, so promoting a measured launch overhead is a source edit. Making it
a field of the profile/context is future work.

### Before promoting to defaults

1. Idle host, AC power, thermal steady state; close background apps.
2. At least two independent runs agreeing within a few percent.
3. Calibrator exit code 0 and `--authoritative` asserted by the operator.
4. Review the snippet against physics, update the provenance comment, re-run CTest.
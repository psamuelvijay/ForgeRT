#!/usr/bin/env python3
"""Regression tests for tools/calibrate_profiles.py failure modes.

Pins the benchmark/calibration defects found while hardening the profiler:

  1. Size reuse: tensors declared `static` inside a per-size lambda were built once,
     so every larger "size" re-measured the smallest tensors. Symptom: time flat
     across a >1000x size range and an impossibly low ns/elem.
  2. Elided/hoisted CPU work: ns/elem below any physical floor.
  3. No false rejection: a plausible, properly scaling run must pass (and still be
     marked provisional unless --authoritative).
  4. Malformed input must fail cleanly (exit 1), not crash.

Exit codes: 0 pass, 1 fail.
"""
import json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.join(HERE, "..", "tools", "calibrate_profiles.py")
fails = 0

def ok(cond, msg):
    global fails
    print(("PASS: " if cond else "FAIL: ") + msg)
    if not cond: fails += 1

def op(name, elems, cpu_ms, ker_ms):
    return {"op": name, "shape": f"[{elems}]", "elems": elems,
            "cpu": {"median_ms": cpu_ms, "stddev_ms": cpu_ms * 0.02},
            "gpu_kernel": {"median_ms": ker_ms}, "gpu_e2e": {"median_ms": ker_ms * 5}}

def raw(ops, bw=6.0, launch=9000.0):
    return {"schema": "forgert.profile.raw/v1",
            "provenance": {"gpu_name": "synthetic", "gpu_sm": "sm_61"},
            "transfers": [{"bytes": 4194304, "h2d_bytes_per_ns": bw, "d2h_bytes_per_ns": bw}],
            "launch_overhead": {"launch_overhead_ns": launch},
            "operators": ops}

def run(doc, *extra, text=None):
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "raw.json")
        with open(f, "w") as fh:
            fh.write(text if text is not None else json.dumps(doc))
        r = subprocess.run([sys.executable, CAL, f, *extra], capture_output=True, text=True)
    return r

# 1. size-reuse bug signature: identical time at 1K and 1M elements
r = run(raw([op("ReLU", 1024, 0.0017, 0.020), op("ReLU", 65536, 0.0020, 0.019),
             op("ReLU", 1048576, 0.0018, 0.018)]))
ok(r.returncode == 2, "flat time across 1024x size range is rejected (exit 2)")
ok("did not run the claimed size" in r.stderr, "rejection names the size-reuse cause")

# 2. elided CPU work: impossible ns/elem even with scaling absent of other signal
r = run(raw([op("ReLU", 1048576, 0.0006, 0.14)]))
ok(r.returncode == 2 and "physically impossible" in r.stderr,
   "sub-floor CPU ns/elem is rejected")

# 3. plausible, correctly scaling run is accepted and stays provisional
good = raw([op("ReLU", 1024, 0.0022, 0.0170), op("ReLU", 65536, 0.078, 0.0194),
            op("ReLU", 1048576, 1.74, 0.142),
            op("MatMul", 65536, 16.4, 0.14), op("MatMul", 262144, 232.0, 1.05)])
r = run(good)
ok(r.returncode == 0, "plausible properly-scaling run is accepted (no false rejection)")
ok("PROVISIONAL" in r.stderr, "accepted run is still marked provisional by default")
r2 = run(good, "--authoritative")
ok(r2.returncode == 0 and "PROVISIONAL" not in r2.stderr,
   "--authoritative clears the provisional flag")

# 4. launch-dominated small GPU kernels must NOT trip the GPU scaling check
ok(run(good).returncode == 0, "launch-dominated small GPU sizes are not falsely flagged")

# 5. malformed input fails cleanly
ok(run(None, text="{not json").returncode == 1, "invalid JSON exits 1 (clean failure)")
bad = raw([]); bad["schema"] = "wrong/v9"
ok(run(bad).returncode == 1, "wrong schema exits 1 (clean failure)")

print("\n" + ("All calibrator tests PASSED" if fails == 0 else f"{fails} calibrator test(s) FAILED"))
sys.exit(1 if fails else 0)
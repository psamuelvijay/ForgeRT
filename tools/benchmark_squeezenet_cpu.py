#!/usr/bin/env python3
"""ForgeRT SqueezeNet 1.1 CPU latency baseline (process-level repetition).

forgert_run performs exactly ONE inference per process, so repetition here means launching it
repeatedly. Nothing in the runtime is modified. Per launch the script records three numbers:

  wall_ms       Launcher-side wall clock around subprocess.run(): process creation, CRT/DLL load,
                .fgx import, reading the 602,112-byte input, ONE inference, writing the 4,000-byte
                output, process exit, and pipe capture. This is END-TO-END PROCESS latency.
  import_ms     Reported by forgert_run: ImportedModel::load() (read .fgx, build graph + weights).
  inference_ms  Reported by forgert_run: the single model->run() call only (graph ordering,
                allocating every intermediate tensor, executing all 66 nodes, freeing them). It
                excludes process start, import, input read and output write. It is the FIRST and only
                inference of that process, so it is a cold-in-process figure, not steady state:
                in-process warm repetition would need a runtime change and is not measured.
  overhead_ms   wall_ms - import_ms - inference_ms (process start/exit, input read, output write,
                launcher spawn/capture cost).

Warm-up launches run first and are discarded from the statistics (they warm the OS file cache and the
CPU clocks); they are still recorded in the JSON. Input: float32 [1,3,224,224] drawn from N(0,1) with
numpy default_rng(--seed). Timing should not depend on the values, but they are fixed for repeatability.
All outputs must be byte-identical across launches (checked), as a guard against timing a different
computation.

    python tools/benchmark_squeezenet_cpu.py --runner build/apps/forgert_run.exe \
        --warmup 3 --runs 20 --json docs/SQUEEZENET_CPU_LATENCY.json

Exit codes: 0 ok, 1 failure, 77 skipped (missing package or file).
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))

try:
    import numpy as np
except ImportError as e:
    print(f"SKIP: {e}")
    sys.exit(77)

IMPORT_RE = re.compile(r"import ([0-9.]+) ms")
INFER_RE = re.compile(r"inference \(CPU\) ([0-9.]+) ms")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    try:
        r = subprocess.run(["git", "-C", os.path.join(HERE, ".."), *args], capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def stats(xs):
    s = sorted(xs)
    q = statistics.quantiles(s, n=4, method="inclusive") if len(s) >= 2 else [s[0]] * 3
    return {
        "n": len(s), "median": statistics.median(s), "min": s[0], "max": s[-1],
        "mean": statistics.fmean(s), "stdev": statistics.stdev(s) if len(s) > 1 else 0.0,
        "q1": q[0], "q3": q[2],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runner", required=True)
    ap.add_argument("--model", default=os.path.join(HERE, "..", "models", "squeezenet1.1-7.onnx"))
    ap.add_argument("--fgx", default=None, help="use an existing .fgx instead of converting --model")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--timeout", type=float, default=300.0)
    a = ap.parse_args()
    if a.runs < 1 or a.warmup < 0:
        print("error: --runs must be >= 1 and --warmup >= 0")
        return 1

    model = os.path.normpath(a.model)
    need = [a.runner] + ([a.fgx] if a.fgx else [model])
    for p in need:
        if not os.path.isfile(p):
            print(f"SKIP: file not found: {p}")
            return 77

    workdir = a.workdir or tempfile.mkdtemp(prefix="forgert_bench_")
    os.makedirs(workdir, exist_ok=True)
    fgx = a.fgx
    if not fgx:                                      # conversion is NOT timed
        sys.path.insert(0, HERE)
        try:
            import onnx_to_fgx
        except ImportError as e:
            print(f"SKIP: {e}")
            return 77
        fgx = os.path.join(workdir, "squeezenet1.1-7.fgx")
        onnx_to_fgx.convert_file(model, fgx)

    x = np.random.default_rng(a.seed).standard_normal((1, 3, 224, 224), dtype=np.float32)
    in_bin, out_bin = os.path.join(workdir, "input.bin"), os.path.join(workdir, "output.bin")
    x.astype("<f4").tofile(in_bin)

    cmd = [a.runner, fgx, in_bin, out_bin]
    print("command per launch:", " ".join(cmd))
    print(f"warm-up launches: {a.warmup} (discarded); measured launches: {a.runs}")

    records, out_hashes = [], set()
    for i in range(a.warmup + a.runs):
        phase = "warmup" if i < a.warmup else "measured"
        if os.path.exists(out_bin):
            os.remove(out_bin)
        t0 = time.perf_counter_ns()
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
        wall_ms = (time.perf_counter_ns() - t0) / 1e6
        if r.returncode != 0:
            print(f"FAIL: launch {i} exited {r.returncode}: {r.stderr.strip()}")
            return 1
        mi, mf = IMPORT_RE.search(r.stderr), INFER_RE.search(r.stderr)
        if not (mi and mf):
            print(f"FAIL: launch {i}: could not parse timings from runner stderr:\n{r.stderr}")
            return 1
        imp, inf = float(mi.group(1)), float(mf.group(1))
        digest = sha256(out_bin)
        out_hashes.add(digest)
        rec = {"index": i, "phase": phase, "wall_ms": round(wall_ms, 3), "import_ms": imp,
               "inference_ms": inf, "overhead_ms": round(wall_ms - imp - inf, 3)}
        records.append(rec)
        print(f"  {phase:<8} #{i:<3} wall {wall_ms:8.1f}  import {imp:7.2f}  inference {inf:8.2f}  "
              f"other {rec['overhead_ms']:7.1f}  ms")

    measured = [r for r in records if r["phase"] == "measured"]
    result = {k: stats([r[k] for r in measured]) for k in ("wall_ms", "import_ms", "inference_ms", "overhead_ms")}
    half = len(measured) // 2
    drift = None
    if half >= 1:
        drift = {"first_half_median_inference_ms": statistics.median(r["inference_ms"] for r in measured[:half]),
                 "second_half_median_inference_ms": statistics.median(r["inference_ms"] for r in measured[half:])}
    identical = len(out_hashes) == 1

    print("\nSUMMARY (measured launches only, milliseconds)")
    print(f"{'metric':<34}{'median':>9}{'min':>9}{'max':>9}{'mean':>9}{'stdev':>8}{'q1':>9}{'q3':>9}")
    labels = {"wall_ms": "end-to-end process (wall)", "inference_ms": "inference only (runner-reported)",
              "import_ms": "model import (runner-reported)", "overhead_ms": "other (start/exit, file I/O)"}
    for k in ("wall_ms", "inference_ms", "import_ms", "overhead_ms"):
        s = result[k]
        print(f"{labels[k]:<34}{s['median']:9.2f}{s['min']:9.2f}{s['max']:9.2f}{s['mean']:9.2f}"
              f"{s['stdev']:8.2f}{s['q1']:9.2f}{s['q3']:9.2f}")
    if drift:
        print(f"drift check (inference median): first half {drift['first_half_median_inference_ms']:.2f} ms, "
              f"second half {drift['second_half_median_inference_ms']:.2f} ms")
    print(f"outputs byte-identical across all {len(records)} launches: {identical}")

    if a.json:
        doc = {
            "what": "ForgeRT SqueezeNet 1.1 CPU latency, process-level repetition (one inference per launch)",
            "generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "command": cmd, "script_args": sys.argv[1:],
            "settings": {"warmup": a.warmup, "runs": a.runs, "seed": a.seed, "timer": "time.perf_counter_ns"},
            "definitions": {
                "wall_ms": "launcher-side wall clock around the whole process: end-to-end process latency",
                "import_ms": "runner-reported ImportedModel::load()",
                "inference_ms": "runner-reported single model->run() call; first and only inference in the process",
                "overhead_ms": "wall_ms - import_ms - inference_ms",
            },
            "environment": {
                "platform": platform.platform(), "processor_string": platform.processor(),
                "logical_cpus": os.cpu_count(), "python": platform.python_version(), "numpy": np.__version__,
                "git_head": git("rev-parse", "--short", "HEAD"),
                "git_tracked_changes": len((git("status", "--porcelain", "--untracked-files=no") or "").splitlines()),
            },
            "artifacts": {
                "runner": {"path": os.path.abspath(a.runner), "bytes": os.path.getsize(a.runner),
                           "sha256": sha256(a.runner)},
                "fgx": {"path": os.path.abspath(fgx), "bytes": os.path.getsize(fgx), "sha256": sha256(fgx)},
                "model_onnx_sha256": sha256(model) if os.path.isfile(model) else None,
                "input_sha256": sha256(in_bin), "output_sha256": sorted(out_hashes)[0] if identical else None,
            },
            "outputs_byte_identical": identical,
            "stats_measured": result, "drift_check": drift, "runs": records,
        }
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(doc, f, indent=2)
            f.write("\n")
        print("wrote", a.json)
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
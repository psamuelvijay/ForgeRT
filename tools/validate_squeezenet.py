#!/usr/bin/env python3
"""End-to-end check: ForgeRT vs ONNX Runtime (CPU) on SqueezeNet 1.1.

Pass criteria, for EVERY one of the seeded inputs (not on average):
  - max |ForgeRT - ORT| < --atol (default 1e-3)
  - argmax(ForgeRT) == argmax(ORT)
Relative error and the ORT top-1/top-2 margin are reported as diagnostics only.

    python tools/validate_squeezenet.py --runner build/apps/forgert_run.exe
        [--model models/squeezenet1.1-7.onnx] [--workdir DIR] [--json report.json]

Exit codes: 0 all inputs pass, 1 any failure, 77 skipped (missing package or model).
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))

try:
    import numpy as np
    import onnx
    import onnxruntime as ort
except ImportError as e:
    print(f"SKIP: {e}")
    sys.exit(77)

sys.path.insert(0, HERE)
import onnx_to_fgx  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", required=True, help="path to forgert_run")
    ap.add_argument("--model", default=os.path.join(HERE, "..", "models", "squeezenet1.1-7.onnx"))
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--json", default=None, help="write a machine-readable report here")
    a = ap.parse_args()

    model = os.path.normpath(a.model)
    if not os.path.isfile(model):
        print(f"SKIP: model not found: {model} (run tools/fetch_squeezenet.py)")
        return 77
    if not os.path.isfile(a.runner):
        print(f"FAIL: runner not found: {a.runner}")
        return 1
    workdir = a.workdir or tempfile.mkdtemp(prefix="forgert_e2e_")
    os.makedirs(workdir, exist_ok=True)

    fgx_path = os.path.join(workdir, "squeezenet1.1-7.fgx")
    onnx_to_fgx.convert_file(model, fgx_path)

    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    in_shape = tuple(sess.get_inputs()[0].shape)

    rng = np.random.default_rng(a.seed)
    rows, all_pass = [], True
    for i in range(a.count):
        x = rng.standard_normal(in_shape, dtype=np.float32)
        ref = sess.run(None, {in_name: x})[0]

        in_bin = os.path.join(workdir, f"input_{i}.bin")
        out_bin = os.path.join(workdir, f"output_{i}.bin")
        x.astype("<f4").tofile(in_bin)
        t0 = time.perf_counter()
        r = subprocess.run([a.runner, fgx_path, in_bin, out_bin], capture_output=True, text=True)
        wall_ms = (time.perf_counter() - t0) * 1000
        if r.returncode != 0:
            print(f"FAIL: input {i}: forgert_run exited {r.returncode}: {r.stderr.strip()}")
            return 1
        got = np.fromfile(out_bin, dtype="<f4")
        if got.size != ref.size:
            print(f"FAIL: input {i}: output has {got.size} values, ORT has {ref.size}")
            return 1
        got = got.reshape(ref.shape)

        diff = np.abs(got.astype(np.float64) - ref.astype(np.float64))
        max_abs = float(diff.max())
        ref_abs = np.abs(ref.astype(np.float64))
        mask = ref_abs >= 1e-2
        srt = np.sort(ref.ravel())
        row = {
            "input": i,
            "max_abs": max_abs,
            "mean_abs": float(diff.mean()),
            "max_abs_over_max_ref": max_abs / float(ref_abs.max()),
            "max_rel_where_ref_ge_1e-2": float((diff[mask] / ref_abs[mask]).max()) if mask.any() else 0.0,
            "argmax_forgert": int(got.argmax()),
            "argmax_ort": int(ref.argmax()),
            "ort_top1_margin": float(srt[-1] - srt[-2]),
            "forgert_wall_ms": wall_ms,
        }
        row["pass"] = bool(max_abs < a.atol and row["argmax_forgert"] == row["argmax_ort"])
        all_pass &= row["pass"]
        rows.append(row)

    print(f"SqueezeNet 1.1 end-to-end: ForgeRT (CPU) vs ONNX Runtime {ort.__version__} "
          f"(CPUExecutionProvider, default optimization); onnx {onnx.__version__}, numpy {np.__version__}")
    print(f"inputs: {a.count} x N(0,1) {list(in_shape)} float32, numpy default_rng(seed={a.seed}); "
          f"criteria: max|d| < {a.atol:g} and argmax match on every input")
    print(f"{'#':>2} {'max|d|':>10} {'mean|d|':>10} {'max|d|/max|ref|':>16} {'max rel(|ref|>=.01)':>20} "
          f"{'argmax F/O':>11} {'ORT margin':>11} {'wall ms':>8}  result")
    for r_ in rows:
        print(f"{r_['input']:>2} {r_['max_abs']:10.3e} {r_['mean_abs']:10.3e} {r_['max_abs_over_max_ref']:16.3e} "
              f"{r_['max_rel_where_ref_ge_1e-2']:20.3e} {r_['argmax_forgert']:>5}/{r_['argmax_ort']:<5} "
              f"{r_['ort_top1_margin']:11.5f} {r_['forgert_wall_ms']:8.0f}  {'PASS' if r_['pass'] else 'FAIL'}")
    worst = max(r_["max_abs"] for r_ in rows)
    passed = sum(r_["pass"] for r_ in rows)
    print(f"{'PASS' if all_pass else 'FAIL'}: {passed}/{a.count} inputs within criteria; "
          f"worst max|d| = {worst:.3e} (limit {a.atol:g})")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"model": os.path.basename(model), "seed": a.seed, "atol": a.atol,
                       "onnxruntime": ort.__version__, "onnx": onnx.__version__, "numpy": np.__version__,
                       "platform": platform.platform(), "processor": platform.processor(),
                       "all_pass": all_pass, "rows": rows}, f, indent=2)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
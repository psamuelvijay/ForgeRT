#!/usr/bin/env python3
"""Derive scheduler operator profiles from a raw profiling run.

Reads the machine-readable JSON produced by benchmarks/benchmark_profile.exe
(schema "forgert.profile.raw/v1") and derives the values the scheduler cost
model uses:

  OperatorProfile.cpu_ns_per_elem = CPU median ms / elems, at the LARGEST shape
                                    (steady-state, amortizes fixed overhead)
  OperatorProfile.gpu_speedup     = cpu_kernel_ns_per_elem / gpu_kernel_ns_per_elem
                                    (kernel-only ratio; transfers modelled apart)
  transfer_bandwidth_bytes_per_ns = median effective bytes/ns across H2D+D2H at
                                    the largest transfer size
  cuda_launch_overhead_ns         = measured null-kernel median (ns)

Outputs (never overwrites scheduler source):
  * a derived-profiles JSON (schema "forgert.profile.derived/v1"), kept separate
    from the raw measurements, and
  * a paste-ready C++ snippet for SchedulingPolicy::defaultProfiles().

Nothing here promotes values into the compiled defaults: that is a reviewed,
manual step. The derived JSON carries a `provisional` flag and the full
provenance so estimates are never silently presented as measured data.

Usage:
    python tools/calibrate_profiles.py RAW.json [--out DERIVED.json]
        [--snippet SNIPPET.txt] [--authoritative]

Exit codes: 0 ok, 1 bad/missing input, 2 unit-consistency check failed.
"""
import argparse
import json
import sys

RAW_SCHEMA = "forgert.profile.raw/v1"
DERIVED_SCHEMA = "forgert.profile.derived/v1"


def ns_per_elem(median_ms, elems):
    if elems <= 0 or median_ms <= 0:
        return None
    return median_ms * 1e6 / elems  # ms -> ns, per element


def derive(raw, authoritative):
    if raw.get("schema") != RAW_SCHEMA:
        raise ValueError(f"unexpected schema {raw.get('schema')!r}, want {RAW_SCHEMA!r}")

    prov = raw.get("provenance", {})

    # --- per-operator: take the LARGEST shape as the steady-state sample ---
    by_op = {}
    for e in raw.get("operators", []):
        by_op.setdefault(e["op"], []).append(e)
    op_profiles = {}
    for op, entries in by_op.items():
        biggest = max(entries, key=lambda e: e["elems"])
        elems = biggest["elems"]
        cpu_npe = ns_per_elem(biggest["cpu"]["median_ms"], elems)
        gpu_kernel_npe = ns_per_elem(biggest["gpu_kernel"]["median_ms"], elems)
        speedup = None
        if cpu_npe and gpu_kernel_npe and gpu_kernel_npe > 0:
            speedup = cpu_npe / gpu_kernel_npe
        # Per-size points kept so scaling can be audited / validated downstream.
        sizes = sorted(
            ({"elems": e["elems"], "cpu_median_ms": e["cpu"]["median_ms"],
              "gpu_kernel_median_ms": e["gpu_kernel"]["median_ms"]} for e in entries),
            key=lambda s: s["elems"])
        op_profiles[op] = {
            "sizes": sizes,
            "cpu_ns_per_elem": cpu_npe,
            "gpu_kernel_ns_per_elem": gpu_kernel_npe,
            "gpu_speedup": speedup,
            "derived_from_shape": biggest["shape"],
            "derived_from_elems": elems,
            "cpu_stddev_ms": biggest["cpu"].get("stddev_ms"),
            "gpu_kernel_stddev_ms": biggest["gpu_kernel"].get("stddev_ms"),
        }

    # --- transfer bandwidth: largest transfer, median of H2D and D2H bytes/ns ---
    transfers = raw.get("transfers", [])
    bw = None
    if transfers:
        big = max(transfers, key=lambda t: t["bytes"])
        bws = [b for b in (big.get("h2d_bytes_per_ns"), big.get("d2h_bytes_per_ns")) if b]
        if bws:
            bw = sum(bws) / len(bws)

    launch_ns = raw.get("launch_overhead", {}).get("launch_overhead_ns")

    derived = {
        "schema": DERIVED_SCHEMA,
        "provisional": (not authoritative),
        "provenance": prov,
        "globals": {
            "transfer_bandwidth_bytes_per_ns": bw,
            "cuda_launch_overhead_ns": launch_ns,
        },
        "operators": op_profiles,
        "notes": (
            "PROVISIONAL: host suitability not certified. Do NOT promote to "
            "SchedulingPolicy::defaultProfiles() until produced on a quiescent, "
            "AC-powered, thermally stable machine and re-run for agreement."
            if not authoritative else
            "Marked authoritative by operator via --authoritative."
        ),
    }
    return derived


# Plausibility bounds. These are deliberately loose -- they reject only
# physically impossible values (the hallmark of a benchmark the compiler
# optimized away), not merely slow or fast machines.
#
# CPU_NS_PER_ELEM_FLOOR: a scalar float op doing real per-element work on a
# commodity CPU cannot sustain < 0.01 ns/elem (= >100 elements/ns ≈ >100 GFLOP/s
# per op on a single thread with memory traffic). A value below this floor means
# the timed loop was hoisted/elided, not measured. Observed on this host: ReLU
# collapsed to ~0.0006 ns/elem, which this floor correctly rejects.
CPU_NS_PER_ELEM_FLOOR = 0.01
# A generous upper sanity bound catches mis-scaled units (e.g. seconds vs ns).
CPU_NS_PER_ELEM_CEIL = 1.0e6
# Transfer bandwidth sanity: PCIe-class links are ~1..100 bytes/ns effective.
BW_FLOOR, BW_CEIL = 0.1, 1000.0
# Size-scaling check. A real elementwise/compute op does MORE work for MORE
# elements. If the largest case is >= SCALE_MIN_RATIO x bigger than the smallest
# yet its time grew by less than SCALE_MIN_FRACTION of that ratio, the benchmark
# did not run the size it claims (e.g. tensors reused across sizes by a
# function-local `static`, which is the bug this guards against). Deliberately
# generous: fixed per-call overhead legitimately flattens tiny sizes, so this only
# fires when growth is a small fraction of linear.
SCALE_MIN_RATIO = 16.0
SCALE_MIN_FRACTION = 0.05


def check_units(derived):
    """Unit-consistency and plausibility guards.

    Returns a list of problems. A non-empty list means the derivation must NOT
    be accepted as calibration (physically impossible or inconsistent values).
    """
    problems = []
    g = derived["globals"]
    bw = g["transfer_bandwidth_bytes_per_ns"]
    if not bw or bw <= 0:
        problems.append("transfer bandwidth missing or <= 0")
    elif not (BW_FLOOR <= bw <= BW_CEIL):
        problems.append(f"transfer bandwidth {bw:.3g} bytes/ns outside plausible "
                        f"[{BW_FLOOR}, {BW_CEIL}] -- suspect units or elision")
    if not g["cuda_launch_overhead_ns"] or g["cuda_launch_overhead_ns"] <= 0:
        problems.append("launch overhead missing or <= 0")

    for op, p in derived["operators"].items():
        cpu = p["cpu_ns_per_elem"]
        if not cpu or cpu <= 0:
            problems.append(f"{op}: cpu_ns_per_elem missing/<=0")
        elif cpu < CPU_NS_PER_ELEM_FLOOR:
            problems.append(
                f"{op}: cpu_ns_per_elem {cpu:.4g} < floor {CPU_NS_PER_ELEM_FLOOR} "
                f"-- physically impossible; the CPU benchmark was optimized away, "
                f"not measured (do NOT calibrate from this)")
        elif cpu > CPU_NS_PER_ELEM_CEIL:
            problems.append(f"{op}: cpu_ns_per_elem {cpu:.4g} > ceil "
                            f"{CPU_NS_PER_ELEM_CEIL} -- suspect units")
        all_sizes = p.get("sizes") or []
        launch_ms = (g["cuda_launch_overhead_ns"] or 0) / 1e6
        for field, label in (("cpu_median_ms", "CPU"),
                             ("gpu_kernel_median_ms", "GPU kernel")):
            sizes = all_sizes
            if field == "gpu_kernel_median_ms":
                # Small kernels are dominated by the fixed launch cost and do not
                # scale with size; only compare sizes whose time clearly exceeds it.
                sizes = [s for s in all_sizes if s[field] >= 2.0 * launch_ms]
            if len(sizes) >= 2:
                lo, hi = sizes[0], sizes[-1]
                ratio = hi["elems"] / lo["elems"] if lo["elems"] else 0
                if ratio >= SCALE_MIN_RATIO:
                    if lo[field] > 0 and hi[field] / lo[field] < SCALE_MIN_FRACTION * ratio:
                        problems.append(
                            f"{op}: {label} time grew only {hi[field] / lo[field]:.2f}x "
                            f"for a {ratio:.0f}x larger input (< {SCALE_MIN_FRACTION:.0%} of "
                            f"linear) -- the benchmark did not run the claimed size "
                            f"(do NOT calibrate from this)")
        if p["gpu_speedup"] is not None and p["gpu_speedup"] <= 0:
            problems.append(f"{op}: gpu_speedup <= 0")
    return problems


def cpp_snippet(derived):
    lines = []
    lines.append("// GENERATED by tools/calibrate_profiles.py -- REVIEW before pasting.")
    flag = "PROVISIONAL (do not promote)" if derived["provisional"] else "authoritative"
    lines.append(f"// Status: {flag}")
    prov = derived["provenance"]
    lines.append(f"// Host: {prov.get('gpu_name')} {prov.get('gpu_sm')}, "
                 f"CUDA rt {prov.get('cuda_runtime')}, "
                 f"{prov.get('samples')} samples x {prov.get('inner_iters')} iters")
    g = derived["globals"]
    lines.append(f"// transfer_bandwidth_bytes_per_ns = {g['transfer_bandwidth_bytes_per_ns']:.3f}")
    lines.append(f"// cuda_launch_overhead_ns        = {g['cuda_launch_overhead_ns']:.0f}")
    lines.append("return {")
    for op, p in sorted(derived["operators"].items()):
        cpu = p["cpu_ns_per_elem"]
        spd = p["gpu_speedup"]
        spd_s = f"{spd:.1f}" if spd is not None else "1.0"
        lines.append(f'    {{"{op}", {{"{op}", {cpu:.3f}, {spd_s}, true}}}},'
                     f'  // from {p["derived_from_shape"]}')
    lines.append("};")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", help="raw profiling JSON from benchmark_profile")
    ap.add_argument("--out", help="write derived-profiles JSON here")
    ap.add_argument("--snippet", help="write paste-ready C++ snippet here")
    ap.add_argument("--authoritative", action="store_true",
                    help="assert the host was controlled; clears the provisional flag")
    a = ap.parse_args()

    try:
        with open(a.raw, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as e:
        print(f"ERROR: cannot read raw JSON: {e}", file=sys.stderr)
        return 1

    try:
        derived = derive(raw, a.authoritative)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    problems = check_units(derived)
    out_json = json.dumps(derived, indent=2)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(out_json)
        print(f"wrote derived profiles: {a.out}", file=sys.stderr)
    else:
        print(out_json)

    if a.snippet:
        with open(a.snippet, "w", encoding="utf-8") as f:
            f.write(cpp_snippet(derived))
        print(f"wrote C++ snippet: {a.snippet}", file=sys.stderr)

    if problems:
        print("UNIT-CONSISTENCY PROBLEMS:", file=sys.stderr)
        for p in problems:
            print("  - " + p, file=sys.stderr)
        return 2
    if derived["provisional"]:
        print("NOTE: results are PROVISIONAL; not for promotion to defaults.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
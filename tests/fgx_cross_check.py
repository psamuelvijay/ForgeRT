#!/usr/bin/env python3
"""Cross-language .fgx check: Python converter manifest vs C++ fgx_dump.

The converter writes the canonical manifest from its in-memory model; the C++
reader parses the written file and prints the same manifest. They must be
byte-identical (names, shapes, dtypes, attributes, per-initializer CRC-32).

    fgx_cross_check.py --dump PATH/fgx_dump --synthetic --workdir DIR
    fgx_cross_check.py --dump PATH/fgx_dump --model squeezenet1.1-7.onnx --workdir DIR

Exit codes: 0 identical, 1 mismatch or error, 77 skipped (missing model or package).
"""
import argparse
import os
import subprocess
import sys

try:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError as e:
    print(f"SKIP: {e}")
    sys.exit(77)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import onnx_to_fgx  # noqa: E402


def synthetic_model(path):
    """Small model exercising every attribute kind, escaping, and both dtypes."""
    w = numpy_helper.from_array(np.linspace(-1, 1, 12, dtype=np.float32).reshape(6, 2, 1, 1), "conv w")
    b = numpy_helper.from_array(np.array([np.nan, np.inf, -np.inf, -0.0, 1e-38, 3.5], np.float32), "b")
    shape = numpy_helper.from_array(np.array([0, -1], np.int64), "shape")
    g = helper.make_graph([
        helper.make_node("Conv", ["x", "conv w", "b"], ["y"], name="",
                         kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0], group=1),
        helper.make_node("Dropout", ["y"], ["d"], name="drop\\out", ratio=0.3),
        helper.make_node("Reshape", ["d", "shape"], ["z"], name="flat"),
    ], "synthetic",
        # IR v3 (like SqueezeNet 1.1) lists initializers as graph inputs too.
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 3]),
         helper.make_tensor_value_info("conv w", TensorProto.FLOAT, [6, 2, 1, 1]),
         helper.make_tensor_value_info("b", TensorProto.FLOAT, [6]),
         helper.make_tensor_value_info("shape", TensorProto.INT64, [2])],
        [helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 54])],
        initializer=[w, b, shape])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 7)], producer_name="unit test \u00fc")
    m.ir_version = 3
    onnx.save(m, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--workdir", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model")
    src.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.workdir, exist_ok=True)
    if a.synthetic:
        model = os.path.join(a.workdir, "synthetic.onnx")
        synthetic_model(model)
    else:
        model = a.model
        if not os.path.isfile(model):
            print(f"SKIP: model not found: {model} (run tools/fetch_squeezenet.py)")
            return 77
    stem = os.path.join(a.workdir, os.path.splitext(os.path.basename(model))[0])
    fgx_path, py_man, cpp_man = stem + ".fgx", stem + ".py.manifest.txt", stem + ".cpp.manifest.txt"

    fgx, size = onnx_to_fgx.convert_file(model, fgx_path, py_man)
    r = subprocess.run([a.dump, fgx_path, cpp_man], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAIL: fgx_dump exited {r.returncode}: {r.stderr.strip()}")
        return 1
    py_text, cpp_text = open(py_man, "rb").read(), open(cpp_man, "rb").read()
    if py_text != cpp_text:
        py_lines, cpp_lines = py_text.split(b"\n"), cpp_text.split(b"\n")
        for i, (p, c) in enumerate(zip(py_lines, cpp_lines)):
            if p != c:
                print(f"FAIL: manifests differ at line {i + 1}\n  python: {p[:200]!r}\n  c++:    {c[:200]!r}")
                break
        else:
            print(f"FAIL: manifests differ in length ({len(py_lines)} vs {len(cpp_lines)} lines)")
        return 1
    lines = py_text.count(b"\n")
    print(f"PASS: {os.path.basename(model)}: {size} bytes, {len(fgx['nodes'])} nodes, "
          f"{len(fgx['initializers'])} initializers, {len(fgx['value_info'])} value_info; "
          f"Python and C++ manifests identical ({lines} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
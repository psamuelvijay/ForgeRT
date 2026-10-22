#!/usr/bin/env python3
"""Real-image check: ForgeRT vs ONNX Runtime (CPU) on SqueezeNet 1.1.

Each image is preprocessed ONCE (ImageNet convention, below) and the identical float32
tensor is given to both runtimes, so any difference comes from execution, not preprocessing.

Preprocessing (ImageNet convention, applied here with Pillow + NumPy; similar to torchvision's
Resize(256) + CenterCrop(224) + Normalize but not identical, see steps 2 and 3):
  1. open, convert to RGB (EXIF orientation is NOT applied)
  2. resize so the shorter side is 256 px (PIL bilinear), keeping aspect ratio; the longer side is
     round(long * 256 / short) (torchvision truncates with int() instead)
  3. center-crop 224 x 224 with offset (size - 224) // 2 (torchvision: int(round((size - 224) / 2)))
  4. scale to [0, 1], subtract mean [0.485, 0.456, 0.406], divide by std [0.229, 0.224, 0.225]
  5. HWC -> CHW, add batch dimension: float32 [1, 3, 224, 224]

Reported per image: max/mean absolute difference of the 1000 logits, top-1 and top-5
agreement, predicted class index and name. This measures AGREEMENT between two
runtimes. It is not accuracy: no ground-truth labels are used or claimed.

    python tools/validate_squeezenet_images.py --runner build/apps/forgert_run.exe \
        --images a.jpg b.jpg c.jpg [--model models/squeezenet1.1-7.onnx] [--workdir DIR]

Pass criteria (per image, same as tools/validate_squeezenet.py): max|d| < --atol and
equal argmax. Exit codes: 0 all pass, 1 any failure, 77 skipped (missing package/file).
"""
import argparse
import ast
import hashlib
import importlib.util
import os
import platform
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

try:
    import numpy as np
    import onnx
    import onnxruntime as ort
    from PIL import Image
except ImportError as e:
    print(f"SKIP: {e}")
    sys.exit(77)

sys.path.insert(0, HERE)
import onnx_to_fgx  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RESIZE, CROP = 256, 224


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def preprocess(path):
    """Return (float32 [1,3,224,224], original (w, h))."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = RESIZE / min(w, h)
        nw, nh = max(RESIZE, round(w * scale)), max(RESIZE, round(h * scale))
        im = im.resize((nw, nh), Image.BILINEAR)
        left, top = (nw - CROP) // 2, (nh - CROP) // 2
        im = im.crop((left, top, left + CROP, top + CROP))
        arr = np.asarray(im, dtype=np.float32) / 255.0          # HWC, [0, 1]
    arr = (arr - MEAN) / STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None, ...], dtype=np.float32), (w, h)


def load_categories():
    """ImageNet-1k class names from torchvision's _meta.py, read as text (torchvision is not imported)."""
    spec = importlib.util.find_spec("torchvision")
    if spec is None or not spec.submodule_search_locations:
        return None, None
    path = os.path.join(list(spec.submodule_search_locations)[0], "models", "_meta.py")
    if not os.path.isfile(path):
        return None, None
    text = open(path, encoding="utf-8").read()
    m = re.search(r"_IMAGENET_CATEGORIES\s*=\s*(\[.*?\n\])", text, re.S)
    if not m:
        return None, None
    cats = ast.literal_eval(m.group(1))
    return (cats, path) if len(cats) == 1000 else (None, None)


def softmax(x):
    e = np.exp(x.astype(np.float64) - x.max())
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--model", default=os.path.join(HERE, "..", "models", "squeezenet1.1-7.onnx"))
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--atol", type=float, default=1e-3)
    a = ap.parse_args()

    model = os.path.normpath(a.model)
    for p in [model, a.runner] + a.images:
        if not os.path.isfile(p):
            print(f"SKIP: file not found: {p}")
            return 77
    workdir = a.workdir or tempfile.mkdtemp(prefix="forgert_img_")
    os.makedirs(workdir, exist_ok=True)

    cats, cats_path = load_categories()
    fgx_path = os.path.join(workdir, "squeezenet1.1-7.fgx")
    onnx_to_fgx.convert_file(model, fgx_path)
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    print(f"model: {os.path.basename(model)} sha256={sha256(model)}")
    print(f"ONNX Runtime {ort.__version__} (CPUExecutionProvider, default graph optimization); "
          f"onnx {onnx.__version__}; numpy {np.__version__}; Pillow {Image.__version__}; "
          f"Python {platform.python_version()}; {platform.platform()}")
    print(f"class names: {cats_path if cats else 'unavailable (indices only)'}")
    print(f"preprocessing: RGB, shorter side -> {RESIZE} (PIL bilinear), center-crop {CROP}, /255, "
          f"mean {[round(float(v), 3) for v in MEAN]}, std {[round(float(v), 3) for v in STD]}, NCHW float32")
    print("accuracy: NOT computed (no ground-truth labels); this is runtime-to-runtime agreement\n")

    rows, all_pass = [], True
    for i, path in enumerate(a.images):
        x, (w, h) = preprocess(path)
        ref = sess.run(None, {in_name: x})[0].reshape(-1)

        in_bin = os.path.join(workdir, f"image_{i}.input.bin")
        out_bin = os.path.join(workdir, f"image_{i}.output.bin")
        x.astype("<f4").tofile(in_bin)
        r = subprocess.run([a.runner, fgx_path, in_bin, out_bin], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAIL: {path}: forgert_run exited {r.returncode}: {r.stderr.strip()}")
            return 1
        got = np.fromfile(out_bin, dtype="<f4")
        if got.size != ref.size:
            print(f"FAIL: {path}: output has {got.size} values, ORT has {ref.size}")
            return 1

        d = np.abs(got.astype(np.float64) - ref.astype(np.float64))
        f1, o1 = int(got.argmax()), int(ref.argmax())
        f5, o5 = np.argsort(-got)[:5].tolist(), np.argsort(-ref)[:5].tolist()
        p = softmax(ref)
        srt = np.sort(ref)
        row = {
            "image": os.path.basename(path), "sha256": sha256(path), "orig_wh": (w, h),
            "max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "top1_forgert": f1, "top1_ort": o1,
            "name": cats[o1] if cats else "?",
            "ort_p_top1": float(p[o1]), "ort_margin": float(srt[-1] - srt[-2]),
            "top5_forgert": f5, "top5_ort": o5, "top5_same_order": f5 == o5,
            "top5_same_set": set(f5) == set(o5),
        }
        row["pass"] = bool(row["max_abs"] < a.atol and f1 == o1)
        all_pass &= row["pass"]
        rows.append(row)

    print("inputs")
    for r_ in rows:
        print(f"  {r_['image']:<18} {r_['orig_wh'][0]}x{r_['orig_wh'][1]}  sha256={r_['sha256']}")
    print("\nresults (per image; criteria: max|d| < %g and equal top-1)" % a.atol)
    print(f"{'image':<18} {'max|d|':>10} {'mean|d|':>10} {'top1 F/O':>11} {'ORT p(top1)':>12} {'ORT margin':>11} "
          f"{'top5 set':>9} {'top5 order':>11}  result")
    for r_ in rows:
        print(f"{r_['image']:<18} {r_['max_abs']:10.3e} {r_['mean_abs']:10.3e} "
              f"{r_['top1_forgert']:>5}/{r_['top1_ort']:<5} {r_['ort_p_top1']:12.4f} {r_['ort_margin']:11.4f} "
              f"{str(r_['top5_same_set']):>9} {str(r_['top5_same_order']):>11}  {'PASS' if r_['pass'] else 'FAIL'}")
    print("\npredicted classes (ONNX Runtime top-1; ForgeRT is identical when it passes)")
    for r_ in rows:
        print(f"  {r_['image']:<18} class {r_['top1_ort']:>3}  {r_['name']}")
        print(f"  {'':<18} top-5 ORT     {r_['top5_ort']}")
        print(f"  {'':<18} top-5 ForgeRT {r_['top5_forgert']}")
    worst = max(r_["max_abs"] for r_ in rows)
    print(f"\n{'PASS' if all_pass else 'FAIL'}: {sum(r_['pass'] for r_ in rows)}/{len(rows)} images within criteria; "
          f"worst max|d| = {worst:.3e} (limit {a.atol:g})")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
# SqueezeNet 1.1 Real-Image Check: ForgeRT vs ONNX Runtime

**Date:** 2026-10-21 (first run); re-run with the final script on 2026-10-21, identical results.  **Scope:** CPU only, batch 1, 3 images. This measures **agreement between
two runtimes**. It is **not** an accuracy measurement: no ground-truth labels were used.

## Result

3/3 images pass (criteria: max absolute logit difference < 1e-3 **and** equal top-1 class).

| Image | Size | max abs diff | mean abs diff | Top-1 ForgeRT / ORT | ORT p(top-1) | ORT top-1 margin | Top-5 set / order |
|---|---|---|---|---|---|---|---|
| `china.jpg` | 640x427 | 7.629e-06 | 1.420e-06 | 663 / 663 | 0.3914 | 0.0322 | same / same |
| `flower.jpg` | 640x427 | 1.144e-05 | 1.485e-06 | 574 / 574 | 0.5490 | 1.0499 | same / same |
| `grace_hopper.jpg` | 512x600 | 7.629e-06 | 1.434e-06 | 667 / 667 | 0.2020 | 0.0580 | same / same |

Worst difference: **1.144e-05** over all 3000 logits. "Margin" is the ORT logit gap between the
top two classes; the smallest (0.0322) is about 2,800x the largest observed difference, so the
top-1 agreement is not a near-tie coincidence.

Top-1 class names (from torchvision's ImageNet-1k list) as predicted by ONNX Runtime: 663
`monastery`, 574 `golf ball`, 667 `mortarboard`. ForgeRT's top-5 lists equal ORT's in membership and
order for every image. These names are what the models predict, **not** verified correct labels. I did
not inspect the images against them, and `golf ball` for `flower.jpg` was not investigated (see
Limitations).

## Inputs and provenance

All images are installed inside Python packages and were used in place (not copied into the repo).

| Image | Package | SHA-256 | Source / licence |
|---|---|---|---|
| `china.jpg` | scikit-learn 1.8.0 `sklearn/datasets/images/` | `8378025ad2519d649d02e32bd98990db4ab572357d9f09841c2fbfbb4fefad29` | Flickr, danielbuechele (photostream `6061409035`), CC BY 2.0, per the package README |
| `flower.jpg` | scikit-learn 1.8.0 `sklearn/datasets/images/` | `a77f6ec41e353afdf8bdff2ea981b2955535d8d83294f8cfa49cf4e423dd5638` | Flickr, vultilion (photostream `6056698931`), CC BY 2.0, per the package README |
| `grace_hopper.jpg` | matplotlib 3.10.8 `mpl-data/sample_data/` | `a8ca6d734765703b09728ab47fe59f473d93ae3967fc24c7c0288c3c7adb7130` | Bundled with matplotlib; its README gives **no source or licence** for this file |

The README labels `flower.jpg`'s attribution line as danielbuechele while its Flickr link is
vultilion; the link is cited above. The two scikit-learn images were retrieved by the package
authors in 2011, so provenance rests on that README, not on independent verification.

## Model and environment

- Model: `models/squeezenet1.1-7.onnx`, SHA-256
  `1eeff551a67ae8d565ca33b572fc4b66e3ef357b0eb2863bb9ff47a918cc4088` (ONNX Model Zoo, opset 7).
- ForgeRT: `build/apps/forgert_run.exe` on the model converted by `tools/onnx_to_fgx.py`
  (HEAD `7fcd386` with the files listed under "Files" added).
- Reference: ONNX Runtime 1.29.0, `CPUExecutionProvider`, default graph optimization.
- onnx 1.23.2, numpy 2.4.3, Pillow 12.1.1, Python 3.14.3, Windows 11 (10.0.26300).

## Preprocessing (identical tensor to both runtimes)

Each image is preprocessed once with Pillow and NumPy, written as raw little-endian float32 for
`forgert_run`, and passed as the same array to ONNX Runtime:

1. Open, convert to RGB (EXIF orientation not applied).
2. Resize so the shorter side is 256 px (PIL bilinear), keeping aspect ratio; the longer side is
   `round(long * 256 / short)`.
3. Center-crop 224x224 with offset `(size - 224) // 2`.
4. Scale to [0, 1]; subtract mean [0.485, 0.456, 0.406]; divide by std [0.229, 0.224, 0.225].
5. HWC to CHW, add batch dimension: float32 `[1, 3, 224, 224]`.

This follows the common ImageNet convention and is similar to torchvision's `Resize(256)`,
`CenterCrop(224)`, `Normalize`, but **not identical**: torchvision truncates the long side with
`int(256 * long / short)` and rounds the crop offset with `int(round(...))` (read from the installed
torchvision source), so the pixel grid can differ by one pixel. I did **not** check it against the ONNX
Model Zoo's documentation for this model (no network lookups were made), so whether it matches the
model's training preprocessing is unverified. It does not affect the ForgeRT-vs-ORT comparison,
because both runtimes receive the same tensor.

## Commands

PowerShell, from the repository root, with the MSVC environment active and the project already
built (`build/apps/forgert_run.exe` present):

```powershell
$site = python -c "import site; print(site.getusersitepackages())"   # or the system purelib if packages are not user-installed
python tools\validate_squeezenet_images.py --runner build\apps\forgert_run.exe `
    --images "$site\sklearn\datasets\images\china.jpg" `
             "$site\sklearn\datasets\images\flower.jpg" `
             "$site\matplotlib\mpl-data\sample_data\grace_hopper.jpg"
```

Class names are read as text from `torchvision/models/_meta.py` if torchvision is installed
(torchvision is not imported); otherwise only indices are printed. The script exits 0 on success,
1 on failure, and 77 (reported as skipped by CTest-style runners) if a package or file is missing.
This check is not registered with CTest.

## Limitations

- **Three images.** Small and not diverse; two come from scikit-learn's sample set and one from
  matplotlib's. The tensors are not saved or hashed, and a different Pillow/libjpeg build could decode
  or resample slightly differently (both runtimes would still receive the same tensor and agree).
- **No accuracy.** There are no trustworthy ground-truth labels for these images, so only
  agreement is reported.
- **`golf ball` for `flower.jpg` looks surprising**, but ONNX Runtime gives the same answer, so it is
  a property of the model or the preprocessing, not of ForgeRT. I did not investigate it, and the
  preprocessing is unconfirmed against the model's documented convention (see above).
- Only the CPU path, batch size 1, and this one model were exercised. CUDA execution of imported
  models is still untested.
- Latency was not measured here.
- After the first run the script received two edits: the mean/std print was reformatted (the first run
  printed float32 artifacts such as `0.48500001430511475`), and the docstring was corrected to describe
  the torchvision differences above. An AST comparison confirmed the executable code is unchanged. The
  validation was then re-run once with the final script (fresh working directory, exit code 0); its
  output matched the first run line for line except the constants line, so all results above are
  reproduced, not just recorded.

## Files

- `tools/validate_squeezenet_images.py` (new): the check above.
- `docs/SQUEEZENET_IMAGE_VALIDATION.md` (new): this report.

No ForgeRT runtime, importer, CUDA code, existing tests, or build configuration was changed.
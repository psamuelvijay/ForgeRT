#!/usr/bin/env python3
"""Download SqueezeNet 1.1 (ONNX opset 7) into models/ and verify its SHA-256.

The model is not committed (*.onnx is git-ignored). Source: ONNX Model Zoo,
Apache-2.0. An existing file is only verified, never overwritten.
"""
import hashlib
import os
import sys
import urllib.request

URL = ("https://github.com/onnx/models/raw/refs/heads/main/validated/vision/"
       "classification/squeezenet/model/squeezenet1.1-7.onnx")
SHA256 = "1eeff551a67ae8d565ca33b572fc4b66e3ef357b0eb2863bb9ff47a918cc4088"
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "squeezenet1.1-7.onnx")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    dest = os.path.normpath(DEST)
    if os.path.exists(dest):
        digest = sha256(dest)
        if digest != SHA256:
            print(f"error: {dest} exists but its SHA-256 is {digest}, expected {SHA256}", file=sys.stderr)
            return 1
        print(f"ok: {dest} already present, SHA-256 verified")
        return 0
    tmp = dest + ".part"
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, tmp)
    digest = sha256(tmp)
    if digest != SHA256:
        os.remove(tmp)
        print(f"error: downloaded file has SHA-256 {digest}, expected {SHA256}", file=sys.stderr)
        return 1
    os.replace(tmp, dest)
    print(f"ok: saved {dest} ({os.path.getsize(dest)} bytes), SHA-256 verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
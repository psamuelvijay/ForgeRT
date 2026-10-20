#!/usr/bin/env python3
"""Convert a static ONNX model to ForgeRT's .fgx v1 format.

Spec: docs/FGX_FORMAT.md. This converter is a faithful serializer: it rejects
only what .fgx cannot express and leaves the operator allowlist to the C++
importer.

Usage:
    python tools/onnx_to_fgx.py model.onnx model.fgx [--manifest model.manifest.txt]
"""
import argparse
import hashlib
import struct
import sys
import zlib

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, numpy_helper, shape_inference

MAGIC = b"FGXM"
VERSION_MAJOR, VERSION_MINOR = 1, 0
HEADER_SIZE = 24
MAX_FILE_SIZE = 1 << 30
MAX_STRING = 4096
MAX_COUNT = 1_000_000
MAX_RANK = 8
MAX_DIM = 2**31 - 1

# ONNX TensorProto code -> (.fgx dtype code, little-endian numpy dtype)
DTYPES = {
    TensorProto.FLOAT: (1, np.dtype("<f4")),
    TensorProto.INT64: (7, np.dtype("<i8")),
}
ELEM_SIZE = {1: 4, 7: 8}

# .fgx-specific attribute kinds (NOT the ONNX AttributeType numbering)
KIND_INT, KIND_FLOAT, KIND_INTS = 1, 2, 3
SECTION_ORDER = (b"META", b"INPT", b"OUTP", b"INIT", b"NODE", b"VINF")


class ConversionError(Exception):
    """The ONNX model uses something .fgx v1 cannot represent."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _check_string(what, s, allow_empty=False):
    b = s.encode("utf-8")
    if not b and not allow_empty:
        raise ConversionError(f"{what}: empty name")
    if len(b) > MAX_STRING:
        raise ConversionError(f"{what}: string longer than {MAX_STRING} bytes")
    return s


def _check_count(what, n):
    if n > MAX_COUNT:
        raise ConversionError(f"{what}: count {n} exceeds {MAX_COUNT}")


def _check_dims(what, dims):
    if len(dims) == 0:
        raise ConversionError(f"{what}: rank-0 tensors are not supported in .fgx v1")
    if len(dims) > MAX_RANK:
        raise ConversionError(f"{what}: rank {len(dims)} exceeds {MAX_RANK}")
    for d in dims:
        if d < 1 or d > MAX_DIM:
            raise ConversionError(f"{what}: dim {d} outside 1..{MAX_DIM}")
    return [int(d) for d in dims]


def _value_desc(what, vi):
    """(name, dtype code, dims) from a ValueInfoProto with a static shape."""
    name = _check_string(what, vi.name)
    if not vi.type.HasField("tensor_type"):
        raise ConversionError(f"{what} '{name}': not a tensor type")
    tt = vi.type.tensor_type
    if tt.elem_type not in DTYPES:
        raise ConversionError(
            f"{what} '{name}': data type {TensorProto.DataType.Name(tt.elem_type)} "
            "is not supported (float32, int64 only)")
    if not tt.HasField("shape"):
        raise ConversionError(f"{what} '{name}': shape is unknown")
    dims = []
    for d in tt.shape.dim:
        if d.WhichOneof("value") != "dim_value":
            raise ConversionError(f"{what} '{name}': symbolic or unknown dim "
                                  f"'{d.dim_param}' is not supported")
        dims.append(d.dim_value)
    return name, DTYPES[tt.elem_type][0], _check_dims(f"{what} '{name}'", dims)


# ---------------------------------------------------------------------------
# ONNX -> in-memory .fgx model
# ---------------------------------------------------------------------------

def convert_model(model, source_bytes):
    """Return a dict describing the .fgx content for `model`."""
    g = model.graph

    # Format-level rejections first, so errors are specific.
    for init in g.initializer:
        if init.data_location == TensorProto.EXTERNAL:
            raise ConversionError(f"initializer '{init.name}': external data is not supported")
        if init.data_type not in DTYPES:
            raise ConversionError(
                f"initializer '{init.name}': data type "
                f"{TensorProto.DataType.Name(init.data_type)} is not supported")
        _check_dims(f"initializer '{init.name}'", list(init.dims))
    for idx, n in enumerate(g.node):
        if n.domain not in ("", "ai.onnx"):
            raise ConversionError(f"node {idx} '{n.name}' ({n.op_type}): "
                                  f"operator domain '{n.domain}' is not supported")

    default_opsets = [o.version for o in model.opset_import if o.domain in ("", "ai.onnx")]
    if len(default_opsets) != 1:
        raise ConversionError("model must import exactly one default-domain opset")
    if not (0 <= model.ir_version <= 0xFFFFFFFF):
        raise ConversionError("ir_version does not fit in u32")

    init_names = {i.name for i in g.initializer}
    inputs = [_value_desc("input", vi) for vi in g.input if vi.name not in init_names]
    outputs = [_value_desc("output", vi) for vi in g.output]

    initializers = []
    for init in g.initializer:
        code, dt = DTYPES[init.data_type]
        arr = np.ascontiguousarray(numpy_helper.to_array(init), dtype=dt)
        dims = _check_dims(f"initializer '{init.name}'", list(init.dims))
        data = arr.tobytes()
        if len(data) != int(np.prod(dims)) * ELEM_SIZE[code]:
            raise ConversionError(f"initializer '{init.name}': data size does not match dims")
        initializers.append((_check_string("initializer", init.name), code, dims, data))

    nodes = []
    for idx, n in enumerate(g.node):
        where = f"node {idx} '{n.name}' ({n.op_type})"
        attrs = []
        for a in n.attribute:
            if a.type == AttributeProto.INT:
                attrs.append((a.name, KIND_INT, int(a.i)))
            elif a.type == AttributeProto.FLOAT:
                attrs.append((a.name, KIND_FLOAT, float(a.f)))
            elif a.type == AttributeProto.INTS:
                _check_count(f"{where} attribute '{a.name}'", len(a.ints))
                attrs.append((a.name, KIND_INTS, [int(v) for v in a.ints]))
            elif a.name == "auto_pad" and a.type == AttributeProto.STRING:
                if a.s != b"NOTSET":
                    raise ConversionError(f"{where}: auto_pad={a.s.decode()} is not supported "
                                          "(only explicit pads)")
                continue  # NOTSET is the ONNX default; nothing to carry
            else:
                raise ConversionError(
                    f"{where}: attribute '{a.name}' of type "
                    f"{AttributeProto.AttributeType.Name(a.type)} is not supported")
            _check_string(f"{where} attribute", a.name)
        for s in list(n.input) + list(n.output):
            _check_string(f"{where} input/output", s)
        _check_count(f"{where} inputs", len(n.input))
        _check_count(f"{where} outputs", len(n.output))
        nodes.append((_check_string(where, n.op_type),
                      _check_string(where, n.name, allow_empty=True),
                      list(n.input), list(n.output), attrs))

    # ONNX validity and shape inference after the format-level checks above,
    # so unsupported attribute types get a specific error.
    onnx.checker.check_model(model)
    inferred = shape_inference.infer_shapes(model, strict_mode=True)
    value_info = [_value_desc("value_info", vi) for vi in inferred.graph.value_info]

    for what, lst in (("inputs", inputs), ("outputs", outputs), ("initializers", initializers),
                      ("nodes", nodes), ("value_info", value_info)):
        _check_count(what, len(lst))

    return {
        "meta": (int(model.ir_version), int(default_opsets[0]),
                 _check_string("producer", model.producer_name, allow_empty=True),
                 hashlib.sha256(source_bytes).digest()),
        "inputs": inputs, "outputs": outputs, "initializers": initializers,
        "nodes": nodes, "value_info": value_info,
    }


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _str(s):
    b = s.encode("utf-8")
    return struct.pack("<I", len(b)) + b


def _tdesc(name, dtype, dims):
    return _str(name) + struct.pack("<II", dtype, len(dims)) + struct.pack(f"<{len(dims)}q", *dims)


def encode(fgx):
    ir, opset, producer, sha = fgx["meta"]
    meta = struct.pack("<II", ir, opset) + _str(producer) + sha

    def tdesc_list(lst):
        return struct.pack("<I", len(lst)) + b"".join(_tdesc(*t) for t in lst)

    init = struct.pack("<I", len(fgx["initializers"]))
    for name, dtype, dims, data in fgx["initializers"]:
        init += _tdesc(name, dtype, dims) + struct.pack("<Q", len(data)) + data

    node = struct.pack("<I", len(fgx["nodes"]))
    for op_type, name, ins, outs, attrs in fgx["nodes"]:
        node += _str(op_type) + _str(name)
        node += struct.pack("<I", len(ins)) + b"".join(_str(s) for s in ins)
        node += struct.pack("<I", len(outs)) + b"".join(_str(s) for s in outs)
        node += struct.pack("<I", len(attrs))
        for aname, kind, value in attrs:
            node += _str(aname) + struct.pack("<B", kind)
            if kind == KIND_INT:
                node += struct.pack("<q", value)
            elif kind == KIND_FLOAT:
                node += struct.pack("<f", value)
            else:
                node += struct.pack("<I", len(value)) + struct.pack(f"<{len(value)}q", *value)

    contents = (meta, tdesc_list(fgx["inputs"]), tdesc_list(fgx["outputs"]), init, node,
                tdesc_list(fgx["value_info"]))
    payload = b"".join(tag + struct.pack("<Q", len(c)) + c
                       for tag, c in zip(SECTION_ORDER, contents))
    total = HEADER_SIZE + len(payload)
    if total > MAX_FILE_SIZE:
        raise ConversionError(f"output would be {total} bytes, above the 1 GiB limit")
    header = MAGIC + struct.pack("<HHIQI", VERSION_MAJOR, VERSION_MINOR, 0, total,
                                 zlib.crc32(payload) & 0xFFFFFFFF)
    assert len(header) == HEADER_SIZE
    return header + payload


# ---------------------------------------------------------------------------
# Canonical manifest (must match the C++ fgx_dump output byte for byte)
# ---------------------------------------------------------------------------

def _esc(s):
    out = []
    for byte in s.encode("utf-8"):
        out.append(chr(byte) if 0x21 <= byte <= 0x7E and byte != 0x5C else f"\\x{byte:02x}")
    return "".join(out)


def _dims_text(dims):
    return "x".join(str(d) for d in dims)


def manifest(fgx):
    ir, opset, producer, sha = fgx["meta"]
    lines = [f"FGX {VERSION_MAJOR}.{VERSION_MINOR}",
             f"META ir={ir} opset={opset} producer={_esc(producer)} sha256={sha.hex()}"]
    for tag, key in (("INPT", "inputs"), ("OUTP", "outputs")):
        lines += [f"{tag} {_esc(n)} dtype={d} dims={_dims_text(s)}" for n, d, s in fgx[key]]
    for name, dtype, dims, data in fgx["initializers"]:
        lines.append(f"INIT {_esc(name)} dtype={dtype} dims={_dims_text(dims)} "
                     f"bytes={len(data)} crc32={zlib.crc32(data) & 0xFFFFFFFF:08x}")
    for idx, (op_type, name, ins, outs, attrs) in enumerate(fgx["nodes"]):
        parts = []
        for aname, kind, value in attrs:
            if kind == KIND_INT:
                parts.append(f"{_esc(aname)}:int={value}")
            elif kind == KIND_FLOAT:
                bits = struct.unpack("<I", struct.pack("<f", value))[0]
                parts.append(f"{_esc(aname)}:float=0x{bits:08x}")
            else:
                parts.append(f"{_esc(aname)}:ints=[{','.join(str(v) for v in value)}]")
        lines.append(f"NODE {idx} op={_esc(op_type)} name={_esc(name)} "
                     f"in={','.join(_esc(s) for s in ins)} out={','.join(_esc(s) for s in outs)} "
                     f"attrs={';'.join(parts)}")
    lines += [f"VINF {_esc(n)} dtype={d} dims={_dims_text(s)}" for n, d, s in fgx["value_info"]]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Self-check reader (round trip only; the safety-checked reader is the C++ one)
# ---------------------------------------------------------------------------

def read_fgx(buf):
    """Parse .fgx bytes back into the convert_model() dict structure."""
    if buf[:4] != MAGIC:
        raise ValueError("bad magic")
    major, minor, flags, total, crc = struct.unpack_from("<HHIQI", buf, 4)
    if (major, minor, flags) != (VERSION_MAJOR, VERSION_MINOR, 0) or total != len(buf):
        raise ValueError("bad header")
    if zlib.crc32(buf[HEADER_SIZE:]) & 0xFFFFFFFF != crc:
        raise ValueError("crc mismatch")
    pos = HEADER_SIZE

    def take(fmt):
        nonlocal pos
        vals = struct.unpack_from(fmt, buf, pos)
        pos += struct.calcsize(fmt)
        return vals if len(vals) > 1 else vals[0]

    def s():
        nonlocal pos
        n = take("<I")
        v = buf[pos:pos + n].decode("utf-8")
        pos += n
        return v

    def tdesc():
        name = s()
        dtype, rank = take("<II")
        return name, dtype, list(take(f"<{rank}q")) if rank > 1 else [take("<q")]

    out = {}
    for tag in SECTION_ORDER:
        if buf[pos:pos + 4] != tag:
            raise ValueError(f"expected section {tag!r}")
        pos += 4
        end = pos + 8 + take("<Q")
        if tag == b"META":
            ir, opset = take("<II")
            producer = s()
            out["meta"] = (ir, opset, producer, bytes(buf[pos:pos + 32]))
            pos += 32
        elif tag in (b"INPT", b"OUTP", b"VINF"):
            key = {b"INPT": "inputs", b"OUTP": "outputs", b"VINF": "value_info"}[tag]
            out[key] = [tdesc() for _ in range(take("<I"))]
        elif tag == b"INIT":
            inits = []
            for _ in range(take("<I")):
                name, dtype, dims = tdesc()
                n = take("<Q")
                inits.append((name, dtype, dims, bytes(buf[pos:pos + n])))
                pos += n
            out["initializers"] = inits
        else:
            nodes = []
            for _ in range(take("<I")):
                op_type, name = s(), s()
                ins = [s() for _ in range(take("<I"))]
                outs = [s() for _ in range(take("<I"))]
                attrs = []
                for _ in range(take("<I")):
                    aname = s()
                    kind = take("<B")
                    if kind == KIND_INT:
                        attrs.append((aname, kind, take("<q")))
                    elif kind == KIND_FLOAT:
                        attrs.append((aname, kind, take("<f")))
                    else:
                        cnt = take("<I")
                        vals = take(f"<{cnt}q") if cnt else ()
                        attrs.append((aname, kind, list(vals) if isinstance(vals, tuple) else [vals]))
                nodes.append((op_type, name, ins, outs, attrs))
            out["nodes"] = nodes
        if pos != end:
            raise ValueError(f"section {tag!r} length mismatch")
    if pos != len(buf):
        raise ValueError("trailing bytes")
    return out


def _canonical(fgx):
    """Normalize FLOAT attributes to f32 bit patterns for exact comparison."""
    def norm_attr(a):
        name, kind, v = a
        return (name, kind, struct.pack("<f", v) if kind == KIND_FLOAT else v)
    nodes = [(o, n, i, out, [norm_attr(a) for a in attrs])
             for o, n, i, out, attrs in fgx["nodes"]]
    return {**fgx, "nodes": nodes}


def convert_file(src, dst, manifest_path=None, self_check=True):
    source_bytes = open(src, "rb").read()
    model = onnx.load_from_string(source_bytes)
    fgx = convert_model(model, source_bytes)
    data = encode(fgx)
    if self_check and _canonical(read_fgx(data)) != _canonical(fgx):
        raise ConversionError("self-check failed: re-parsed file differs from the converted model")
    with open(dst, "wb") as f:
        f.write(data)
    if manifest_path:
        with open(manifest_path, "w", encoding="ascii", newline="\n") as f:
            f.write(manifest(fgx))
    return fgx, len(data)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("onnx_model")
    ap.add_argument("output_fgx")
    ap.add_argument("--manifest", help="also write the canonical manifest text")
    ap.add_argument("--no-self-check", action="store_true")
    args = ap.parse_args(argv)
    try:
        fgx, size = convert_file(args.onnx_model, args.output_fgx, args.manifest,
                                 not args.no_self_check)
    except (ConversionError, onnx.checker.ValidationError,
            onnx.shape_inference.InferenceError) as e:
        print(f"onnx_to_fgx: error: {e}", file=sys.stderr)
        return 1
    print(f"wrote {args.output_fgx}: {size} bytes, {len(fgx['nodes'])} nodes, "
          f"{len(fgx['initializers'])} initializers, {len(fgx['value_info'])} value_info, "
          f"opset {fgx['meta'][1]}"
          + ("" if args.no_self_check else ", self-check passed"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
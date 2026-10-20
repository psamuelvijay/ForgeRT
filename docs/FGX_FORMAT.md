# ForgeRT `.fgx` Format, Version 1.0

`.fgx` carries a static ONNX inference graph from a build-time Python converter
(`tools/onnx_to_fgx.py`, which uses the `onnx` package) to a dependency-free C++
reader (`src/import/fgx_reader.cpp`). The format is deliberately small: two data
types, three attribute kinds, six sections. Malformed-file cases are listed in
`.kiro/specs/onnx-importer-pilot/fgx-review.md`.

## 1. Encoding rules

- **Byte order.** Every multi-byte integer and float is little-endian, with no
  padding or alignment anywhere in the file. The C++ reader refuses to run on a
  big-endian host, because initializer bytes are used in place as host floats.
- **Floats.** `f32` is IEEE-754 binary32. Bit patterns, including NaN and Inf,
  are preserved exactly.
- **Strings.** `u32 length` then `length` bytes of UTF-8, no terminator.
  `length` is 0..4096. The reader validates UTF-8 (no overlong forms, no
  surrogates, nothing above U+10FFFF). Only `META.producer` and `NODE.node_name`
  may be empty; every other string must be non-empty.
- **Counts.** Every list count (`u32`) is at most 1,000,000, and the reader also
  checks that `count * minimum_entry_size` fits in the bytes remaining before it
  reserves memory.
- **Data types** use the ONNX `TensorProto.DataType` codes:

  | Code | Type | Element size |
  |---|---|---|
  | 1 | float32 | 4 |
  | 7 | int64 | 8 |

  Any other code is rejected.
- **Tensor descriptor** (`TDESC`): `string name`, `u32 dtype`, `u32 rank`,
  `i64 dims[rank]`. Rank is 1..8; **rank 0 is rejected in v1**. Every dim is
  1..2,147,483,647 (static shapes only; no symbolic or zero dims). The product
  of the dims times the element size must not overflow `u64`.

## 2. Header (24 bytes)

| Offset | Type | Field | Rule |
|---|---|---|---|
| 0 | 4 bytes | magic | ASCII `F` `G` `X` `M` |
| 4 | u16 | version_major | must be 1 |
| 6 | u16 | version_minor | must be 0 for a v1.0 reader (a newer minor is rejected) |
| 8 | u32 | flags | must be 0 |
| 12 | u64 | total_file_size | must equal the actual file size |
| 20 | u32 | payload_crc32 | CRC-32 (ISO-HDLC, as `zlib.crc32`) of bytes 24..EOF |

Maximum file size: 1 GiB.

## 3. Sections

Exactly six sections follow the header, once each, in this order:
`META`, `INPT`, `OUTP`, `INIT`, `NODE`, `VINF`.

Each section is: 4 ASCII tag bytes in file order, `u64 byte_length`, then
`byte_length` bytes of content. The parsed content must consume exactly
`byte_length` bytes, and the file must end exactly after `VINF`.

| Section | Content |
|---|---|
| `META` | `u32 onnx_ir_version`, `u32 opset_version` (default domain), `string producer`, `u8 source_sha256[32]` (SHA-256 of the source `.onnx` file bytes) |
| `INPT` | `u32 count`, then `count` x `TDESC`. Graph inputs that are **not** initializers. |
| `OUTP` | `u32 count`, then `count` x `TDESC`. Graph outputs. |
| `INIT` | `u32 count`, then per entry: `TDESC`, `u64 byte_length`, `byte_length` raw bytes. `byte_length` must equal `prod(dims) * element_size`. Bytes are little-endian, C-contiguous. |
| `NODE` | `u32 count`, then per node in topological order: `string op_type`, `string node_name`, `u32 n_in` + `n_in` strings, `u32 n_out` + `n_out` strings, `u32 n_attr` + `n_attr` attributes. `n_out` >= 1. |
| `VINF` | `u32 count`, then `count` x `TDESC`. Expected shapes of intermediate values (from `onnx.shape_inference.infer_shapes`). |

**Attribute:** `string name`, `u8 kind`, value.

| Kind | Meaning | Value encoding |
|---|---|---|
| 1 | INT | `i64` |
| 2 | FLOAT | `f32` |
| 3 | INTS | `u32 count`, then `count` x `i64` |

These kind codes are **.fgx-specific** and are not ONNX's `AttributeProto.AttributeType`
values (ONNX uses FLOAT=1, INT=2, INTS=7). The converter maps them explicitly.
Any other kind is rejected.

## 4. Graph well-formedness (enforced by the reader)

- INPT, OUTP, INIT and VINF names are unique within their section.
- No INPT name equals an INIT name.
- Every node output name is produced by exactly one node and does not equal an
  INPT or INIT name.
- Every node input is an INPT name, an INIT name, or an output of an **earlier**
  node (empty input names, ONNX's "omitted optional input", are rejected in v1).
- Every OUTP name is an INPT name or a node output.
- Attribute names are unique within a node.

## 5. Check order and error contract

The reader checks: host byte order -> file size (>= 24, <= 1 GiB) -> magic ->
version -> flags -> `total_file_size` -> CRC -> sections in order -> graph
well-formedness.

The CRC detects accidental corruption only. It is not a security boundary, so
every bounds and limit check above still applies to files with a valid CRC.
Length fields are checked against the bytes remaining before any allocation.

On any violation the reader throws `forgert::fgx::FormatError`, which carries an
error category plus a message naming the section, entry index and byte offset.
No partially parsed model is ever returned.

## 6. Division of responsibility

- **Converter** (`tools/onnx_to_fgx.py`): a faithful serializer. It reads tensor
  data through `onnx.numpy_helper.to_array()` (SqueezeNet 1.1 stores weights in
  `float_data`, not `raw_data`) and runs shape inference to fill `VINF` (the
  source model has no `value_info`). It rejects only what the format cannot
  express: external tensor data, non-default operator domains, symbolic or zero
  dims, rank-0 tensors, data types other than float32/int64, and attribute
  types other than INT/FLOAT/INTS. It drops `auto_pad="NOTSET"` (the ONNX
  default) and rejects any other `auto_pad`. It has no operator allowlist.
- **Reader** (`src/import/fgx_reader.cpp`): format and well-formedness only. It
  keeps int64 as dtype 7.
- **Importer** (`src/import/importer.cpp`): the authoritative operator and
  attribute allowlist (opset 7 only), int64 range checks, Reshape target
  extraction, and exact shape cross-checks against `VINF` and `OUTP`.

## 7. Canonical manifest (cross-language check)

`onnx_to_fgx.py --manifest` writes this text from the converter's in-memory
model, and the C++ `fgx_dump` tool writes it from a parsed file. The two must be
byte-identical. Lines end with `\n`; entries appear in file order.

```
FGX 1.0
META ir=<u32> opset=<u32> producer=<s> sha256=<64 lowercase hex>
INPT <s> dtype=<u32> dims=<d0>x<d1>...
OUTP <s> dtype=<u32> dims=<d0>x<d1>...
INIT <s> dtype=<u32> dims=<d0>x<d1>... bytes=<u64> crc32=<8 lowercase hex>
NODE <index> op=<s> name=<s> in=<s>,<s>,... out=<s>,... attrs=<a>;<a>;...
VINF <s> dtype=<u32> dims=<d0>x<d1>...
```

- `<s>`: each byte in 0x21..0x7E other than `\` is written as-is; every other
  byte is written as `\xHH` (lowercase hex). An empty string writes nothing.
- `<a>`: `name:int=<i64>`, `name:float=0x<8 lowercase hex of the f32 bits>`,
  or `name:ints=[<i64>,<i64>,...]` (`[]` when empty).
- `crc32` is the CRC-32 of that initializer's data bytes.
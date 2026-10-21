# .fgx v1 Format — Draft for Review

Purpose: carry a static ONNX graph (SqueezeNet 1.1 scope) from a build-time Python
converter (`onnx` package) to a dependency-free C++ reader.

## Conventions
- All integers little-endian. Reader rejects big-endian hosts at startup.
- Strings: `u32 length` + UTF-8 bytes, no terminator, length <= 4096.
- Dims are `i64`; v1 requires every dim >= 1 (static shapes only, no symbolic dims).
- Reader caps: rank <= 8, any count <= 1,000,000, every byte length checked against
  remaining section bytes before allocation; dim products checked for overflow.

## Header (24 bytes)
| Offset | Type | Field | Rule |
|---|---|---|---|
| 0 | char[4] | magic = `FGXM` | reject otherwise |
| 4 | u16 | version_major = 1 | reject != 1 |
| 6 | u16 | version_minor = 0 | accept <= reader's known minor |
| 8 | u32 | flags = 0 | reject nonzero |
| 12 | u64 | total_file_size | must equal actual file size |
| 20 | u32 | payload_crc32 | CRC-32 (zlib polynomial) of bytes [24, EOF) |

## Sections
Fixed order: META, INPT, OUTP, INIT, NODE, VINF. Each section is
`u32 tag` (four-cc) + `u64 byte_length` + content. The reader must consume exactly
`byte_length` bytes per section and reach EOF exactly after VINF.

- META: `u32 onnx_ir_version`, `u32 opset_version`, `string producer`, `u8[32] source_sha256`
- INPT / OUTP: `u32 count`; each = `string name`, `u32 dtype`, `u32 rank`, `i64 dims[rank]`
  (INPT excludes names that are initializers)
- INIT: `u32 count`; each = `string name`, `u32 dtype` (ONNX TensorProto enum:
  1 = float32, 7 = int64), `u32 rank`, `i64 dims[rank]`, `u64 byte_length`, raw bytes.
  `byte_length` must equal `prod(dims) * elem_size(dtype)`. Converter produces raw
  bytes from `numpy_helper.to_array()` (SqueezeNet stores weights in `float_data`,
  not `raw_data`), forced to little-endian contiguous.
- NODE: `u32 count`, topological order; each = `string op_type`, `string node_name`,
  `u32 n_in` + names, `u32 n_out` + names, `u32 n_attr`; attribute = `string name`,
  `u8 kind` (1 INT -> i64, 2 FLOAT -> f32, 3 INTS -> u32 count + i64[]).
  Any other ONNX attribute type is a converter error.
- VINF: `u32 count`; each = `string value_name`, `u32 dtype`, `u32 rank`, `i64 dims[rank]`.
  Produced by `onnx.shape_inference.infer_shapes` (the source model has zero
  value_info entries). Used by the C++ importer to cross-check `inferOutputShapes()`.

## Policy split
- Converter: faithful serializer. No op allowlist; rejects only things the format
  cannot represent (external data, non-INT/FLOAT/INTS attributes, symbolic dims,
  non-default domains).
- C++ importer: authoritative op/attribute allowlist, int64 range checks, shape
  cross-check, clear errors naming node index, node name, op type.
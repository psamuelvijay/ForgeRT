# .fgx v1 Draft Review (requirement-detailer subagent, pilot run)

Source: QA analysis of `fgx-format-draft.md` by the `requirement-detailer` subagent.
The subagent returned its findings in conversation and wrote no files; this document
preserves them verbatim in substance. Resolutions adopted in `docs/FGX_FORMAT.md` are
marked **[adopted]**.

## Ambiguities found
1. Attribute kind codes (1 INT, 2 FLOAT, 3 INTS) differ from ONNX's enum (FLOAT=1, INT=2, INTS=7); copying `attr.type` would silently swap INT/FLOAT. **[adopted: codes documented as .fgx-specific; converter maps explicitly]**
2. Section tag byte order undefined (ASCII bytes vs u32 value). **[adopted: four ASCII bytes in file order]**
3. dtype enum for INPT/OUTP/VINF unspecified. **[adopted: ONNX TensorProto 1/7 everywhere]**
4. Where int64 becomes Int32 unspecified. **[adopted: reader keeps dtype 7; importer narrows with range check]**
5. Rank 0 unclear. **[adopted: rank 0 rejected in v1]**
6. Scope of the 1,000,000 count limit unclear; no per-dim cap. **[adopted: applies to all list counts; dims 1..2^31-1]**
7. Empty strings unspecified. **[adopted: only producer and node_name may be empty]**
8. Name uniqueness unspecified. **[adopted: reader enforces uniqueness rules]**
9. Topological order stated but unchecked. **[adopted: reader rejects use-before-definition]**
10. CRC check order and its non-security role unspecified. **[adopted: header -> size -> CRC -> sections; bounds checks still apply]**
11. Minor-version semantics conflict with exact-EOF rule. **[adopted: v1.0 reader rejects minor > 0]**
12. META `source_sha256` input and `ir_version` narrowing unspecified. **[adopted: SHA-256 of source .onnx bytes; ir_version must fit u32]**
13. Float encoding unspecified. **[adopted: IEEE-754 binary32 LE, bit patterns preserved]**
14. "Oversized" undefined. **[adopted: 1 GiB maximum]**

## Missing safety checks
1. No maximum file size. 2. No upper bound per dim. 3. `prod(dims) * elem_size` overflow.
4. Unknown dtype / attribute kind codes. 5. UTF-8 validity. 6. Section structure
(missing, duplicate, unknown, out-of-order, under/over-consumed, trailing bytes).
7. Name rules (empty, duplicate, input/initializer overlap, multiple producers,
use-before-definition, duplicate attribute names). 8. Error contract (no partial graph;
error names section, entry index, byte offset). 9. Check order; bounds checks apply even
with valid CRC. 10. Big-endian host check point. **[all adopted]**

## Malformed-file test cases
Each case starts from a valid file and changes one thing; payload_crc32 is recomputed
except in case 15. Every case must fail, return no graph, and name the failing check.

| # | Mutation | Expected rejection |
|---|---|---|
| 1 | 0-byte file | truncated |
| 2 | 23-byte file | truncated |
| 3 | magic `FGXN` | bad magic |
| 4 | magic `MXGF` | bad magic |
| 5 | version_major = 2 | bad version |
| 6 | version_major = 0 | bad version |
| 7 | version_minor = 1 | bad version |
| 8 | flags = 1 | bad flags |
| 9 | total_file_size = size + 1 | size mismatch |
| 10 | total_file_size = size - 1 | size mismatch |
| 11 | truncated by 1 byte, header unchanged | size mismatch |
| 12 | 1 trailing byte, total_file_size updated | trailing bytes |
| 13 | file > 1 GiB | oversized |
| 14 | total_file_size = 2^64-1 | size mismatch |
| 15 | payload byte flipped, CRC not recomputed | CRC mismatch |
| 16 | first section INPT instead of META | section order |
| 17 | INIT and NODE swapped | section order |
| 18 | META twice | section order |
| 19 | unknown tag `XXXX` in place of VINF | section order |
| 20 | VINF missing | section order |
| 21 | section byte_length > remaining | section length |
| 22 | section byte_length 4 more than content | section length |
| 23 | section byte_length less than content | section length |
| 24 | string length 4097 | string too long |
| 25 | string length 0xFFFFFFFF | string too long / exceeds remaining, no allocation |
| 26 | invalid UTF-8 (0xFF) | invalid UTF-8 |
| 27 | empty INIT name | empty name |
| 28 | INPT count 1,000,001 | count limit |
| 29 | INIT count 1,000,000 with few bytes left | count limit / exceeds remaining, no allocation |
| 30 | rank 9 | rank limit |
| 31 | dim 0 | dim range |
| 32 | dim -1 | dim range |
| 33 | dim 2^31 | dim range |
| 34 | `prod(dims) * 4` overflows u64 | overflow |
| 35 | INIT byte_length = prod*4 - 1 | byte length |
| 36 | INIT byte_length correct but > remaining | section length / no allocation |
| 37 | INIT dtype 10 | unsupported dtype |
| 38 | int64 INIT with byte_length = 2*4 | byte length |
| 39 | OUTP dtype 0 | unsupported dtype |
| 40 | attribute kind 4 | unsupported attribute kind |
| 41 | attribute kind 7 (ONNX INTS code) | unsupported attribute kind |
| 42 | INTS count 1,000,001 | count limit |
| 43 | duplicate attribute name in one node | duplicate name |
| 44 | duplicate INIT name | duplicate name |
| 45 | INPT name equals INIT name | duplicate name |
| 46 | two nodes produce the same output | duplicate name |
| 47 | node input produced by a later node | undefined input |
| 48 | node input defined nowhere | undefined input |
| 49 | empty node output name | empty name |
| 50 | valid file on a big-endian host | host endianness (not runnable on x86) |

Additional case adopted for v1: rank 0 (empty dims) -> rank limit.
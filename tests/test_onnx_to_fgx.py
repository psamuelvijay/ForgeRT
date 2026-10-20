#!/usr/bin/env python3
"""Unit tests for tools/onnx_to_fgx.py on small synthetic ONNX models.

Exit codes: 0 pass, 1 fail, 77 skipped (onnx / numpy not installed).
"""
import os
import struct
import sys
import zlib

try:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError as e:  # pragma: no cover
    print(f"SKIP: {e}")
    sys.exit(77)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import onnx_to_fgx as c  # noqa: E402

failures = 0


def ok(cond, msg):
    global failures
    print(("PASS: " if cond else "FAIL: ") + msg)
    if not cond:
        failures += 1


def make_model(nodes=None, inits=None, inputs=None, outputs=None, opset=7, domain_opsets=()):
    w = numpy_helper.from_array(np.arange(6, dtype=np.float32).reshape(3, 2, 1, 1), "w")
    b = numpy_helper.from_array(np.array([0.5, -0.5, 0.25], dtype=np.float32), "b")
    shape = numpy_helper.from_array(np.array([0, -1], dtype=np.int64), "shape")
    nodes = nodes if nodes is not None else [
        helper.make_node("Conv", ["x", "w", "b"], ["y"], name="conv",
                         kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0],
                         dilations=[1, 1], group=1),
        helper.make_node("Dropout", ["y"], ["d"], name="drop", ratio=0.5),
        helper.make_node("Reshape", ["d", "shape"], ["z"], name="flat"),
    ]
    g = helper.make_graph(
        nodes, "t",
        inputs if inputs is not None else [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 2, 2])],
        outputs if outputs is not None else [helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 12])],
        initializer=inits if inits is not None else [w, b, shape])
    opsets = [helper.make_opsetid("", opset)] + [helper.make_opsetid(d, v) for d, v in domain_opsets]
    m = helper.make_model(g, opset_imports=opsets, producer_name="unit-test")
    m.ir_version = 7
    return m


def rejects(model, fragment, msg):
    try:
        c.convert_model(model, b"src")
    except c.ConversionError as e:
        ok(fragment in str(e), f"{msg} (error: {e})")
        return
    except Exception as e:  # checker / shape-inference errors are also rejections
        ok(fragment == "*", f"{msg} (raised {type(e).__name__}: {e})")
        return
    ok(False, f"{msg}: conversion unexpectedly succeeded")


# ---- valid model: structure, kinds, bytes, round trip ----
m = make_model()
fgx = c.convert_model(m, b"src-bytes")
ok(fgx["meta"][:3] == (7, 7, "unit-test"), "META carries ir, opset, producer")
ok(fgx["meta"][3] == __import__("hashlib").sha256(b"src-bytes").digest(), "META sha256 hashes the source bytes")
ok([i[0] for i in fgx["inputs"]] == ["x"], "INPT excludes initializer names")
ok([i[0] for i in fgx["initializers"]] == ["w", "b", "shape"], "INIT keeps initializer order")
ok(fgx["initializers"][2][1] == 7 and fgx["initializers"][2][3] == struct.pack("<2q", 0, -1),
   "int64 initializer stored as dtype 7 with little-endian bytes")
ok(fgx["initializers"][0][3] == np.arange(6, dtype="<f4").tobytes(), "float32 initializer bytes exact")
conv_attrs = {a[0]: a[1] for a in fgx["nodes"][0][4]}
ok(conv_attrs == {"kernel_shape": 3, "strides": 3, "pads": 3, "dilations": 3, "group": 1},
   "attribute kinds: INTS -> 3, INT -> 1 (not ONNX codes)")
ok(fgx["nodes"][1][4] == [("ratio", 2, 0.5)], "FLOAT attribute kind -> 2 (not ONNX code 1)")
ok({v[0] for v in fgx["value_info"]} == {"y", "d"}, "VINF filled by shape inference")
data = c.encode(fgx)
ok(data[:4] == b"FGXM" and struct.unpack_from("<HHIQ", data, 4) == (1, 0, 0, len(data)),
   "header magic, version, flags, total size")
ok(struct.unpack_from("<I", data, 20)[0] == zlib.crc32(data[24:]) & 0xFFFFFFFF, "header CRC covers payload")
ok([data[24:28]] == [b"META"], "first section tag stored as ASCII bytes")
ok(c._canonical(c.read_fgx(data)) == c._canonical(fgx), "round trip through read_fgx is exact")
man = c.manifest(fgx)
ok("ratio:float=0x3f000000" in man and "group:int=1" in man and "shape dtype=7 dims=2" in man,
   "manifest encodes floats as bit patterns")

# float_data storage (as in SqueezeNet) is read correctly
fd = helper.make_tensor("w", TensorProto.FLOAT, [3, 2, 1, 1], list(range(6)))  # float_data, not raw_data
ok(not fd.raw_data and len(fd.float_data) == 6, "test tensor uses float_data storage")
m2 = make_model(inits=[fd, numpy_helper.from_array(np.zeros(3, np.float32), "b"),
                       numpy_helper.from_array(np.array([0, -1], np.int64), "shape")])
ok(c.convert_model(m2, b"")["initializers"][0][3] == np.arange(6, dtype="<f4").tobytes(),
   "float_data initializer converted to exact bytes")

# auto_pad=NOTSET is dropped, other auto_pad values rejected
m3 = make_model()
m3.graph.node[0].attribute.append(helper.make_attribute("auto_pad", "NOTSET"))
ok("auto_pad" not in {a[0] for a in c.convert_model(m3, b"")["nodes"][0][4]}, "auto_pad=NOTSET dropped")
m4 = make_model()
del m4.graph.node[0].attribute[:]
m4.graph.node[0].attribute.extend([helper.make_attribute("auto_pad", "SAME_UPPER"),
                                   helper.make_attribute("kernel_shape", [1, 1])])
rejects(m4, "auto_pad=SAME_UPPER", "auto_pad=SAME_UPPER rejected")

# ---- rejections ----
m5 = make_model()
m5.graph.node[1].attribute.append(helper.make_attribute("note", "text"))
rejects(m5, "type STRING", "STRING attribute rejected")
m6 = make_model(inits=[numpy_helper.from_array(np.arange(6, dtype=np.float32).reshape(3, 2, 1, 1), "w"),
                       numpy_helper.from_array(np.zeros(3, np.float32), "b"),
                       numpy_helper.from_array(np.array([0, -1], np.int64), "shape"),
                       numpy_helper.from_array(np.array(1.0, np.float32), "scalar")])
rejects(m6, "rank-0", "rank-0 initializer rejected")
m7 = make_model(inits=[numpy_helper.from_array(np.zeros((3, 2, 1, 1), np.float16), "w"),
                       numpy_helper.from_array(np.zeros(3, np.float32), "b"),
                       numpy_helper.from_array(np.array([0, -1], np.int64), "shape")])
rejects(m7, "FLOAT16", "float16 initializer rejected")
m8 = make_model(inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 2, 2, 2])],
                outputs=[helper.make_tensor_value_info("z", TensorProto.FLOAT, ["N", 12])])
rejects(m8, "symbolic", "symbolic input dim rejected")
m9 = make_model()
m9.graph.initializer[0].data_location = TensorProto.EXTERNAL
rejects(m9, "external data", "external initializer data rejected")
m10 = make_model(domain_opsets=[("com.example", 1)])
m10.graph.node.append(helper.make_node("Custom", ["z"], ["q"], domain="com.example"))
rejects(m10, "domain", "non-default operator domain rejected")
m11 = make_model(inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 0, 2])],
                 outputs=[helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 0])])
rejects(m11, "outside 1..", "zero dim rejected")

# No operator allowlist in the converter: an op the importer does not support still converts.
m12 = make_model(nodes=[helper.make_node("Sigmoid", ["x"], ["z"], name="sig")], inits=[],
                 outputs=[helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2, 2, 2])])
ok(c.convert_model(m12, b"")["nodes"][0][0] == "Sigmoid", "converter has no op allowlist (importer decides)")

print(f"\n{'All converter tests PASSED' if failures == 0 else f'{failures} converter test(s) FAILED'}")
sys.exit(1 if failures else 0)
/*
 * test_importer.cpp  --  .fgx model -> ForgeRT Graph (ONNX opset 7 subset)
 *
 * Uses in-memory fgx::Model structs. A tiny model with every supported op is
 * imported, run, and compared against a hand-written reference; then each
 * error path is hit by changing one thing.
 */
#include "forgert/import/importer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

using namespace forgert;
using fgx::Attribute;

static int g_fail = 0;
static void ok(bool cond, const std::string& msg) {
    std::cout << (cond ? "PASS: " : "FAIL: ") << msg << "\n";
    if (!cond) ++g_fail;
}

static Attribute A_ints(const std::string& n, std::vector<int64_t> v) { Attribute a; a.name = n; a.kind = fgx::kAttrInts; a.ints = std::move(v); return a; }
static Attribute A_int(const std::string& n, int64_t v) { Attribute a; a.name = n; a.kind = fgx::kAttrInt; a.i = v; return a; }
static Attribute A_float(const std::string& n, float v) { Attribute a; a.name = n; a.kind = fgx::kAttrFloat; a.f = v; return a; }

static fgx::TensorDesc TD(const std::string& n, std::vector<int64_t> d, uint32_t dt = fgx::kDtypeFloat32) {
    fgx::TensorDesc t; t.name = n; t.dtype = dt; t.dims = std::move(d); return t;
}
static fgx::Initializer floatInit(const std::string& n, std::vector<int64_t> d, std::vector<float> v) {
    fgx::Initializer i; i.desc = TD(n, std::move(d)); i.data.resize(v.size() * 4);
    std::memcpy(i.data.data(), v.data(), i.data.size()); return i;
}
static fgx::Initializer int64Init(const std::string& n, std::vector<int64_t> v) {
    fgx::Initializer i; i.desc = TD(n, {static_cast<int64_t>(v.size())}, fgx::kDtypeInt64);
    i.data.resize(v.size() * 8); std::memcpy(i.data.data(), v.data(), i.data.size()); return i;
}
static fgx::Node N(const std::string& op, const std::string& name, std::vector<std::string> in,
                   std::vector<std::string> out, std::vector<Attribute> attrs = {}) {
    fgx::Node n; n.op_type = op; n.name = name; n.inputs = std::move(in); n.outputs = std::move(out);
    n.attributes = std::move(attrs); return n;
}

// x[1,2,4,4] -> Conv1x1 -> Relu -> MaxPool2x2 -> Concat(self, axis -3) -> AvgPool2x2 -> Dropout -> Reshape[0,-1]
static fgx::Model tinyModel() {
    fgx::Model m;
    m.version_major = 1;
    m.meta.ir_version = 3;
    m.meta.opset_version = 7;
    m.inputs = {TD("x", {1, 2, 4, 4})};
    m.outputs = {TD("out", {1, 4})};
    m.initializers = {
        floatInit("w", {2, 2, 1, 1}, {1, 0, 0, 1}),
        floatInit("b", {2}, {0.5f, -0.5f}),
        int64Init("shape", {0, -1}),
    };
    const std::vector<Attribute> pool = {A_ints("kernel_shape", {2, 2}), A_ints("strides", {2, 2}),
                                         A_ints("pads", {0, 0, 0, 0})};
    m.nodes = {
        N("Conv", "conv", {"x", "w", "b"}, {"y"},
          {A_ints("dilations", {1, 1}), A_int("group", 1), A_ints("kernel_shape", {1, 1}),
           A_ints("pads", {0, 0, 0, 0}), A_ints("strides", {1, 1})}),
        N("Relu", "relu", {"y"}, {"r"}),
        N("MaxPool", "mp", {"r"}, {"m"}, pool),
        N("Concat", "cat", {"m", "m"}, {"cc"}, {A_int("axis", -3)}),
        N("AveragePool", "ap", {"cc"}, {"a"}, pool),
        N("Dropout", "drop", {"a"}, {"d"}, {A_float("ratio", 0.5f)}),
        N("Reshape", "flat", {"d", "shape"}, {"out"}),
    };
    m.value_info = {TD("y", {1, 2, 4, 4}), TD("r", {1, 2, 4, 4}), TD("m", {1, 2, 2, 2}),
                    TD("cc", {1, 4, 2, 2}), TD("a", {1, 4, 1, 1}), TD("d", {1, 4, 1, 1})};
    return m;
}

static std::vector<float> reference(const std::vector<float>& x) {
    const float bias[2] = {0.5f, -0.5f};
    float r[2][4][4], mp[2][2][2];
    for (int c = 0; c < 2; ++c)
        for (int h = 0; h < 4; ++h)
            for (int w = 0; w < 4; ++w) r[c][h][w] = std::max(0.0f, x[(c * 4 + h) * 4 + w] + bias[c]);
    for (int c = 0; c < 2; ++c)
        for (int h = 0; h < 2; ++h)
            for (int w = 0; w < 2; ++w)
                mp[c][h][w] = std::max({r[c][2*h][2*w], r[c][2*h][2*w+1], r[c][2*h+1][2*w], r[c][2*h+1][2*w+1]});
    std::vector<float> out(4);
    for (int c = 0; c < 4; ++c) {
        const int src = c % 2;  // concat of m with itself
        out[c] = (mp[src][0][0] + mp[src][0][1] + mp[src][1][0] + mp[src][1][1]) / 4.0f;
    }
    return out;
}

static void expectError(const std::string& label, std::function<void(fgx::Model&)> mutate,
                        const std::string& fragment) {
    fgx::Model m = tinyModel();
    mutate(m);
    try {
        (void)ImportedModel::fromFgx(m);
        ok(false, label + ": import succeeded, expected error containing '" + fragment + "'");
    } catch (const ImportError& e) {
        const std::string w = e.what();
        ok(w.find(fragment) != std::string::npos, label + (w.find(fragment) != std::string::npos
                                                           ? "" : std::string(" -- got: ") + w));
    } catch (const std::exception& e) {
        ok(false, label + ": threw non-ImportError: " + e.what());
    }
}

static void testValidModel() {
    ImportedModel im = ImportedModel::fromFgx(tinyModel());
    ok(im.graph().numNodes() == 7, "tiny model imports with 7 nodes");
    ok(im.inputNames() == std::vector<std::string>{"x"} && im.numWeights() == 2,
       "one caller input; two float32 weights become extra graph inputs (int64 shape folded)");
    ok(im.graph().numInputs() == 3, "graph inputs = 1 data + 2 weights");
    ok(im.warnings().empty(), "no warnings when every output has value_info");

    Tensor x(TensorShape({1, 2, 4, 4}), DataType::Float32, Device::CPU);
    std::vector<float> xv(32);
    for (int i = 0; i < 32; ++i) xv[i] = static_cast<float>(i) - 8.0f;
    std::memcpy(x.data(), xv.data(), 32 * 4);
    auto outs = im.run({&x});
    ok(outs.size() == 1 && outs[0]->shape().dimensions() == std::vector<size_t>({1, 4}),
       "output shape is exactly [1,4]");
    const auto ref = reference(xv);
    const float* o = static_cast<const float*>(outs[0]->data());
    bool match = true;
    for (int i = 0; i < 4; ++i) match = match && std::fabs(o[i] - ref[i]) < 1e-6f;
    ok(match, "output matches hand-written reference (Conv, Relu, MaxPool, Concat, AvgPool, Dropout, Reshape)");

    auto again = im.run({&x});
    ok(std::memcmp(again[0]->data(), outs[0]->data(), 16) == 0, "repeat run is identical");

    Tensor wrong(TensorShape({1, 2, 2, 8}), DataType::Float32, Device::CPU);
    try { (void)im.run({&wrong}); ok(false, "run rejects wrong input shape"); }
    catch (const ImportError&) { ok(true, "run rejects wrong input shape (same element count)"); }
    try { (void)im.run({}); ok(false, "run rejects missing input"); }
    catch (const ImportError&) { ok(true, "run rejects missing input"); }

    fgx::Model nv = tinyModel();
    nv.value_info.erase(nv.value_info.begin() + 1);
    ImportedModel im2 = ImportedModel::fromFgx(nv);
    ok(im2.warnings().size() == 1 && im2.warnings()[0].find("'r'") != std::string::npos,
       "missing value_info produces a warning, not an error");
}

static void testErrors() {
    expectError("unsupported op", [](fgx::Model& m) { m.nodes[1].op_type = "Sigmoid"; },
                "node 1 'relu' (Sigmoid): unsupported operator 'Sigmoid' (supported: Conv, Relu");
    expectError("opset 8 rejected", [](fgx::Model& m) { m.meta.opset_version = 8; }, "opset 8 is not supported");
    expectError("unknown attribute", [](fgx::Model& m) { m.nodes[0].attributes.push_back(A_int("foo", 1)); },
                "node 0 'conv' (Conv): unsupported attribute 'foo'");
    expectError("opset-10 MaxPool attribute", [](fgx::Model& m) {
        m.nodes[2].attributes.push_back(A_int("ceil_mode", 0)); }, "unsupported attribute 'ceil_mode'");
    expectError("asymmetric pads", [](fgx::Model& m) { m.nodes[0].attributes[3] = A_ints("pads", {0, 0, 1, 1}); },
                "asymmetric pads");
    expectError("group 2", [](fgx::Model& m) { m.nodes[0].attributes[1] = A_int("group", 2); }, "group != 1");
    expectError("dilation 2", [](fgx::Model& m) { m.nodes[0].attributes[0] = A_ints("dilations", {2, 2}); },
                "dilations [2,2]");
    expectError("kernel_shape vs weight", [](fgx::Model& m) { m.nodes[0].attributes[2] = A_ints("kernel_shape", {3, 3}); },
                "does not match weight dims");
    expectError("wrong attribute kind", [](fgx::Model& m) { m.nodes[0].attributes[1] = A_float("group", 1.0f); },
                "'group' must be INT");
    expectError("INTS wrong length", [](fgx::Model& m) { m.nodes[2].attributes[1] = A_ints("strides", {2}); },
                "'strides' must have 2 values");
    expectError("MaxPool missing kernel_shape", [](fgx::Model& m) { m.nodes[2].attributes.erase(m.nodes[2].attributes.begin()); },
                "missing required attribute 'kernel_shape'");
    expectError("Concat missing axis", [](fgx::Model& m) { m.nodes[3].attributes.clear(); }, "missing required attribute 'axis'");
    expectError("Concat axis out of range", [](fgx::Model& m) { m.nodes[3].attributes[0] = A_int("axis", 4); }, "axis 4 is out of range");
    expectError("Concat single input", [](fgx::Model& m) { m.nodes[3].inputs = {"m"}; }, "fewer than 2 inputs");
    expectError("Dropout mask output", [](fgx::Model& m) { m.nodes[5].outputs.push_back("mask"); }, "training mask output");
    expectError("count_include_pad 2", [](fgx::Model& m) { m.nodes[4].attributes.push_back(A_int("count_include_pad", 2)); },
                "count_include_pad must be 0 or 1");
    expectError("Reshape shape from a non-initializer", [](fgx::Model& m) { m.nodes[6].inputs[1] = "d"; },
                "must be a constant int64 initializer");
    expectError("Reshape shape value -2", [](fgx::Model& m) { m.initializers[2] = int64Init("shape", {0, -2}); },
                "is invalid");
    expectError("Reshape shape value 2^31", [](fgx::Model& m) { m.initializers[2] = int64Init("shape", {0, 2147483648ll}); },
                "outside the int32 range");
    expectError("Reshape element-count mismatch", [](fgx::Model& m) { m.initializers[2] = int64Init("shape", {1, 5}); },
                "element count mismatch");
    expectError("int64 initializer as tensor input", [](fgx::Model& m) {
        m.initializers.push_back(int64Init("ib", {0, 0})); m.nodes[0].inputs[2] = "ib"; }, "int64 initializer 'ib'");
    expectError("VINF shape mismatch", [](fgx::Model& m) { m.value_info[2].dims = {1, 2, 3, 3}; },
                "output 'm' shape mismatch: ForgeRT inferred [1,2,2,2], ONNX shape inference recorded [1,2,3,3]");
    expectError("OUTP shape mismatch", [](fgx::Model& m) { m.outputs[0].dims = {4}; }, "graph output 'out' shape mismatch");
    expectError("int64 graph input", [](fgx::Model& m) { m.inputs[0].dtype = fgx::kDtypeInt64; }, "only float32 inputs");
    expectError("Relu with attribute", [](fgx::Model& m) { m.nodes[1].attributes.push_back(A_int("alpha", 1)); },
                "unsupported attribute 'alpha'");
}

static void testNarrow() {
    auto v = narrowInt64ToInt32({0, -1, 2147483647ll, -2147483648ll}, "t");
    ok(v == std::vector<int32_t>({0, -1, 2147483647, -2147483647 - 1}), "narrowInt64ToInt32 keeps in-range values");
    try { (void)narrowInt64ToInt32({1, 2147483648ll}, "t"); ok(false, "2^31 rejected"); }
    catch (const ImportError& e) { ok(std::string(e.what()).find("index 1") != std::string::npos, "2^31 rejected with index"); }
    try { (void)narrowInt64ToInt32({-2147483649ll}, "t"); ok(false, "-2^31-1 rejected"); }
    catch (const ImportError&) { ok(true, "-2^31-1 rejected"); }
}

int main() {
    std::cout << "=== ONNX importer tests ===\n";
    testValidModel();
    testErrors();
    testNarrow();
    std::cout << (g_fail ? "\n" + std::to_string(g_fail) + " FAILED\n" : "\n=== All importer tests PASSED ===\n");
    return g_fail ? 1 : 0;
}
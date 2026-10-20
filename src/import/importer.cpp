/**
 * @file importer.cpp
 * @brief .fgx model -> ForgeRT Graph for the ONNX opset 7 subset used by SqueezeNet 1.1.
 */
#include "forgert/import/importer.h"

#include "forgert/operator/averagepool2d.h"
#include "forgert/operator/concat.h"
#include "forgert/operator/conv2d.h"
#include "forgert/operator/dropout.h"
#include "forgert/operator/maxpool2d.h"
#include "forgert/operator/relu.h"
#include "forgert/operator/reshape.h"

#include <cstring>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace forgert {

std::vector<int32_t> narrowInt64ToInt32(const std::vector<int64_t>& values,
                                        const std::string& what) {
    std::vector<int32_t> out;
    out.reserve(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        const int64_t v = values[i];
        if (v < std::numeric_limits<int32_t>::min() || v > std::numeric_limits<int32_t>::max())
            throw ImportError(what + ": value " + std::to_string(v) + " at index " +
                              std::to_string(i) + " is outside the int32 range");
        out.push_back(static_cast<int32_t>(v));
    }
    return out;
}

namespace {

constexpr uint32_t kSupportedOpset = 7;

const char* kSupportedOps = "Conv, Relu, Concat, MaxPool, AveragePool, Dropout, Reshape";

// Opset-7 attributes accepted per op. auto_pad never reaches here: the
// converter drops NOTSET and rejects every other value.
const std::unordered_map<std::string, std::unordered_set<std::string>>& allowedAttrs() {
    static const std::unordered_map<std::string, std::unordered_set<std::string>> m = {
        {"Conv",        {"dilations", "group", "kernel_shape", "pads", "strides"}},
        {"Relu",        {}},
        {"Concat",      {"axis"}},
        {"MaxPool",     {"kernel_shape", "pads", "strides"}},
        {"AveragePool", {"count_include_pad", "kernel_shape", "pads", "strides"}},
        {"Dropout",     {"ratio"}},
        {"Reshape",     {}},
    };
    return m;
}

template <typename T>
std::string dimsStr(const std::vector<T>& d) {
    std::string s = "[";
    for (size_t i = 0; i < d.size(); ++i) {
        if (i) s += ",";
        s += std::to_string(d[i]);
    }
    return s + "]";
}

std::vector<size_t> toSizes(const std::vector<int64_t>& dims) {
    return std::vector<size_t>(dims.begin(), dims.end());  // reader guarantees 1..2^31-1
}

std::vector<int64_t> decodeInt64(const fgx::Initializer& init) {
    std::vector<int64_t> v(init.data.size() / 8);
    std::memcpy(v.data(), init.data.data(), v.size() * 8);  // little-endian host checked by reader
    return v;
}

struct NodeCtx {
    size_t index;
    const fgx::Node& node;

    [[noreturn]] void fail(const std::string& msg) const {
        throw ImportError("node " + std::to_string(index) + " '" + node.name + "' (" +
                          node.op_type + "): " + msg);
    }
};

class Attrs {
public:
    explicit Attrs(const NodeCtx& c) : c_(c) {
        const auto& allowed = allowedAttrs().at(c.node.op_type);
        for (const auto& a : c.node.attributes) {
            if (!allowed.count(a.name))
                c_.fail("unsupported attribute '" + a.name + "' for opset " +
                        std::to_string(kSupportedOpset));
            map_[a.name] = &a;
        }
    }

    bool has(const std::string& n) const { return map_.count(n) != 0; }

    std::vector<int64_t> ints(const std::string& n, size_t len, std::vector<int64_t> dflt,
                              bool required) const {
        auto it = map_.find(n);
        if (it == map_.end()) {
            if (required) c_.fail("missing required attribute '" + n + "'");
            return dflt;
        }
        if (it->second->kind != fgx::kAttrInts) c_.fail("attribute '" + n + "' must be INTS");
        if (it->second->ints.size() != len)
            c_.fail("attribute '" + n + "' must have " + std::to_string(len) + " values, got " +
                    std::to_string(it->second->ints.size()));
        return it->second->ints;
    }

    int64_t integer(const std::string& n, int64_t dflt, bool required) const {
        auto it = map_.find(n);
        if (it == map_.end()) {
            if (required) c_.fail("missing required attribute '" + n + "'");
            return dflt;
        }
        if (it->second->kind != fgx::kAttrInt) c_.fail("attribute '" + n + "' must be INT");
        return it->second->i;
    }

    float real(const std::string& n, float dflt) const {
        auto it = map_.find(n);
        if (it == map_.end()) return dflt;
        if (it->second->kind != fgx::kAttrFloat) c_.fail("attribute '" + n + "' must be FLOAT");
        return it->second->f;
    }

private:
    const NodeCtx& c_;
    std::unordered_map<std::string, const fgx::Attribute*> map_;
};

int checkedInt(const NodeCtx& c, const std::string& what, int64_t v, int64_t lo) {
    if (v < lo || v > std::numeric_limits<int>::max())
        c.fail(what + " value " + std::to_string(v) + " is out of range");
    return static_cast<int>(v);
}

/// kernel_shape / strides / pads common to Conv and the pooling ops.
struct Window { int kH, kW, sH, sW, pH, pW; };

Window readWindow(const NodeCtx& c, const Attrs& a, bool kernel_required,
                  const std::vector<int64_t>& kernel_default) {
    const auto k = a.ints("kernel_shape", 2, kernel_default, kernel_required);
    const auto s = a.ints("strides", 2, {1, 1}, false);
    const auto p = a.ints("pads", 4, {0, 0, 0, 0}, false);
    Window w{};
    w.kH = checkedInt(c, "kernel_shape", k[0], 1);
    w.kW = checkedInt(c, "kernel_shape", k[1], 1);
    w.sH = checkedInt(c, "strides", s[0], 1);
    w.sW = checkedInt(c, "strides", s[1], 1);
    // ONNX pads: [h_begin, w_begin, h_end, w_end]; ForgeRT supports symmetric padding.
    if (p[0] != p[2] || p[1] != p[3])
        c.fail("asymmetric pads " + dimsStr(p) + " are not supported (begin must equal end)");
    w.pH = checkedInt(c, "pads", p[0], 0);
    w.pW = checkedInt(c, "pads", p[1], 0);
    return w;
}

void expectArity(const NodeCtx& c, size_t min_in, size_t max_in, size_t outs) {
    const size_t n_in = c.node.inputs.size(), n_out = c.node.outputs.size();
    if (n_in < min_in || n_in > max_in)
        c.fail("expected " + (min_in == max_in ? std::to_string(min_in)
                                                : std::to_string(min_in) + ".." + std::to_string(max_in)) +
               " inputs, got " + std::to_string(n_in));
    if (n_out != outs)
        c.fail("expected " + std::to_string(outs) + " output(s), got " + std::to_string(n_out) +
               (c.node.op_type == "Dropout" ? " (the training mask output is not supported)" : ""));
}

} // namespace

ImportedModel ImportedModel::fromFgx(const fgx::Model& model) {
    if (model.meta.opset_version != kSupportedOpset)
        throw ImportError("ONNX opset " + std::to_string(model.meta.opset_version) +
                          " is not supported (this importer accepts opset 7 only)");

    ImportedModel im;
    Graph& g = im.graph_;
    std::unordered_map<std::string, size_t> ids;                     // value name -> graph value id
    std::unordered_map<std::string, std::vector<int64_t>> int64_inits;

    // 1. Caller-supplied inputs.
    for (const auto& in : model.inputs) {
        if (in.dtype != fgx::kDtypeFloat32)
            throw ImportError("graph input '" + in.name + "' has dtype " + std::to_string(in.dtype) +
                              "; only float32 inputs are supported");
        TensorShape shape(toSizes(in.dims));
        ids[in.name] = g.addInput(shape, DataType::Float32, in.name);
        im.input_names_.push_back(in.name);
        im.input_shapes_.push_back(shape);
    }

    // 2. Initializers: float32 -> extra graph inputs (weights); int64 -> held for Reshape.
    for (const auto& init : model.initializers) {
        if (init.desc.dtype == fgx::kDtypeInt64) {
            int64_inits[init.desc.name] = decodeInt64(init);
            continue;
        }
        TensorShape shape(toSizes(init.desc.dims));
        auto t = std::make_unique<Tensor>(shape, DataType::Float32, Device::CPU);
        if (t->numBytes() != init.data.size())
            throw ImportError("initializer '" + init.desc.name + "': byte size mismatch");
        std::memcpy(t->data(), init.data.data(), init.data.size());
        ids[init.desc.name] = g.addInput(shape, DataType::Float32, init.desc.name);
        im.weights_.push_back(std::move(t));
    }

    std::unordered_map<std::string, const fgx::TensorDesc*> vinf, outp;
    for (const auto& v : model.value_info) vinf[v.name] = &v;
    for (const auto& o : model.outputs) outp[o.name] = &o;

    // 3. Nodes, in file (topological) order.
    for (size_t k = 0; k < model.nodes.size(); ++k) {
        const fgx::Node& node = model.nodes[k];
        const NodeCtx c{k, node};
        if (!allowedAttrs().count(node.op_type))
            c.fail("unsupported operator '" + node.op_type + "' (supported: " + kSupportedOps + ")");
        const Attrs a(c);
        const std::string& op = node.op_type;

        // Resolve tensor inputs. Only Reshape's shape input may name an int64 initializer.
        std::vector<size_t> in_ids;
        for (size_t i = 0; i < node.inputs.size(); ++i) {
            const std::string& name = node.inputs[i];
            if (op == "Reshape" && i == 1) continue;
            if (int64_inits.count(name))
                c.fail("int64 initializer '" + name + "' is used as a tensor input; "
                       "only a Reshape shape input may be int64");
            auto it = ids.find(name);
            if (it == ids.end()) c.fail("input '" + name + "' is not defined");
            in_ids.push_back(it->second);
        }

        std::unique_ptr<Operator> opn;
        if (op == "Conv") {
            expectArity(c, 2, 3, 1);
            const TensorShape ws = g.getValueShape(in_ids[1]);
            if (ws.ndim() != 4) c.fail("weight must be 4-D, got " + std::to_string(ws.ndim()) + "-D");
            if (a.integer("group", 1, false) != 1) c.fail("group != 1 is not supported");
            const auto dil = a.ints("dilations", 2, {1, 1}, false);
            if (dil[0] != 1 || dil[1] != 1) c.fail("dilations " + dimsStr(dil) + " are not supported");
            const Window w = readWindow(c, a, false,
                                        {static_cast<int64_t>(ws.dim(2)), static_cast<int64_t>(ws.dim(3))});
            if (static_cast<size_t>(w.kH) != ws.dim(2) || static_cast<size_t>(w.kW) != ws.dim(3))
                c.fail("kernel_shape [" + std::to_string(w.kH) + "," + std::to_string(w.kW) +
                       "] does not match weight dims " + dimsStr(ws.dimensions()));
            Conv2DParams p;
            p.kH = w.kH; p.kW = w.kW; p.stride_h = w.sH; p.stride_w = w.sW; p.pad_h = w.pH; p.pad_w = w.pW;
            opn = std::make_unique<Conv2DOp>(p);
        } else if (op == "Relu") {
            expectArity(c, 1, 1, 1);
            opn = std::make_unique<ReLUOp>();
        } else if (op == "Concat") {
            if (node.inputs.size() < 2)
                c.fail("Concat with fewer than 2 inputs is not supported");
            expectArity(c, 2, node.inputs.size(), 1);
            const int64_t rank = static_cast<int64_t>(g.getValueShape(in_ids[0]).ndim());
            int64_t axis = a.integer("axis", 0, true);
            if (axis < -rank || axis >= rank)
                c.fail("axis " + std::to_string(axis) + " is out of range for rank " + std::to_string(rank));
            if (axis < 0) axis += rank;
            opn = std::make_unique<ConcatOp>(static_cast<int>(axis));
        } else if (op == "MaxPool") {
            expectArity(c, 1, 1, 1);
            const Window w = readWindow(c, a, true, {});
            MaxPool2DParams p;
            p.kH = w.kH; p.kW = w.kW; p.stride_h = w.sH; p.stride_w = w.sW; p.pad_h = w.pH; p.pad_w = w.pW;
            opn = std::make_unique<MaxPool2DOp>(p);
        } else if (op == "AveragePool") {
            expectArity(c, 1, 1, 1);
            const Window w = readWindow(c, a, true, {});
            const int64_t cip = a.integer("count_include_pad", 0, false);
            if (cip != 0 && cip != 1) c.fail("count_include_pad must be 0 or 1");
            AvgPool2DParams p;
            p.kH = w.kH; p.kW = w.kW; p.stride_h = w.sH; p.stride_w = w.sW; p.pad_h = w.pH; p.pad_w = w.pW;
            p.count_include_pad = (cip == 1);
            try { opn = std::make_unique<AveragePool2DOp>(p); }
            catch (const std::invalid_argument& e) { c.fail(e.what()); }
        } else if (op == "Dropout") {
            expectArity(c, 1, 1, 1);
            DropoutParams p;
            p.ratio = a.real("ratio", 0.5f);
            opn = std::make_unique<DropoutOp>(p);
        } else {  // Reshape
            expectArity(c, 2, 2, 1);
            const std::string& sname = node.inputs[1];
            auto it = int64_inits.find(sname);
            if (it == int64_inits.end())
                c.fail("shape input '" + sname + "' must be a constant int64 initializer");
            const auto narrowed = narrowInt64ToInt32(it->second, "Reshape shape '" + sname + "'");
            std::vector<int64_t> target(narrowed.begin(), narrowed.end());
            try { opn = std::make_unique<ReshapeOp>(std::move(target)); }
            catch (const std::invalid_argument& e) { c.fail(e.what()); }
        }

        std::vector<size_t> out_ids;
        try {
            out_ids = g.addNode(std::move(opn), in_ids, node.name);
        } catch (const std::exception& e) {
            c.fail(e.what());
        }
        if (out_ids.size() != node.outputs.size())
            c.fail("operator produced " + std::to_string(out_ids.size()) + " outputs, model declares " +
                   std::to_string(node.outputs.size()));

        for (size_t i = 0; i < out_ids.size(); ++i) {
            const std::string& oname = node.outputs[i];
            ids[oname] = out_ids[i];
            const std::vector<size_t> got = g.getValueShape(out_ids[i]).dimensions();
            auto vi = vinf.find(oname);
            if (vi != vinf.end()) {
                if (vi->second->dtype != fgx::kDtypeFloat32)
                    c.fail("output '" + oname + "' is recorded as dtype " + std::to_string(vi->second->dtype) +
                           "; ForgeRT produces float32");
                if (got != toSizes(vi->second->dims))
                    c.fail("output '" + oname + "' shape mismatch: ForgeRT inferred " + dimsStr(got) +
                           ", ONNX shape inference recorded " + dimsStr(vi->second->dims));
            } else if (!outp.count(oname)) {
                im.warnings_.push_back("node " + std::to_string(k) + " '" + node.name + "' (" + op +
                                       "): output '" + oname + "' has no value_info entry; shape not cross-checked");
            }
        }
    }

    // 4. Graph outputs: exact shape and dtype check.
    for (const auto& o : model.outputs) {
        if (o.dtype != fgx::kDtypeFloat32)
            throw ImportError("graph output '" + o.name + "' has dtype " + std::to_string(o.dtype) +
                              "; only float32 outputs are supported");
        auto it = ids.find(o.name);
        if (it == ids.end()) throw ImportError("graph output '" + o.name + "' is never produced");
        const std::vector<size_t> got = g.getValueShape(it->second).dimensions();
        if (got != toSizes(o.dims))
            throw ImportError("graph output '" + o.name + "' shape mismatch: ForgeRT inferred " +
                              dimsStr(got) + ", model declares " + dimsStr(o.dims));
        g.markOutput(it->second);
        im.output_names_.push_back(o.name);
    }

    try {
        g.validate();
    } catch (const std::exception& e) {
        throw ImportError(std::string("graph validation failed: ") + e.what());
    }
    return im;
}

ImportedModel ImportedModel::load(const std::string& path) {
    return fromFgx(fgx::readFile(path));
}

std::vector<std::unique_ptr<Tensor>> ImportedModel::run(const std::vector<const Tensor*>& inputs) {
    if (inputs.size() != input_names_.size())
        throw ImportError("run: expected " + std::to_string(input_names_.size()) + " input(s), got " +
                          std::to_string(inputs.size()));
    std::vector<const Tensor*> all;
    all.reserve(inputs.size() + weights_.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        const Tensor* t = inputs[i];
        if (!t) throw ImportError("run: input '" + input_names_[i] + "' is null");
        if (t->dtype() != DataType::Float32 || t->device() != Device::CPU)
            throw ImportError("run: input '" + input_names_[i] + "' must be a float32 CPU tensor");
        if (t->shape().dimensions() != input_shapes_[i].dimensions())
            throw ImportError("run: input '" + input_names_[i] + "' has shape " +
                              dimsStr(t->shape().dimensions()) + ", expected " +
                              dimsStr(input_shapes_[i].dimensions()));
        all.push_back(t);
    }
    for (const auto& w : weights_) all.push_back(w.get());
    return graph_.execute(all, Backend::CPU);
}

} // namespace forgert
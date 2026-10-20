/*
 * test_fgx_reader.cpp  --  strict .fgx v1 reader: valid parse + malformed files
 *
 * Every malformed case starts from a valid in-memory file and changes one
 * thing. The payload CRC is recomputed except in the CRC case, so each case
 * reaches the check it targets and proves that bounds checks still apply to
 * files with a valid CRC. Case numbers follow
 * .kiro/specs/onnx-importer-pilot/fgx-review.md.
 */
#include "forgert/import/fgx.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

using namespace forgert::fgx;
using Bytes = std::vector<uint8_t>;

static int g_fail = 0;
static void ok(bool cond, const std::string& msg) {
    std::cout << (cond ? "PASS: " : "FAIL: ") << msg << "\n";
    if (!cond) ++g_fail;
}

// ---------------------------------------------------------------- writer ---
struct W {
    Bytes b;
    W& u8(uint8_t v)  { b.push_back(v); return *this; }
    W& u16(uint16_t v) { for (int i = 0; i < 2; ++i) b.push_back(uint8_t(v >> (8 * i))); return *this; }
    W& u32(uint32_t v) { for (int i = 0; i < 4; ++i) b.push_back(uint8_t(v >> (8 * i))); return *this; }
    W& u64(uint64_t v) { for (int i = 0; i < 8; ++i) b.push_back(uint8_t(v >> (8 * i))); return *this; }
    W& i64(int64_t v)  { return u64(uint64_t(v)); }
    W& f32(float f)    { uint32_t x; std::memcpy(&x, &f, 4); return u32(x); }
    W& str(const std::string& s) { u32(uint32_t(s.size())); b.insert(b.end(), s.begin(), s.end()); return *this; }
    W& raw(const Bytes& v) { b.insert(b.end(), v.begin(), v.end()); return *this; }
};

struct TD { std::string name; uint32_t dtype = kDtypeFloat32; std::vector<int64_t> dims; };
struct InitS { TD d; Bytes data; int64_t len_override = -1; };
struct AttrS { std::string name; uint8_t kind; int64_t i = 0; float f = 0; std::vector<int64_t> ints; int64_t count_override = -1; };
struct NodeS { std::string op, name; std::vector<std::string> in, out; std::vector<AttrS> attrs; };
struct Spec {
    uint32_t ir = 3, opset = 7;
    std::string producer = "unit-test";
    std::vector<TD> inputs, outputs, vinf;
    std::vector<InitS> inits;
    std::vector<NodeS> nodes;
    int64_t inpt_count_override = -1, init_count_override = -1;
};

static Bytes floatBytes(std::vector<float> v) { W w; for (float f : v) w.f32(f); return w.b; }
static Bytes int64Bytes(std::vector<int64_t> v) { W w; for (auto x : v) w.i64(x); return w.b; }

static Spec validSpec() {
    Spec s;
    s.inputs  = {{"x", kDtypeFloat32, {1, 2, 4, 4}}};
    s.outputs = {{"z", kDtypeFloat32, {1, 48}}};
    s.inits = {
        {{"w", kDtypeFloat32, {3, 2, 1, 1}}, floatBytes({1, 2, 3, 4, 5, 6})},
        {{"b", kDtypeFloat32, {3}}, floatBytes({0.5f, -0.5f, 0.25f})},
        {{"shape", kDtypeInt64, {2}}, int64Bytes({0, -1})},
    };
    s.nodes = {
        {"Conv", "conv", {"x", "w", "b"}, {"y"},
         {{"kernel_shape", kAttrInts, 0, 0, {1, 1}}, {"strides", kAttrInts, 0, 0, {1, 1}},
          {"pads", kAttrInts, 0, 0, {0, 0, 0, 0}}, {"group", kAttrInt, 1}}},
        {"Dropout", "drop", {"y"}, {"d"}, {{"ratio", kAttrFloat, 0, 0.5f}}},
        {"Reshape", "", {"d", "shape"}, {"z"}, {}},
    };
    s.vinf = {{"y", kDtypeFloat32, {1, 3, 4, 4}}, {"d", kDtypeFloat32, {1, 3, 4, 4}}};
    return s;
}

static void tdesc(W& w, const TD& t) {
    w.str(t.name).u32(t.dtype).u32(uint32_t(t.dims.size()));
    for (auto d : t.dims) w.i64(d);
}

struct Sec { std::string tag; Bytes content; int64_t len_override = -1; };

static std::vector<Sec> sections(const Spec& s) {
    std::vector<Sec> out;
    W meta; meta.u32(s.ir).u32(s.opset).str(s.producer);
    for (int i = 0; i < 32; ++i) meta.u8(uint8_t(i));
    out.push_back({"META", meta.b});

    auto descList = [](const std::vector<TD>& l, int64_t count_override) {
        W w; w.u32(count_override >= 0 ? uint32_t(count_override) : uint32_t(l.size()));
        for (const auto& t : l) tdesc(w, t);
        return w.b;
    };
    out.push_back({"INPT", descList(s.inputs, s.inpt_count_override)});
    out.push_back({"OUTP", descList(s.outputs, -1)});

    W init; init.u32(s.init_count_override >= 0 ? uint32_t(s.init_count_override) : uint32_t(s.inits.size()));
    for (const auto& i : s.inits) {
        tdesc(init, i.d);
        init.u64(i.len_override >= 0 ? uint64_t(i.len_override) : i.data.size()).raw(i.data);
    }
    out.push_back({"INIT", init.b});

    W node; node.u32(uint32_t(s.nodes.size()));
    for (const auto& n : s.nodes) {
        node.str(n.op).str(n.name).u32(uint32_t(n.in.size()));
        for (const auto& x : n.in) node.str(x);
        node.u32(uint32_t(n.out.size()));
        for (const auto& x : n.out) node.str(x);
        node.u32(uint32_t(n.attrs.size()));
        for (const auto& a : n.attrs) {
            node.str(a.name).u8(a.kind);
            if (a.kind == kAttrInt) node.i64(a.i);
            else if (a.kind == kAttrFloat) node.f32(a.f);
            else {
                node.u32(a.count_override >= 0 ? uint32_t(a.count_override) : uint32_t(a.ints.size()));
                for (auto v : a.ints) node.i64(v);
            }
        }
    }
    out.push_back({"NODE", node.b});
    out.push_back({"VINF", descList(s.vinf, -1)});
    return out;
}

struct Hdr {
    std::string magic = "FGXM";
    uint16_t major = 1, minor = 0;
    uint32_t flags = 0;
    int64_t size_delta = 0;
    bool size_override = false;
    uint64_t size_value = 0;
    bool fix_crc = true;
    Bytes trailing;
};

static Bytes assemble(const std::vector<Sec>& secs, const Hdr& h = {}) {
    W payload;
    for (const auto& s : secs) {
        payload.b.insert(payload.b.end(), s.tag.begin(), s.tag.end());
        payload.u64(s.len_override >= 0 ? uint64_t(s.len_override) : s.content.size()).raw(s.content);
    }
    payload.raw(h.trailing);
    W out;
    out.b.insert(out.b.end(), h.magic.begin(), h.magic.end());
    const uint64_t total = kHeaderSize + payload.b.size();
    out.u16(h.major).u16(h.minor).u32(h.flags)
       .u64(h.size_override ? h.size_value : uint64_t(int64_t(total) + h.size_delta))
       .u32(h.fix_crc ? crc32(payload.b.data(), payload.b.size()) : 0u);
    out.raw(payload.b);
    return out.b;
}

static Bytes build(const Spec& s) { return assemble(sections(s)); }

/// Recompute the payload CRC after a byte-level edit.
static Bytes refixCrc(Bytes b) {
    const uint32_t c = crc32(b.data() + kHeaderSize, b.size() - kHeaderSize);
    for (int i = 0; i < 4; ++i) b[20 + i] = uint8_t(c >> (8 * i));
    return b;
}

static void expectReject(const std::string& label, const Bytes& bytes, ErrorCategory want) {
    try {
        (void)parse(bytes);
        ok(false, label + ": accepted, expected " + toString(want));
    } catch (const FormatError& e) {
        ok(e.category() == want,
           label + " -> " + toString(e.category()) +
           (e.category() == want ? "" : std::string(" (expected ") + toString(want) + "; " + e.what() + ")"));
    } catch (const std::exception& e) {
        ok(false, label + ": threw non-FormatError: " + e.what());
    }
}

// ---------------------------------------------------------------- tests ---
static void testValid() {
    const Bytes bytes = build(validSpec());
    Model m;
    try { m = parse(bytes); ok(true, "valid file parses"); }
    catch (const std::exception& e) { ok(false, std::string("valid file parses: ") + e.what()); return; }

    ok(m.version_major == 1 && m.version_minor == 0, "version 1.0");
    ok(m.meta.ir_version == 3 && m.meta.opset_version == 7 && m.meta.producer == "unit-test", "META fields");
    ok(m.meta.source_sha256[0] == 0 && m.meta.source_sha256[31] == 31, "META sha256 bytes");
    ok(m.inputs.size() == 1 && m.inputs[0].dims == std::vector<int64_t>({1, 2, 4, 4}), "INPT dims");
    ok(m.initializers.size() == 3 && m.initializers[0].data == floatBytes({1, 2, 3, 4, 5, 6}),
       "float32 initializer bytes identical");
    ok(m.initializers[2].desc.dtype == kDtypeInt64 && m.initializers[2].data == int64Bytes({0, -1}),
       "int64 initializer kept as dtype 7 with identical bytes");
    ok(m.nodes.size() == 3 && m.nodes[0].attributes[3].kind == kAttrInt && m.nodes[0].attributes[3].i == 1,
       "INT attribute");
    ok(m.nodes[0].attributes[2].ints == std::vector<int64_t>({0, 0, 0, 0}), "INTS attribute");
    ok(m.nodes[1].attributes[0].kind == kAttrFloat && m.nodes[1].attributes[0].f == 0.5f, "FLOAT attribute");
    ok(m.nodes[2].name.empty(), "empty node_name allowed");
    ok(m.value_info.size() == 2, "VINF entries");

    const std::string d = dump(m);
    ok(d.rfind("FGX 1.0\nMETA ir=3 opset=7 producer=unit-test sha256=000102", 0) == 0, "dump header lines");
    ok(d.find("NODE 1 op=Dropout name=drop in=y out=d attrs=ratio:float=0x3f000000\n") != std::string::npos,
       "dump encodes FLOAT as bit pattern");
    ok(d.find("NODE 2 op=Reshape name= in=d,shape out=z attrs=\n") != std::string::npos, "dump empty name/attrs");
    ok(d.find("INIT shape dtype=7 dims=2 bytes=16 crc32=") != std::string::npos, "dump INIT line");

    // File path round trip
    const auto path = (std::filesystem::temp_directory_path() / "forgert_test_valid.fgx").string();
    { std::ofstream f(path, std::ios::binary); f.write(reinterpret_cast<const char*>(bytes.data()), bytes.size()); }
    try { ok(dump(readFile(path)) == d, "readFile matches parse"); }
    catch (const std::exception& e) { ok(false, std::string("readFile: ") + e.what()); }
    std::filesystem::remove(path);
    try { (void)readFile(path); ok(false, "missing file rejected"); }
    catch (const FormatError& e) { ok(e.category() == ErrorCategory::Io, "missing file -> io"); }

    const char* v = "123456789";
    ok(crc32(reinterpret_cast<const uint8_t*>(v), 9) == 0xCBF43926u, "CRC-32 check value");
}

static void testMalformed() {
    using EC = ErrorCategory;
    const Bytes good = build(validSpec());
    auto spec = [](std::function<void(Spec&)> f) { Spec s = validSpec(); f(s); return build(s); };
    auto secs = [](std::function<void(std::vector<Sec>&)> f) { auto v = sections(validSpec()); f(v); return assemble(v); };
    auto hdr  = [](std::function<void(Hdr&)> f) { Hdr h; f(h); return assemble(sections(validSpec()), h); };

    expectReject("1  empty file", {}, EC::Truncated);
    expectReject("2  23-byte file", Bytes(good.begin(), good.begin() + 23), EC::Truncated);
    expectReject("3  magic FGXN", hdr([](Hdr& h) { h.magic = "FGXN"; }), EC::BadMagic);
    expectReject("4  magic MXGF", hdr([](Hdr& h) { h.magic = "MXGF"; }), EC::BadMagic);
    expectReject("5  version_major 2", hdr([](Hdr& h) { h.major = 2; }), EC::BadVersion);
    expectReject("6  version_major 0", hdr([](Hdr& h) { h.major = 0; }), EC::BadVersion);
    expectReject("7  version_minor 1", hdr([](Hdr& h) { h.minor = 1; }), EC::BadVersion);
    expectReject("8  flags 1", hdr([](Hdr& h) { h.flags = 1; }), EC::BadFlags);
    expectReject("9  total_file_size +1", hdr([](Hdr& h) { h.size_delta = 1; }), EC::SizeMismatch);
    expectReject("10 total_file_size -1", hdr([](Hdr& h) { h.size_delta = -1; }), EC::SizeMismatch);
    expectReject("11 truncated by 1 byte", Bytes(good.begin(), good.end() - 1), EC::SizeMismatch);
    expectReject("12 trailing byte", hdr([](Hdr& h) { h.trailing = {0}; }), EC::TrailingBytes);
    // 13 (file > 1 GiB) needs a >1 GiB buffer; the check is a single size comparison, not exercised here.
    expectReject("14 total_file_size 2^64-1", hdr([](Hdr& h) { h.size_override = true; h.size_value = ~0ull; }),
                 EC::SizeMismatch);
    { Bytes b = good; b[40] ^= 0xFF; expectReject("15 payload byte flipped, CRC stale", b, EC::CrcMismatch); }
    expectReject("16 INPT first", secs([](auto& v) { std::swap(v[0], v[1]); }), EC::SectionOrder);
    expectReject("17 INIT/NODE swapped", secs([](auto& v) { std::swap(v[3], v[4]); }), EC::SectionOrder);
    expectReject("18 META twice", secs([](auto& v) { v.insert(v.begin() + 1, v[0]); }), EC::SectionOrder);
    expectReject("19 unknown tag XXXX", secs([](auto& v) { v[5].tag = "XXXX"; }), EC::SectionOrder);
    expectReject("20 VINF missing", secs([](auto& v) { v.pop_back(); }), EC::SectionOrder);
    expectReject("21 section length > remaining", secs([](auto& v) { v[3].len_override = 1ll << 40; }),
                 EC::SectionLength);
    expectReject("22 section has 4 unconsumed bytes",
                 secs([](auto& v) { v[0].content.insert(v[0].content.end(), 4, 0); }), EC::SectionLength);
    expectReject("23 section length < content",
                 secs([](auto& v) { v[0].len_override = int64_t(v[0].content.size()) - 1; }), EC::SectionLength);
    expectReject("24 string length 4097", spec([](Spec& s) { s.producer = std::string(4097, 'a'); }),
                 EC::StringTooLong);
    expectReject("25 string length 0xFFFFFFFF", secs([](auto& v) {
        W w; w.u32(3).u32(7).u32(0xFFFFFFFFu); v[0].content = w.b; }), EC::StringTooLong);
    expectReject("26 invalid UTF-8 (0xFF)", spec([](Spec& s) { s.inputs[0].name = "x\xff"; }), EC::InvalidUtf8);
    expectReject("26b overlong UTF-8 (C0 80)", spec([](Spec& s) { s.producer = "\xc0\x80"; }), EC::InvalidUtf8);
    expectReject("27 empty INIT name", spec([](Spec& s) { s.inits[0].d.name = ""; }), EC::EmptyName);
    expectReject("28 INPT count 1,000,001", spec([](Spec& s) { s.inpt_count_override = 1000001; }), EC::CountLimit);
    expectReject("29 INIT count 1,000,000, no room", spec([](Spec& s) { s.init_count_override = 1000000; }),
                 EC::SectionLength);
    expectReject("30 rank 9", spec([](Spec& s) { s.inputs[0].dims = std::vector<int64_t>(9, 1); }), EC::RankLimit);
    // A lone rank-0 entry is shorter than the minimum entry size and is caught by the count
    // pre-check; a second, larger entry gives the section room so the rank rule itself is hit.
    expectReject("30b rank 0", spec([](Spec& s) { s.inputs[0].dims = {}; s.inputs.push_back({"extra", kDtypeFloat32, {1, 1, 1, 1}}); }), EC::RankLimit);
    expectReject("30c lone rank-0 entry", spec([](Spec& s) { s.inputs[0].dims = {}; }), EC::SectionLength);
    expectReject("31 dim 0", spec([](Spec& s) { s.inputs[0].dims[2] = 0; }), EC::DimRange);
    expectReject("32 dim -1", spec([](Spec& s) { s.inputs[0].dims[2] = -1; }), EC::DimRange);
    expectReject("33 dim 2^31", spec([](Spec& s) { s.inputs[0].dims[2] = 1ll << 31; }), EC::DimRange);
    expectReject("34 element count overflows u64",
                 spec([](Spec& s) { s.inputs[0].dims = std::vector<int64_t>(8, kMaxDim); }), EC::Overflow);
    expectReject("35 INIT byte_length prod*4-1", spec([](Spec& s) { s.inits[0].len_override = 23; }), EC::ByteLength);
    expectReject("36 INIT data shorter than byte_length", spec([](Spec& s) {
        s.inits[2].data.resize(8); s.inits[2].len_override = 16; }), EC::SectionLength);
    expectReject("37 INIT dtype 10", spec([](Spec& s) { s.inits[0].d.dtype = 10; }), EC::UnsupportedDtype);
    expectReject("38 int64 INIT with 4-byte elements", spec([](Spec& s) {
        s.inits[2].data.resize(8); s.inits[2].len_override = 8; }), EC::ByteLength);
    expectReject("39 OUTP dtype 0", spec([](Spec& s) { s.outputs[0].dtype = 0; }), EC::UnsupportedDtype);
    expectReject("40 attribute kind 4", spec([](Spec& s) { s.nodes[0].attrs[3].kind = 4; }),
                 EC::UnsupportedAttrKind);
    expectReject("41 attribute kind 7 (ONNX INTS code)", spec([](Spec& s) { s.nodes[0].attrs[3].kind = 7; }),
                 EC::UnsupportedAttrKind);
    expectReject("42 INTS count 1,000,001", spec([](Spec& s) { s.nodes[0].attrs[0].count_override = 1000001; }),
                 EC::CountLimit);
    expectReject("43 duplicate attribute name", spec([](Spec& s) { s.nodes[0].attrs[1].name = "kernel_shape"; }),
                 EC::DuplicateName);
    expectReject("44 duplicate INIT name", spec([](Spec& s) { s.inits[1].d.name = "w"; }), EC::DuplicateName);
    expectReject("45 INPT name equals INIT name", spec([](Spec& s) {
        s.inputs.push_back({"w", kDtypeFloat32, {3, 2, 1, 1}}); }), EC::DuplicateName);
    expectReject("46 two producers of one value", spec([](Spec& s) { s.nodes[1].out = {"y"}; }), EC::DuplicateName);
    expectReject("47 input produced by a later node", spec([](Spec& s) { std::swap(s.nodes[0], s.nodes[1]); }),
                 EC::UndefinedInput);
    expectReject("48 undefined input", spec([](Spec& s) { s.nodes[2].in[0] = "nope"; }), EC::UndefinedInput);
    expectReject("49 empty node output name", spec([](Spec& s) { s.nodes[0].out[0] = ""; }), EC::EmptyName);
    // 50 (big-endian host) cannot run on x86; the check is the first statement of parse().

    // Additional v1 rules
    expectReject("51 empty op_type", spec([](Spec& s) { s.nodes[0].op = ""; }), EC::EmptyName);
    expectReject("52 node with no outputs", spec([](Spec& s) { s.nodes[2].out = {}; s.outputs[0].name = "d"; }),
                 EC::CountLimit);
    expectReject("53 graph output never produced", spec([](Spec& s) { s.outputs[0].name = "q"; }),
                 EC::UndefinedInput);
    expectReject("54 duplicate VINF name", spec([](Spec& s) { s.vinf[1].name = "y"; }), EC::DuplicateName);
    expectReject("55 empty node input (omitted optional)", spec([](Spec& s) { s.nodes[0].in.push_back(""); }),
                 EC::EmptyName);
    expectReject("56 node output shadows an input", spec([](Spec& s) { s.nodes[1].out = {"x"}; }),
                 EC::DuplicateName);
}

int main() {
    std::cout << "=== .fgx reader tests ===\n";
    testValid();
    testMalformed();
    std::cout << (g_fail ? "\n" + std::to_string(g_fail) + " FAILED\n" : "\n=== All .fgx reader tests PASSED ===\n");
    return g_fail ? 1 : 0;
}
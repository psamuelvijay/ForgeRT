/**
 * @file fgx_reader.cpp
 * @brief Strict, dependency-free reader for .fgx v1 (docs/FGX_FORMAT.md).
 *
 * Every length is checked against the bytes remaining before anything is
 * allocated, and every check applies regardless of the CRC result.
 */
#include "forgert/import/fgx.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <unordered_set>

namespace forgert {
namespace fgx {

const char* toString(ErrorCategory c) {
    switch (c) {
        case ErrorCategory::Io:                  return "io";
        case ErrorCategory::HostEndianness:      return "host-endianness";
        case ErrorCategory::Truncated:           return "truncated";
        case ErrorCategory::Oversized:           return "oversized";
        case ErrorCategory::BadMagic:            return "bad-magic";
        case ErrorCategory::BadVersion:          return "bad-version";
        case ErrorCategory::BadFlags:            return "bad-flags";
        case ErrorCategory::SizeMismatch:        return "size-mismatch";
        case ErrorCategory::CrcMismatch:         return "crc-mismatch";
        case ErrorCategory::SectionOrder:        return "section-order";
        case ErrorCategory::SectionLength:       return "section-length";
        case ErrorCategory::TrailingBytes:       return "trailing-bytes";
        case ErrorCategory::StringTooLong:       return "string-too-long";
        case ErrorCategory::InvalidUtf8:         return "invalid-utf8";
        case ErrorCategory::EmptyName:           return "empty-name";
        case ErrorCategory::CountLimit:          return "count-limit";
        case ErrorCategory::RankLimit:           return "rank-limit";
        case ErrorCategory::DimRange:            return "dim-range";
        case ErrorCategory::Overflow:            return "overflow";
        case ErrorCategory::ByteLength:          return "byte-length";
        case ErrorCategory::UnsupportedDtype:    return "unsupported-dtype";
        case ErrorCategory::UnsupportedAttrKind: return "unsupported-attr-kind";
        case ErrorCategory::DuplicateName:       return "duplicate-name";
        case ErrorCategory::UndefinedInput:      return "undefined-input";
    }
    return "unknown";
}

uint32_t crc32(const uint8_t* data, size_t size, uint32_t crc) {
    static const auto table = [] {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            t[i] = c;
        }
        return t;
    }();
    crc = ~crc;
    for (size_t i = 0; i < size; ++i) crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
    return ~crc;
}

namespace {

constexpr const char* kSectionTags[6] = {"META", "INPT", "OUTP", "INIT", "NODE", "VINF"};

// Conservative minimum encoded sizes, used to reject counts that cannot fit.
constexpr size_t kMinTensorDesc = 20;  // name len + dtype + rank + one dim
constexpr size_t kMinInit       = 28;  // tensor desc + byte_length
constexpr size_t kMinNode       = 25;  // two strings, three counts, one output
constexpr size_t kMinString     = 4;
constexpr size_t kMinAttr       = 9;   // name len + kind + smallest payload

bool hostIsLittleEndian() {
    const uint16_t v = 1;
    uint8_t first = 0;
    std::memcpy(&first, &v, 1);
    return first == 1;
}

uint16_t le16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }
uint32_t le32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}
uint64_t le64(const uint8_t* p) {
    return static_cast<uint64_t>(le32(p)) | (static_cast<uint64_t>(le32(p + 4)) << 32);
}

std::string printable(const uint8_t* p, size_t n) {
    std::string out;
    for (size_t i = 0; i < n; ++i) {
        if (p[i] >= 0x21 && p[i] <= 0x7E && p[i] != '\\') {
            out += static_cast<char>(p[i]);
        } else {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\x%02x", p[i]);
            out += buf;
        }
    }
    return out;
}

bool validUtf8(const std::string& s) {
    const auto* p = reinterpret_cast<const uint8_t*>(s.data());
    size_t i = 0, n = s.size();
    while (i < n) {
        const uint8_t c = p[i];
        size_t len;
        uint32_t cp;
        if (c < 0x80)              { ++i; continue; }
        else if ((c & 0xE0) == 0xC0) { len = 2; cp = c & 0x1Fu; }
        else if ((c & 0xF0) == 0xE0) { len = 3; cp = c & 0x0Fu; }
        else if ((c & 0xF8) == 0xF0) { len = 4; cp = c & 0x07u; }
        else return false;
        if (i + len > n) return false;
        for (size_t k = 1; k < len; ++k) {
            if ((p[i + k] & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (p[i + k] & 0x3Fu);
        }
        if ((len == 2 && cp < 0x80) || (len == 3 && cp < 0x800) || (len == 4 && cp < 0x10000))
            return false;  // overlong
        if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return false;
        i += len;
    }
    return true;
}

/// Bounded reader over one section's content.
class Cursor {
public:
    Cursor(const uint8_t* base, size_t begin, size_t end, const char* section)
        : base_(base), pos_(begin), end_(end), section_(section) {}

    size_t remaining() const { return end_ - pos_; }

    [[noreturn]] void fail(ErrorCategory c, const std::string& what) const {
        throw FormatError(c, std::string("section ") + section_ + ", offset " +
                                 std::to_string(pos_) + ": " + what);
    }

    void need(uint64_t n, const std::string& what) const {
        if (n > remaining())
            fail(ErrorCategory::SectionLength,
                 what + " needs " + std::to_string(n) + " bytes but only " +
                 std::to_string(remaining()) + " remain in the section");
    }

    uint8_t  u8 (const char* what) { need(1, what); return base_[pos_++]; }
    uint32_t u32(const char* what) { need(4, what); uint32_t v = le32(base_ + pos_); pos_ += 4; return v; }
    uint64_t u64(const char* what) { need(8, what); uint64_t v = le64(base_ + pos_); pos_ += 8; return v; }
    int64_t  i64(const char* what) { return static_cast<int64_t>(u64(what)); }
    float    f32(const char* what) {
        const uint32_t bits = u32(what);
        float f;
        std::memcpy(&f, &bits, 4);
        return f;
    }

    std::string str(const std::string& what, bool allow_empty) {
        const uint32_t n = u32(what.c_str());
        if (n > kMaxString)
            fail(ErrorCategory::StringTooLong,
                 what + ": length " + std::to_string(n) + " exceeds " + std::to_string(kMaxString));
        need(n, what);
        std::string s(reinterpret_cast<const char*>(base_ + pos_), n);
        pos_ += n;
        if (!validUtf8(s)) fail(ErrorCategory::InvalidUtf8, what + ": invalid UTF-8");
        if (s.empty() && !allow_empty) fail(ErrorCategory::EmptyName, what + ": empty string");
        return s;
    }

    uint32_t count(const std::string& what, size_t min_entry) {
        const uint32_t n = u32(what.c_str());
        if (n > kMaxCount)
            fail(ErrorCategory::CountLimit,
                 what + ": count " + std::to_string(n) + " exceeds " + std::to_string(kMaxCount));
        need(static_cast<uint64_t>(n) * min_entry, what + " (" + std::to_string(n) + " entries)");
        return n;
    }

    void copy(uint8_t* out, size_t n) { std::memcpy(out, base_ + pos_, n); pos_ += n; }

private:
    const uint8_t* base_;
    size_t pos_, end_;
    const char* section_;
};

uint64_t elemSize(uint32_t dtype) { return dtype == kDtypeInt64 ? 8 : 4; }

TensorDesc readDesc(Cursor& c, const std::string& where) {
    TensorDesc d;
    d.name = c.str(where + " name", false);
    const std::string w = where + " '" + d.name + "'";
    d.dtype = c.u32("dtype");
    if (d.dtype != kDtypeFloat32 && d.dtype != kDtypeInt64)
        c.fail(ErrorCategory::UnsupportedDtype, w + ": dtype " + std::to_string(d.dtype) +
                                                " (supported: 1 float32, 7 int64)");
    const uint32_t rank = c.u32("rank");
    if (rank == 0 || rank > kMaxRank)
        c.fail(ErrorCategory::RankLimit, w + ": rank " + std::to_string(rank) + " outside 1..8");
    c.need(static_cast<uint64_t>(rank) * 8, w + " dims");
    uint64_t product = 1;
    d.dims.reserve(rank);
    for (uint32_t i = 0; i < rank; ++i) {
        const int64_t v = c.i64("dim");
        if (v < 1 || v > kMaxDim)
            c.fail(ErrorCategory::DimRange, w + ": dim " + std::to_string(i) + " = " +
                                            std::to_string(v) + " outside 1..2147483647");
        if (product > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(v))
            c.fail(ErrorCategory::Overflow, w + ": element count overflows u64");
        product *= static_cast<uint64_t>(v);
        d.dims.push_back(v);
    }
    if (product > std::numeric_limits<uint64_t>::max() / elemSize(d.dtype))
        c.fail(ErrorCategory::Overflow, w + ": byte size overflows u64");
    return d;
}

uint64_t byteSize(const TensorDesc& d) {
    uint64_t n = elemSize(d.dtype);
    for (int64_t v : d.dims) n *= static_cast<uint64_t>(v);  // overflow excluded by readDesc
    return n;
}

std::vector<TensorDesc> readDescList(Cursor& c, const char* tag) {
    const uint32_t n = c.count(std::string(tag) + " entries", kMinTensorDesc);
    std::vector<TensorDesc> out;
    out.reserve(n);
    for (uint32_t i = 0; i < n; ++i)
        out.push_back(readDesc(c, std::string(tag) + "[" + std::to_string(i) + "]"));
    return out;
}

void readMeta(Cursor& c, Meta& m) {
    m.ir_version    = c.u32("ir_version");
    m.opset_version = c.u32("opset_version");
    m.producer      = c.str("producer", true);
    c.need(32, "source_sha256");
    c.copy(m.source_sha256.data(), 32);
}

void readInits(Cursor& c, std::vector<Initializer>& out) {
    const uint32_t n = c.count("INIT entries", kMinInit);
    out.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        Initializer init;
        init.desc = readDesc(c, "INIT[" + std::to_string(i) + "]");
        const std::string w = "INIT[" + std::to_string(i) + "] '" + init.desc.name + "'";
        const uint64_t declared = c.u64("byte_length");
        const uint64_t expected = byteSize(init.desc);
        if (declared != expected)
            c.fail(ErrorCategory::ByteLength, w + ": byte_length " + std::to_string(declared) +
                                              " but dims and dtype require " + std::to_string(expected));
        c.need(declared, w + " data");
        init.data.resize(static_cast<size_t>(declared));
        c.copy(init.data.data(), static_cast<size_t>(declared));
        out.push_back(std::move(init));
    }
}

void readNodes(Cursor& c, std::vector<Node>& out) {
    const uint32_t n = c.count("NODE entries", kMinNode);
    out.reserve(n);
    for (uint32_t k = 0; k < n; ++k) {
        Node node;
        const std::string where = "NODE[" + std::to_string(k) + "]";
        node.op_type = c.str(where + " op_type", false);
        node.name    = c.str(where + " node_name", true);
        const std::string w = where + " '" + node.name + "' (" + node.op_type + ")";

        const uint32_t n_in = c.count(w + " inputs", kMinString);
        for (uint32_t i = 0; i < n_in; ++i)
            node.inputs.push_back(c.str(w + " input " + std::to_string(i), false));
        const uint32_t n_out = c.count(w + " outputs", kMinString);
        if (n_out == 0) c.fail(ErrorCategory::CountLimit, w + ": node has no outputs");
        for (uint32_t i = 0; i < n_out; ++i)
            node.outputs.push_back(c.str(w + " output " + std::to_string(i), false));

        const uint32_t n_attr = c.count(w + " attributes", kMinAttr);
        for (uint32_t i = 0; i < n_attr; ++i) {
            Attribute a;
            a.name = c.str(w + " attribute name", false);
            const std::string aw = w + " attribute '" + a.name + "'";
            a.kind = c.u8("attribute kind");
            switch (a.kind) {
                case kAttrInt:   a.i = c.i64("INT value"); break;
                case kAttrFloat: a.f = c.f32("FLOAT value"); break;
                case kAttrInts: {
                    const uint32_t cnt = c.count(aw + " INTS", 8);
                    a.ints.reserve(cnt);
                    for (uint32_t j = 0; j < cnt; ++j) a.ints.push_back(c.i64("INTS value"));
                    break;
                }
                default:
                    c.fail(ErrorCategory::UnsupportedAttrKind,
                           aw + ": kind " + std::to_string(a.kind) +
                           " (supported: 1 INT, 2 FLOAT, 3 INTS)");
            }
            node.attributes.push_back(std::move(a));
        }
        out.push_back(std::move(node));
    }
}

[[noreturn]] void graphFail(ErrorCategory c, const std::string& what) {
    throw FormatError(c, "graph: " + what);
}

void checkUnique(const std::vector<TensorDesc>& list, const char* tag,
                 std::unordered_set<std::string>& seen) {
    for (size_t i = 0; i < list.size(); ++i)
        if (!seen.insert(list[i].name).second)
            graphFail(ErrorCategory::DuplicateName, std::string(tag) + "[" + std::to_string(i) +
                                                    "] name '" + list[i].name + "' is not unique");
}

void validateGraph(const Model& m) {
    std::unordered_set<std::string> inputs, outputs, inits, vinf;
    checkUnique(m.inputs, "INPT", inputs);
    checkUnique(m.outputs, "OUTP", outputs);
    checkUnique(m.value_info, "VINF", vinf);
    for (size_t i = 0; i < m.initializers.size(); ++i) {
        const std::string& name = m.initializers[i].desc.name;
        if (!inits.insert(name).second || inputs.count(name))
            graphFail(ErrorCategory::DuplicateName,
                      "INIT[" + std::to_string(i) + "] name '" + name +
                      "' duplicates another initializer or a graph input");
    }

    std::unordered_set<std::string> defined(inputs);
    defined.insert(inits.begin(), inits.end());
    for (size_t k = 0; k < m.nodes.size(); ++k) {
        const Node& n = m.nodes[k];
        const std::string w = "node " + std::to_string(k) + " '" + n.name + "' (" + n.op_type + ")";
        for (const auto& in : n.inputs)
            if (!defined.count(in))
                graphFail(ErrorCategory::UndefinedInput,
                          w + ": input '" + in + "' is not a graph input, an initializer, "
                              "or an output of an earlier node");
        std::unordered_set<std::string> attr_names;
        for (const auto& a : n.attributes)
            if (!attr_names.insert(a.name).second)
                graphFail(ErrorCategory::DuplicateName, w + ": attribute '" + a.name + "' repeated");
        for (const auto& out : n.outputs)
            if (!defined.insert(out).second)
                graphFail(ErrorCategory::DuplicateName,
                          w + ": output '" + out + "' is already defined");
    }
    for (const auto& o : m.outputs)
        if (!defined.count(o.name))
            graphFail(ErrorCategory::UndefinedInput, "graph output '" + o.name + "' is never produced");
}

} // namespace

Model parse(const uint8_t* data, size_t size) {
    if (!hostIsLittleEndian())
        throw FormatError(ErrorCategory::HostEndianness, "the .fgx reader requires a little-endian host");
    if (size < kHeaderSize)
        throw FormatError(ErrorCategory::Truncated,
                          "file is " + std::to_string(size) + " bytes; the header alone is 24");
    if (size > kMaxFileSize)
        throw FormatError(ErrorCategory::Oversized,
                          "file is " + std::to_string(size) + " bytes; the limit is 1 GiB");
    if (std::memcmp(data, "FGXM", 4) != 0)
        throw FormatError(ErrorCategory::BadMagic,
                          "incompatible .fgx file: magic '" + printable(data, 4) + "', expected 'FGXM'");

    Model m;
    m.version_major = le16(data + 4);
    m.version_minor = le16(data + 6);
    if (m.version_major != kVersionMajor || m.version_minor > kVersionMinor)
        throw FormatError(ErrorCategory::BadVersion,
                          "incompatible .fgx file: version " + std::to_string(m.version_major) + "." +
                          std::to_string(m.version_minor) + ", this reader supports 1.0");
    const uint32_t flags = le32(data + 8);
    if (flags != 0)
        throw FormatError(ErrorCategory::BadFlags,
                          "incompatible .fgx file: flags " + std::to_string(flags) + ", expected 0");
    const uint64_t total = le64(data + 12);
    if (total != size)
        throw FormatError(ErrorCategory::SizeMismatch,
                          "header total_file_size " + std::to_string(total) +
                          " but the file is " + std::to_string(size) + " bytes");
    const uint32_t stored_crc = le32(data + 20);
    const uint32_t actual_crc = crc32(data + kHeaderSize, size - kHeaderSize);
    if (stored_crc != actual_crc)
        throw FormatError(ErrorCategory::CrcMismatch, "payload CRC-32 does not match the header");

    size_t pos = kHeaderSize;
    for (int s = 0; s < 6; ++s) {
        const char* tag = kSectionTags[s];
        if (pos == size)
            throw FormatError(ErrorCategory::SectionOrder,
                              std::string("missing section ") + tag + " at offset " + std::to_string(pos));
        if (size - pos < 12)
            throw FormatError(ErrorCategory::SectionLength,
                              std::string("truncated header for section ") + tag);
        if (std::memcmp(data + pos, tag, 4) != 0)
            throw FormatError(ErrorCategory::SectionOrder,
                              std::string("expected section ") + tag + " at offset " +
                              std::to_string(pos) + ", found '" + printable(data + pos, 4) + "'");
        const uint64_t len = le64(data + pos + 4);
        if (len > size - pos - 12)
            throw FormatError(ErrorCategory::SectionLength,
                              std::string("section ") + tag + " byte_length " + std::to_string(len) +
                              " exceeds the " + std::to_string(size - pos - 12) + " bytes remaining");
        const size_t begin = pos + 12, end = begin + static_cast<size_t>(len);
        Cursor c(data, begin, end, tag);
        switch (s) {
            case 0: readMeta(c, m.meta); break;
            case 1: m.inputs = readDescList(c, "INPT"); break;
            case 2: m.outputs = readDescList(c, "OUTP"); break;
            case 3: readInits(c, m.initializers); break;
            case 4: readNodes(c, m.nodes); break;
            case 5: m.value_info = readDescList(c, "VINF"); break;
        }
        if (c.remaining() != 0)
            c.fail(ErrorCategory::SectionLength,
                   "content ends " + std::to_string(c.remaining()) +
                   " bytes before the declared byte_length");
        pos = end;
    }
    if (pos != size)
        throw FormatError(ErrorCategory::TrailingBytes,
                          std::to_string(size - pos) + " bytes after section VINF");

    validateGraph(m);
    return m;
}

Model readFile(const std::string& path) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec) throw FormatError(ErrorCategory::Io, "cannot read '" + path + "': " + ec.message());
    if (size > kMaxFileSize)
        throw FormatError(ErrorCategory::Oversized,
                          "'" + path + "' is " + std::to_string(size) + " bytes; the limit is 1 GiB");
    std::ifstream f(path, std::ios::binary);
    if (!f) throw FormatError(ErrorCategory::Io, "cannot open '" + path + "'");
    std::vector<uint8_t> buf(static_cast<size_t>(size));
    if (size > 0) f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(size));
    if (static_cast<uintmax_t>(f.gcount()) != size)
        throw FormatError(ErrorCategory::Io, "short read on '" + path + "'");
    return parse(buf);
}

// ---------------------------------------------------------------------------
// Canonical manifest
// ---------------------------------------------------------------------------
namespace {

std::string esc(const std::string& s) {
    return printable(reinterpret_cast<const uint8_t*>(s.data()), s.size());
}

std::string dimsText(const std::vector<int64_t>& dims) {
    std::string out;
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i) out += 'x';
        out += std::to_string(dims[i]);
    }
    return out;
}

std::string hex32(uint32_t v) {
    char buf[16];
    std::snprintf(buf, sizeof buf, "%08x", v);
    return buf;
}

void descLines(std::string& out, const char* tag, const std::vector<TensorDesc>& list) {
    for (const auto& d : list)
        out += std::string(tag) + " " + esc(d.name) + " dtype=" + std::to_string(d.dtype) +
               " dims=" + dimsText(d.dims) + "\n";
}

std::string joinNames(const std::vector<std::string>& v) {
    std::string out;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) out += ',';
        out += esc(v[i]);
    }
    return out;
}

} // namespace

std::string dump(const Model& m) {
    std::string out = "FGX " + std::to_string(m.version_major) + "." +
                      std::to_string(m.version_minor) + "\n";
    std::string sha;
    for (uint8_t b : m.meta.source_sha256) {
        char buf[4];
        std::snprintf(buf, sizeof buf, "%02x", b);
        sha += buf;
    }
    out += "META ir=" + std::to_string(m.meta.ir_version) + " opset=" +
           std::to_string(m.meta.opset_version) + " producer=" + esc(m.meta.producer) +
           " sha256=" + sha + "\n";
    descLines(out, "INPT", m.inputs);
    descLines(out, "OUTP", m.outputs);
    for (const auto& init : m.initializers)
        out += "INIT " + esc(init.desc.name) + " dtype=" + std::to_string(init.desc.dtype) +
               " dims=" + dimsText(init.desc.dims) + " bytes=" + std::to_string(init.data.size()) +
               " crc32=" + hex32(crc32(init.data.data(), init.data.size())) + "\n";
    for (size_t k = 0; k < m.nodes.size(); ++k) {
        const Node& n = m.nodes[k];
        std::string attrs;
        for (size_t i = 0; i < n.attributes.size(); ++i) {
            const Attribute& a = n.attributes[i];
            if (i) attrs += ';';
            attrs += esc(a.name);
            if (a.kind == kAttrInt) {
                attrs += ":int=" + std::to_string(a.i);
            } else if (a.kind == kAttrFloat) {
                uint32_t bits;
                std::memcpy(&bits, &a.f, 4);
                attrs += ":float=0x" + hex32(bits);
            } else {
                attrs += ":ints=[";
                for (size_t j = 0; j < a.ints.size(); ++j) {
                    if (j) attrs += ',';
                    attrs += std::to_string(a.ints[j]);
                }
                attrs += "]";
            }
        }
        out += "NODE " + std::to_string(k) + " op=" + esc(n.op_type) + " name=" + esc(n.name) +
               " in=" + joinNames(n.inputs) + " out=" + joinNames(n.outputs) + " attrs=" + attrs + "\n";
    }
    descLines(out, "VINF", m.value_info);
    return out;
}

} // namespace fgx
} // namespace forgert
#pragma once

/**
 * @file fgx.h
 * @brief Strict reader for the ForgeRT .fgx v1 model format.
 *
 * Specification: docs/FGX_FORMAT.md. The reader validates the binary layout
 * and graph well-formedness only; operator semantics belong to the importer
 * (forgert/import/importer.h). It has no dependencies beyond the C++17
 * standard library.
 *
 * Every violation throws FormatError. No partially parsed model is ever
 * returned.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace forgert {
namespace fgx {

constexpr uint16_t kVersionMajor = 1;
constexpr uint16_t kVersionMinor = 0;
constexpr size_t   kHeaderSize   = 24;
constexpr uint64_t kMaxFileSize  = 1ull << 30;   // 1 GiB
constexpr uint32_t kMaxString    = 4096;
constexpr uint32_t kMaxCount     = 1000000;
constexpr uint32_t kMaxRank      = 8;
constexpr int64_t  kMaxDim       = 2147483647;

constexpr uint32_t kDtypeFloat32 = 1;  // ONNX TensorProto.FLOAT
constexpr uint32_t kDtypeInt64   = 7;  // ONNX TensorProto.INT64

// .fgx-specific attribute kinds (not ONNX's AttributeType numbering).
constexpr uint8_t kAttrInt   = 1;
constexpr uint8_t kAttrFloat = 2;
constexpr uint8_t kAttrInts  = 3;

/// Why a file was rejected. Tests assert on the category, users read the message.
enum class ErrorCategory {
    Io,
    HostEndianness,
    Truncated,
    Oversized,
    BadMagic,
    BadVersion,
    BadFlags,
    SizeMismatch,
    CrcMismatch,
    SectionOrder,
    SectionLength,
    TrailingBytes,
    StringTooLong,
    InvalidUtf8,
    EmptyName,
    CountLimit,
    RankLimit,
    DimRange,
    Overflow,
    ByteLength,
    UnsupportedDtype,
    UnsupportedAttrKind,
    DuplicateName,
    UndefinedInput,
};

const char* toString(ErrorCategory c);

class FormatError : public std::runtime_error {
public:
    FormatError(ErrorCategory category, const std::string& message)
        : std::runtime_error(std::string("fgx: ") + toString(category) + ": " + message),
          category_(category) {}
    ErrorCategory category() const { return category_; }
private:
    ErrorCategory category_;
};

struct TensorDesc {
    std::string          name;
    uint32_t             dtype = 0;   // kDtypeFloat32 or kDtypeInt64
    std::vector<int64_t> dims;        // rank 1..8, each dim 1..2^31-1
};

struct Initializer {
    TensorDesc           desc;
    std::vector<uint8_t> data;        // little-endian, C-contiguous
};

struct Attribute {
    std::string          name;
    uint8_t              kind = 0;    // kAttrInt / kAttrFloat / kAttrInts
    int64_t              i = 0;
    float                f = 0.0f;
    std::vector<int64_t> ints;
};

struct Node {
    std::string              op_type;
    std::string              name;    // may be empty
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<Attribute>   attributes;
};

struct Meta {
    uint32_t                ir_version    = 0;
    uint32_t                opset_version = 0;
    std::string             producer;
    std::array<uint8_t, 32> source_sha256{};
};

struct Model {
    uint16_t                 version_major = 0;
    uint16_t                 version_minor = 0;
    Meta                     meta;
    std::vector<TensorDesc>  inputs;
    std::vector<TensorDesc>  outputs;
    std::vector<Initializer> initializers;
    std::vector<Node>        nodes;
    std::vector<TensorDesc>  value_info;
};

/// CRC-32 (ISO-HDLC, same as zlib.crc32).
uint32_t crc32(const uint8_t* data, size_t size, uint32_t crc = 0);

/// Parse an in-memory .fgx image.
Model parse(const uint8_t* data, size_t size);
inline Model parse(const std::vector<uint8_t>& bytes) { return parse(bytes.data(), bytes.size()); }

/// Read and parse a .fgx file. The size limit is checked before reading.
Model readFile(const std::string& path);

/// Canonical manifest text (docs/FGX_FORMAT.md section 7).
std::string dump(const Model& model);

} // namespace fgx
} // namespace forgert
#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace playworld {

// Magic identifier: "PWMF" in little-endian ('P'=0x50, 'W'=0x57, 'M'=0x4D, 'F'=0x46)
constexpr uint32_t PWMF_MAGIC               = 0x464D5750;
constexpr uint32_t PWMF_MAGIC_LITTLE_ENDIAN = 0x464D5750;
constexpr uint32_t PWMF_MAGIC_BIG_ENDIAN    = 0x50574D46;

constexpr uint16_t PWMF_VERSION_MAJOR       = 1;
constexpr uint16_t PWMF_VERSION_MINOR       = 0;
constexpr uint32_t PWMF_DEFAULT_ALIGNMENT   = 64;

enum class QuantType : uint8_t {
    FP32         = 0, // 32-bit IEEE 754 float
    FP16         = 1, // 16-bit IEEE 754 half-precision float
    BF16         = 2, // 16-bit Brain Floating Point
    INT8_SYMM    = 3, // 8-bit signed symmetric integer (per-tensor/channel scale)
    INT4_BLOCK32 = 4, // 4-bit block-quantized (32 weights/block, 20 bytes/block)
    INT4_BLOCK64 = 5, // 4-bit block-quantized (64 weights/block, 36 bytes/block)
    FP8_E4M3     = 6, // 8-bit float: 1 sign, 4 exp, 3 mantissa (OCP format)
    FP8_E5M2     = 7  // 8-bit float: 1 sign, 5 exp, 2 mantissa (IEEE format)
};

enum class PWMFError : int32_t {
    Success               = 0,
    FileTooSmall          = -1,
    FileTruncated         = -2,
    InvalidMagic          = -3,
    UnsupportedVersion    = -4,
    AlignmentViolation    = -5,
    PayloadTruncated      = -6,
    OffsetOutOfBounds     = -7,
    UnsupportedQuant      = -8,
    ChecksumMismatch      = -9,
    IOError               = -10,
    InvalidDescriptor     = -11
};

inline const char* PWMFErrorToString(PWMFError err) noexcept {
    switch (err) {
        case PWMFError::Success:            return "Success";
        case PWMFError::FileTooSmall:       return "File size smaller than header";
        case PWMFError::FileTruncated:      return "File truncated unexpectedly";
        case PWMFError::InvalidMagic:       return "Invalid magic identifier bytes";
        case PWMFError::UnsupportedVersion: return "Unsupported PWMF format version";
        case PWMFError::AlignmentViolation: return "Memory alignment violation (expected 64-byte alignment)";
        case PWMFError::PayloadTruncated:   return "Tensor payload data truncated";
        case PWMFError::OffsetOutOfBounds:  return "Data offset points out of file bounds";
        case PWMFError::UnsupportedQuant:   return "Unsupported quantization type";
        case PWMFError::ChecksumMismatch:   return "CRC32 checksum verification failed";
        case PWMFError::IOError:            return "File I/O or memory mapping error";
        case PWMFError::InvalidDescriptor:  return "Invalid tensor descriptor in table";
        default:                            return "Unknown error";
    }
}

inline const char* GetQuantTypeName(QuantType qt) noexcept {
    switch (qt) {
        case QuantType::FP32:         return "FP32";
        case QuantType::FP16:         return "FP16";
        case QuantType::BF16:         return "BF16";
        case QuantType::INT8_SYMM:    return "INT8_SYMM";
        case QuantType::INT4_BLOCK32: return "INT4_BLOCK32";
        case QuantType::INT4_BLOCK64: return "INT4_BLOCK64";
        case QuantType::FP8_E4M3:     return "FP8_E4M3";
        case QuantType::FP8_E5M2:     return "FP8_E5M2";
        default:                      return "UNKNOWN";
    }
}

#pragma pack(push, 1)

// Container Header: Exactly 96 bytes
struct PWMFHeader {
    uint32_t magic;               // 0x00: 0x464D5750 ("PWMF")
    uint16_t version_major;       // 0x04: Format major version (1)
    uint16_t version_minor;       // 0x06: Format minor version (0)
    uint32_t header_size;         // 0x08: Total bytes of header (96)
    uint32_t num_tensors;         // 0x0C: Number of tensors in index table
    uint64_t metadata_offset;     // 0x10: Byte offset to JSON-LD metadata block
    uint64_t metadata_length;     // 0x18: Length of JSON-LD metadata block in bytes
    uint64_t tensor_table_offset; // 0x20: Byte offset to tensor descriptor table
    uint64_t tensor_table_length; // 0x28: Length of tensor descriptor table in bytes
    uint64_t weight_data_offset;  // 0x30: 64-byte aligned start of tensor payload blobs
    uint64_t weight_data_length;  // 0x38: Total payload length in bytes
    uint32_t alignment;           // 0x40: Data alignment in bytes (default: 64)
    uint8_t  reserved[28];        // 0x44: CRC32 checksum in bytes [0..3], remaining zero-filled

    [[nodiscard]] uint32_t GetCRC32() const noexcept {
        uint32_t crc = 0;
        std::memcpy(&crc, reserved, sizeof(uint32_t));
        return crc;
    }

    void SetCRC32(uint32_t crc) noexcept {
        std::memcpy(reserved, &crc, sizeof(uint32_t));
    }
};
static_assert(sizeof(PWMFHeader) == 96, "PWMFHeader must be exactly 96 bytes");
static_assert(offsetof(PWMFHeader, magic) == 0x00, "PWMFHeader::magic offset error");
static_assert(offsetof(PWMFHeader, version_major) == 0x04, "PWMFHeader::version_major offset error");
static_assert(offsetof(PWMFHeader, version_minor) == 0x06, "PWMFHeader::version_minor offset error");
static_assert(offsetof(PWMFHeader, header_size) == 0x08, "PWMFHeader::header_size offset error");
static_assert(offsetof(PWMFHeader, num_tensors) == 0x0C, "PWMFHeader::num_tensors offset error");
static_assert(offsetof(PWMFHeader, metadata_offset) == 0x10, "PWMFHeader::metadata_offset offset error");
static_assert(offsetof(PWMFHeader, metadata_length) == 0x18, "PWMFHeader::metadata_length offset error");
static_assert(offsetof(PWMFHeader, tensor_table_offset) == 0x20, "PWMFHeader::tensor_table_offset offset error");
static_assert(offsetof(PWMFHeader, tensor_table_length) == 0x28, "PWMFHeader::tensor_table_length offset error");
static_assert(offsetof(PWMFHeader, weight_data_offset) == 0x30, "PWMFHeader::weight_data_offset offset error");
static_assert(offsetof(PWMFHeader, weight_data_length) == 0x38, "PWMFHeader::weight_data_length offset error");
static_assert(offsetof(PWMFHeader, alignment) == 0x40, "PWMFHeader::alignment offset error");
static_assert(offsetof(PWMFHeader, reserved) == 0x44, "PWMFHeader::reserved offset error");

// Tensor Descriptor: Exactly 112 bytes
struct PWMFTensorDescriptor {
    char      name[64];           // 0x00: Null-terminated unique tensor identifier
    uint32_t  ndims;              // 0x40: Number of dimensions (1 to 5)
    uint32_t  shape[5];           // 0x44: Dimensions [D0, D1, D2, D3, D4]
    QuantType quant_type;         // 0x58: Quantization type enum
    uint8_t   reserved[3];        // 0x59: 3 bytes padding to align to 4-byte boundary
    uint64_t  data_offset;        // 0x5C: Byte offset relative to weight_data_offset
    uint64_t  data_bytes;         // 0x64: Total byte size of compressed tensor data
    float     global_scale;       // 0x6C: Global scale multiplier (fallback or per-tensor scale)
};
static_assert(sizeof(PWMFTensorDescriptor) == 112, "PWMFTensorDescriptor must be exactly 112 bytes");
static_assert(offsetof(PWMFTensorDescriptor, name) == 0x00, "PWMFTensorDescriptor::name offset error");
static_assert(offsetof(PWMFTensorDescriptor, ndims) == 0x40, "PWMFTensorDescriptor::ndims offset error");
static_assert(offsetof(PWMFTensorDescriptor, shape) == 0x44, "PWMFTensorDescriptor::shape offset error");
static_assert(offsetof(PWMFTensorDescriptor, quant_type) == 0x58, "PWMFTensorDescriptor::quant_type offset error");
static_assert(offsetof(PWMFTensorDescriptor, reserved) == 0x59, "PWMFTensorDescriptor::reserved offset error");
static_assert(offsetof(PWMFTensorDescriptor, data_offset) == 0x5C, "PWMFTensorDescriptor::data_offset offset error");
static_assert(offsetof(PWMFTensorDescriptor, data_bytes) == 0x64, "PWMFTensorDescriptor::data_bytes offset error");
static_assert(offsetof(PWMFTensorDescriptor, global_scale) == 0x6C, "PWMFTensorDescriptor::global_scale offset error");

// AWQ-style 4-bit Block-32 layout: Exactly 20 bytes
struct INT4Block32 {
    uint8_t  qs[16];     // 16 bytes: 32 packed 4-bit unsigned integers (2 nibbles per byte)
    uint16_t scale_fp16; // IEEE 754 half-precision float scale factor (alpha)
    uint16_t bias_fp16;  // IEEE 754 half-precision float zero-point / bias (beta)
};
static_assert(sizeof(INT4Block32) == 20, "INT4Block32 must be exactly 20 bytes");
static_assert(offsetof(INT4Block32, qs) == 0x00, "INT4Block32::qs offset error");
static_assert(offsetof(INT4Block32, scale_fp16) == 0x10, "INT4Block32::scale_fp16 offset error");
static_assert(offsetof(INT4Block32, bias_fp16) == 0x12, "INT4Block32::bias_fp16 offset error");

// 4-bit Block-64 layout: Exactly 36 bytes
struct INT4Block64 {
    uint8_t  qs[32];     // 32 bytes: 64 packed 4-bit unsigned integers
    uint16_t scale_fp16; // IEEE 754 half-precision float scale
    uint16_t bias_fp16;  // IEEE 754 half-precision float bias
};
static_assert(sizeof(INT4Block64) == 36, "INT4Block64 must be exactly 36 bytes");

#pragma pack(pop)

// ISO 3309 / IEEE 802.3 CRC32 checksum computation with polynomial 0xEDB88320
uint32_t ComputeCRC32(const uint8_t* data, size_t length, uint32_t initial_crc = 0xFFFFFFFF) noexcept;

class Tensor;

// ============================================================================
// PWMF Container Parser: Memory-mapped & Fallback Zero-Copy Container Reader
// ============================================================================

class PWMFParser {
public:
    PWMFParser();
    ~PWMFParser();

    PWMFParser(const PWMFParser&) = delete;
    PWMFParser& operator=(const PWMFParser&) = delete;

    PWMFParser(PWMFParser&& other) noexcept;
    PWMFParser& operator=(PWMFParser&& other) noexcept;

    // Opens and memory-maps a .pwmf file (with POSIX mmap and buffer fallback)
    bool Open(const std::string& path);
    bool ParseFile(const std::string& path) { return Open(path); }

    // Parses a memory buffer directly (zero-copy)
    bool ParseMemory(const uint8_t* data, size_t size);

    // Closes and unmaps any active file mapping
    void Close();

    // Accessors
    [[nodiscard]] bool IsLoaded() const noexcept { return is_loaded_; }
    [[nodiscard]] const PWMFHeader& GetHeader() const noexcept { return header_; }
    [[nodiscard]] uint32_t GetNumTensors() const noexcept { return header_.num_tensors; }

    [[nodiscard]] const PWMFTensorDescriptor* GetTensorDescriptor(size_t index) const noexcept;
    [[nodiscard]] const PWMFTensorDescriptor* FindTensorDescriptor(const std::string& name) const noexcept;

    [[nodiscard]] const uint8_t* GetTensorData(size_t index) const noexcept;
    [[nodiscard]] const uint8_t* GetTensorData(const std::string& name) const noexcept;

    [[nodiscard]] Tensor GetTensor(size_t index) const;
    [[nodiscard]] Tensor GetTensor(const std::string& name) const;

    [[nodiscard]] std::string GetMetadataJSON() const;

    [[nodiscard]] PWMFError GetLastError() const noexcept { return last_error_; }
    [[nodiscard]] const char* GetLastErrorString() const noexcept { return PWMFErrorToString(last_error_); }

    // Verifies CRC32 checksum against embedded header checksum
    [[nodiscard]] bool VerifyCRC32() const noexcept;

private:
    PWMFError last_error_{PWMFError::Success};
    bool is_loaded_{false};
    PWMFHeader header_{};

    const uint8_t* data_ptr_{nullptr};
    size_t data_size_{0};

    // POSIX mmap state
    int fd_{-1};
    void* mmap_addr_{nullptr};
    size_t mmap_size_{0};

    // Fallback buffer when mmap is unavailable
    std::vector<uint8_t> fallback_buffer_;

    // Parsed descriptors
    std::vector<PWMFTensorDescriptor> tensor_descriptors_;
    std::vector<size_t> tensor_payload_offsets_;
};

// ============================================================================
// PWMF Container Serializer / Writer
// ============================================================================

class PWMFWriter {
public:
    PWMFWriter();
    ~PWMFWriter() = default;

    // Set JSON-LD global model metadata
    void SetMetadata(const std::string& json_str);

    // Add tensor descriptor and payload blob
    void AddTensor(const std::string& name, uint32_t ndims, const uint32_t shape[5],
                   QuantType quant_type, const void* data, size_t data_bytes,
                   float global_scale = 1.0f);

    // Add Tensor object directly
    void AddTensor(const Tensor& tensor);

    // Serializes the full .pwmf container into a contiguous byte buffer with CRC32
    [[nodiscard]] std::vector<uint8_t> Serialize();

    // Directly saves to disk
    bool SaveToFile(const std::string& path);

    void Reset();

private:
    struct PendingTensor {
        PWMFTensorDescriptor descriptor;
        std::vector<uint8_t> payload;
    };

    std::string metadata_json_;
    std::vector<PendingTensor> tensors_;
    uint32_t alignment_{PWMF_DEFAULT_ALIGNMENT};
};

// Standalone export helper
bool SavePWMF(const std::string& path, const std::vector<Tensor>& tensors,
              const std::string& metadata_json = "{}");

} // namespace playworld


#include "playworld/pwmf_format.h"
#include "playworld/tensor.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#endif

namespace playworld {

// ============================================================================
// ISO 3309 / IEEE 802.3 CRC32 Checksum Implementation
// ============================================================================

#if !defined(__ARM_FEATURE_CRC32)
static constexpr uint32_t MakeCRC32TableEntry(uint32_t index) noexcept {
    uint32_t crc = index;
    for (int j = 0; j < 8; ++j) {
        crc = (crc & 1) ? (0xEDB88320U ^ (crc >> 1)) : (crc >> 1);
    }
    return crc;
}

struct CRC32Table {
    uint32_t table[256];
    constexpr CRC32Table() : table{} {
        for (uint32_t i = 0; i < 256; ++i) {
            table[i] = MakeCRC32TableEntry(i);
        }
    }
};

static constexpr CRC32Table g_crc32_table;
#endif

uint32_t ComputeCRC32(const uint8_t* data, size_t length, uint32_t initial_crc) noexcept {
    if (!data || length == 0) {
        return 0;
    }

    uint32_t crc = initial_crc;

#if defined(__ARM_FEATURE_CRC32)
    // Hardware accelerated ISO 3309 / IEEE 802.3 CRC32 on ARMv8/v9
    size_t i = 0;
    while (i + 8 <= length) {
        uint64_t v = 0;
        std::memcpy(&v, data + i, sizeof(uint64_t));
        crc = __crc32d(crc, v);
        i += 8;
    }
    while (i < length) {
        crc = __crc32b(crc, data[i]);
        i += 1;
    }
    return crc ^ 0xFFFFFFFFU;
#else
    for (size_t i = 0; i < length; ++i) {
        crc = g_crc32_table.table[(crc ^ data[i]) & 0xFFU] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFU;
#endif
}

// Helper to calculate aligned offset
static inline uint64_t AlignTo(uint64_t offset, uint64_t alignment) noexcept {
    return (offset + alignment - 1) & ~(alignment - 1);
}

// ============================================================================
// PWMFParser Implementation
// ============================================================================

PWMFParser::PWMFParser() = default;

PWMFParser::~PWMFParser() {
    Close();
}

PWMFParser::PWMFParser(PWMFParser&& other) noexcept
    : last_error_(other.last_error_),
      is_loaded_(other.is_loaded_),
      header_(other.header_),
      data_ptr_(other.data_ptr_),
      data_size_(other.data_size_),
      fd_(other.fd_),
      mmap_addr_(other.mmap_addr_),
      mmap_size_(other.mmap_size_),
      fallback_buffer_(std::move(other.fallback_buffer_)),
      tensor_descriptors_(std::move(other.tensor_descriptors_)),
      tensor_payload_offsets_(std::move(other.tensor_payload_offsets_)) {
    other.is_loaded_ = false;
    other.data_ptr_ = nullptr;
    other.data_size_ = 0;
    other.fd_ = -1;
    other.mmap_addr_ = nullptr;
    other.mmap_size_ = 0;
}

PWMFParser& PWMFParser::operator=(PWMFParser&& other) noexcept {
    if (this != &other) {
        Close();

        last_error_ = other.last_error_;
        is_loaded_ = other.is_loaded_;
        header_ = other.header_;
        data_ptr_ = other.data_ptr_;
        data_size_ = other.data_size_;
        fd_ = other.fd_;
        mmap_addr_ = other.mmap_addr_;
        mmap_size_ = other.mmap_size_;
        fallback_buffer_ = std::move(other.fallback_buffer_);
        tensor_descriptors_ = std::move(other.tensor_descriptors_);
        tensor_payload_offsets_ = std::move(other.tensor_payload_offsets_);

        other.is_loaded_ = false;
        other.data_ptr_ = nullptr;
        other.data_size_ = 0;
        other.fd_ = -1;
        other.mmap_addr_ = nullptr;
        other.mmap_size_ = 0;
    }
    return *this;
}

void PWMFParser::Close() {
    if (mmap_addr_ != nullptr && mmap_size_ > 0) {
        munmap(mmap_addr_, mmap_size_);
        mmap_addr_ = nullptr;
        mmap_size_ = 0;
    }
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
    fallback_buffer_.clear();
    tensor_descriptors_.clear();
    tensor_payload_offsets_.clear();
    data_ptr_ = nullptr;
    data_size_ = 0;
    is_loaded_ = false;
    std::memset(&header_, 0, sizeof(header_));
    last_error_ = PWMFError::Success;
}

bool PWMFParser::Open(const std::string& path) {
    Close();

    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        last_error_ = PWMFError::IOError;
        return false;
    }

    struct stat st{};
    if (fstat(fd_, &st) != 0) {
        close(fd_);
        fd_ = -1;
        last_error_ = PWMFError::IOError;
        return false;
    }

    const size_t file_size = static_cast<size_t>(st.st_size);
    if (file_size < sizeof(PWMFHeader)) {
        close(fd_);
        fd_ = -1;
        last_error_ = PWMFError::FileTooSmall;
        return false;
    }

    // Attempt zero-copy POSIX mmap
    void* addr = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd_, 0);
    if (addr != MAP_FAILED) {
        mmap_addr_ = addr;
        mmap_size_ = file_size;
        return ParseMemory(static_cast<const uint8_t*>(addr), file_size);
    }

    // Fallback: allocate memory and read into buffer
    try {
        fallback_buffer_.resize(file_size);
        ssize_t bytes_read = pread(fd_, fallback_buffer_.data(), file_size, 0);
        if (bytes_read != static_cast<ssize_t>(file_size)) {
            Close();
            last_error_ = PWMFError::IOError;
            return false;
        }
        return ParseMemory(fallback_buffer_.data(), file_size);
    } catch (...) {
        Close();
        last_error_ = PWMFError::IOError;
        return false;
    }
}

bool PWMFParser::ParseMemory(const uint8_t* data, size_t size) {
    if (!data || size < sizeof(PWMFHeader)) {
        last_error_ = PWMFError::FileTooSmall;
        return false;
    }

    data_ptr_ = data;
    data_size_ = size;

    // 1. Read header
    std::memcpy(&header_, data, sizeof(PWMFHeader));

    // 2. Validate magic bytes (0x464D5750 / "PWMF")
    if (header_.magic != PWMF_MAGIC) {
        last_error_ = PWMFError::InvalidMagic;
        return false;
    }

    // 3. Validate version
    if (header_.version_major != PWMF_VERSION_MAJOR) {
        last_error_ = PWMFError::UnsupportedVersion;
        return false;
    }

    // 4. Validate header size
    if (header_.header_size < sizeof(PWMFHeader)) {
        last_error_ = PWMFError::FileTruncated;
        return false;
    }

    // 5. Validate alignment (default 64)
    const uint32_t align = header_.alignment ? header_.alignment : PWMF_DEFAULT_ALIGNMENT;
    if (header_.weight_data_offset % align != 0) {
        last_error_ = PWMFError::AlignmentViolation;
        return false;
    }

    // 6. Validate bounds of sections
    if (header_.metadata_offset + header_.metadata_length > size) {
        last_error_ = PWMFError::OffsetOutOfBounds;
        return false;
    }

    if (header_.tensor_table_offset + header_.tensor_table_length > size) {
        last_error_ = PWMFError::OffsetOutOfBounds;
        return false;
    }

    const size_t expected_table_bytes = static_cast<size_t>(header_.num_tensors) * sizeof(PWMFTensorDescriptor);
    if (header_.tensor_table_length != expected_table_bytes) {
        last_error_ = PWMFError::InvalidDescriptor;
        return false;
    }

    if (header_.weight_data_offset + header_.weight_data_length > size) {
        last_error_ = PWMFError::PayloadTruncated;
        return false;
    }

    // 7. Parse and validate tensor descriptors
    tensor_descriptors_.clear();
    tensor_descriptors_.reserve(header_.num_tensors);
    tensor_payload_offsets_.clear();
    tensor_payload_offsets_.reserve(header_.num_tensors);

    const uint8_t* table_ptr = data + header_.tensor_table_offset;
    for (uint32_t i = 0; i < header_.num_tensors; ++i) {
        PWMFTensorDescriptor desc{};
        std::memcpy(&desc, table_ptr + i * sizeof(PWMFTensorDescriptor), sizeof(PWMFTensorDescriptor));

        // Enforce null-termination of tensor name
        desc.name[sizeof(desc.name) - 1] = '\0';

        // Check dimensions
        if (desc.ndims < 1 || desc.ndims > 5) {
            last_error_ = PWMFError::InvalidDescriptor;
            return false;
        }

        // Check quantization enum validity
        const uint8_t qt = static_cast<uint8_t>(desc.quant_type);
        if (qt > static_cast<uint8_t>(QuantType::FP8_E5M2)) {
            last_error_ = PWMFError::UnsupportedQuant;
            return false;
        }

        // Check payload offsets within weight data blob
        if (desc.data_offset + desc.data_bytes > header_.weight_data_length) {
            last_error_ = PWMFError::OffsetOutOfBounds;
            return false;
        }

        tensor_descriptors_.push_back(desc);
        tensor_payload_offsets_.push_back(header_.weight_data_offset + desc.data_offset);
    }

    // 8. Verify CRC32 checksum if header contains one
    if (header_.GetCRC32() != 0) {
        if (!VerifyCRC32()) {
            last_error_ = PWMFError::ChecksumMismatch;
            return false;
        }
    }

    is_loaded_ = true;
    last_error_ = PWMFError::Success;
    return true;
}

bool PWMFParser::VerifyCRC32() const noexcept {
    if (!data_ptr_ || data_size_ <= sizeof(PWMFHeader)) {
        return false;
    }
    const uint32_t expected_crc = header_.GetCRC32();
    if (expected_crc == 0) {
        return true;
    }

    // Compute CRC32 across all data following the 96-byte header
    const uint32_t computed_crc = ComputeCRC32(data_ptr_ + sizeof(PWMFHeader), data_size_ - sizeof(PWMFHeader));
    return computed_crc == expected_crc;
}

const PWMFTensorDescriptor* PWMFParser::GetTensorDescriptor(size_t index) const noexcept {
    if (index >= tensor_descriptors_.size()) {
        return nullptr;
    }
    return &tensor_descriptors_[index];
}

const PWMFTensorDescriptor* PWMFParser::FindTensorDescriptor(const std::string& name) const noexcept {
    for (const auto& desc : tensor_descriptors_) {
        if (std::string_view(desc.name) == name) {
            return &desc;
        }
    }
    return nullptr;
}

const uint8_t* PWMFParser::GetTensorData(size_t index) const noexcept {
    if (!data_ptr_ || index >= tensor_descriptors_.size()) {
        return nullptr;
    }
    return data_ptr_ + tensor_payload_offsets_[index];
}

const uint8_t* PWMFParser::GetTensorData(const std::string& name) const noexcept {
    for (size_t i = 0; i < tensor_descriptors_.size(); ++i) {
        if (std::string_view(tensor_descriptors_[i].name) == name) {
            return GetTensorData(i);
        }
    }
    return nullptr;
}

Tensor PWMFParser::GetTensor(size_t index) const {
    const auto* desc = GetTensorDescriptor(index);
    if (!desc) {
        throw std::out_of_range("Tensor index out of range in PWMFParser::GetTensor");
    }
    const uint8_t* payload = GetTensorData(index);
    return Tensor(desc->name, desc->ndims, desc->shape, desc->quant_type,
                  payload, desc->data_bytes, desc->global_scale);
}

Tensor PWMFParser::GetTensor(const std::string& name) const {
    const auto* desc = FindTensorDescriptor(name);
    if (!desc) {
        throw std::runtime_error("Tensor '" + name + "' not found in PWMF container");
    }
    const uint8_t* payload = GetTensorData(name);
    return Tensor(desc->name, desc->ndims, desc->shape, desc->quant_type,
                  payload, desc->data_bytes, desc->global_scale);
}

std::string PWMFParser::GetMetadataJSON() const {
    if (!data_ptr_ || header_.metadata_length == 0) {
        return "{}";
    }
    return std::string(reinterpret_cast<const char*>(data_ptr_ + header_.metadata_offset),
                       static_cast<size_t>(header_.metadata_length));
}

// ============================================================================
// PWMFWriter Implementation
// ============================================================================

PWMFWriter::PWMFWriter() = default;

void PWMFWriter::Reset() {
    metadata_json_.clear();
    tensors_.clear();
    alignment_ = PWMF_DEFAULT_ALIGNMENT;
}

void PWMFWriter::SetMetadata(const std::string& json_str) {
    metadata_json_ = json_str;
}

void PWMFWriter::AddTensor(const std::string& name, uint32_t ndims, const uint32_t shape[5],
                           QuantType quant_type, const void* data, size_t data_bytes,
                           float global_scale) {
    PendingTensor pt{};
    std::memset(&pt.descriptor, 0, sizeof(PWMFTensorDescriptor));

    const size_t copy_len = std::min(name.size(), sizeof(pt.descriptor.name) - 1);
    std::memcpy(pt.descriptor.name, name.data(), copy_len);
    pt.descriptor.name[copy_len] = '\0';

    pt.descriptor.ndims = ndims;
    if (shape) {
        for (uint32_t i = 0; i < 5; ++i) {
            pt.descriptor.shape[i] = shape[i];
        }
    }
    pt.descriptor.quant_type = quant_type;
    pt.descriptor.data_bytes = data_bytes;
    pt.descriptor.global_scale = global_scale;

    if (data && data_bytes > 0) {
        pt.payload.resize(data_bytes);
        std::memcpy(pt.payload.data(), data, data_bytes);
    }

    tensors_.push_back(std::move(pt));
}

void PWMFWriter::AddTensor(const Tensor& tensor) {
    AddTensor(tensor.name(), tensor.ndims(), tensor.shape().data(),
              tensor.quant_type(), tensor.data(), tensor.size_bytes(),
              tensor.global_scale());
}

std::vector<uint8_t> PWMFWriter::Serialize() {
    // 1. Calculate offsets and lengths
    const uint64_t header_len = sizeof(PWMFHeader); // 96
    const uint64_t meta_offset = header_len;
    const uint64_t meta_len = metadata_json_.size();

    // Align tensor table to alignment boundary (64 bytes)
    const uint64_t table_offset = AlignTo(meta_offset + meta_len, alignment_);
    const uint64_t table_len = static_cast<uint64_t>(tensors_.size()) * sizeof(PWMFTensorDescriptor);

    // Align weight data to alignment boundary (64 bytes)
    const uint64_t weight_offset = AlignTo(table_offset + table_len, alignment_);

    // Compute layout of individual tensors within weight data
    uint64_t current_payload_offset = 0;
    for (auto& t : tensors_) {
        // Align each tensor payload to 64 bytes for DMA/GPU SIMD performance
        current_payload_offset = AlignTo(current_payload_offset, alignment_);
        t.descriptor.data_offset = current_payload_offset;
        current_payload_offset += t.payload.size();
    }
    const uint64_t weight_len = current_payload_offset;
    const uint64_t total_size = weight_offset + weight_len;

    // 2. Allocate contiguous buffer
    std::vector<uint8_t> buffer(total_size, 0);

    // 3. Prepare Header
    PWMFHeader header{};
    header.magic = PWMF_MAGIC;
    header.version_major = PWMF_VERSION_MAJOR;
    header.version_minor = PWMF_VERSION_MINOR;
    header.header_size = static_cast<uint32_t>(header_len);
    header.num_tensors = static_cast<uint32_t>(tensors_.size());
    header.metadata_offset = meta_offset;
    header.metadata_length = meta_len;
    header.tensor_table_offset = table_offset;
    header.tensor_table_length = table_len;
    header.weight_data_offset = weight_offset;
    header.weight_data_length = weight_len;
    header.alignment = alignment_;

    // 4. Write metadata
    if (meta_len > 0) {
        std::memcpy(buffer.data() + meta_offset, metadata_json_.data(), meta_len);
    }

    // 5. Write tensor descriptors and payload blobs
    for (size_t i = 0; i < tensors_.size(); ++i) {
        const auto& t = tensors_[i];
        // Descriptor in table
        std::memcpy(buffer.data() + table_offset + i * sizeof(PWMFTensorDescriptor),
                    &t.descriptor, sizeof(PWMFTensorDescriptor));

        // Weight payload
        if (!t.payload.empty()) {
            std::memcpy(buffer.data() + weight_offset + t.descriptor.data_offset,
                        t.payload.data(), t.payload.size());
        }
    }

    // 6. Compute CRC32 across all payload data (excluding the 96-byte header)
    const uint32_t crc = ComputeCRC32(buffer.data() + header_len, total_size - header_len);
    header.SetCRC32(crc);

    // 7. Write finalized header at offset 0
    std::memcpy(buffer.data(), &header, sizeof(PWMFHeader));

    return buffer;
}

bool PWMFWriter::SaveToFile(const std::string& path) {
    const std::vector<uint8_t> data = Serialize();
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        return false;
    }
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return out.good();
}

bool SavePWMF(const std::string& path, const std::vector<Tensor>& tensors,
              const std::string& metadata_json) {
    PWMFWriter writer;
    if (!metadata_json.empty()) {
        writer.SetMetadata(metadata_json);
    }
    for (const auto& t : tensors) {
        writer.AddTensor(t);
    }
    return writer.SaveToFile(path);
}

} // namespace playworld

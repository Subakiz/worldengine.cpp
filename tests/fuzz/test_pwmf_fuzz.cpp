#include "test_runner.h"
#include "playworld/pwmf_format.h"
#include "playworld/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

using namespace playworld;

// Native test runner macro compatibility aliases
#define PW_TEST_SUITE(Name)
#define PW_TEST_CASE(Suite, Name) TEST(Suite, Name)
#define PW_CHECK_EQ(a, b) EXPECT_EQ(a, b)
#define PW_CHECK_NE(a, b) EXPECT_NE(a, b)
#define PW_CHECK_GT(a, b) EXPECT_GT(a, b)
#define PW_CHECK_LT(a, b) EXPECT_LT(a, b)
#define PW_CHECK_GE(a, b) EXPECT_GE(a, b)
#define PW_CHECK_LE(a, b) EXPECT_LE(a, b)
#define PW_CHECK_TRUE(expr) EXPECT_TRUE(expr)
#define PW_CHECK_FALSE(expr) EXPECT_FALSE(expr)
#define PW_ASSERT_EQ(a, b) ASSERT_EQ(a, b)
#define PW_ASSERT_NE(a, b) ASSERT_NE(a, b)
#define PW_ASSERT_GT(a, b) ASSERT_GT(a, b)
#define PW_ASSERT_LT(a, b) ASSERT_LT(a, b)
#define PW_ASSERT_TRUE(expr) ASSERT_TRUE(expr)
#define PW_ASSERT_FALSE(expr) ASSERT_FALSE(expr)
#define PW_ASSERT_NEAR(a, b, eps) ASSERT_NEAR(a, b, eps)

// ============================================================================
// Helper: Helper to construct a valid minimal serialized .PWMF model buffer
// ============================================================================
static std::vector<uint8_t> CreateMinimalValidPWMFBuffer() {
    PWMFWriter writer;
    writer.SetMetadata("{\"model_type\": \"fuzz_target\"}");
    std::vector<float> data(32, 1.0f);
    uint32_t shape[5] = {32, 0, 0, 0, 0};
    writer.AddTensor("layer.weight", 1, shape, QuantType::FP32,
                     data.data(), data.size() * sizeof(float));
    return writer.Serialize();
}

// ============================================================================
// 1. FileTooSmall_EmptyBuffer
// buffer size 0 bytes -> returns false, last_error == PWMFError::FileTooSmall (-1)
// ============================================================================
PW_TEST_CASE(PWMFFuzzSuite, FileTooSmall_EmptyBuffer) {
    PWMFParser parser;
    const uint8_t dummy = 0;

    // Zero-byte buffer with valid non-null pointer
    bool ok = parser.ParseMemory(&dummy, 0);
    PW_CHECK_FALSE(ok);
    PW_CHECK_EQ(parser.GetLastError(), PWMFError::FileTooSmall);

    // Verify error code integer value matches specification (-1)
    PW_CHECK_EQ(static_cast<int32_t>(parser.GetLastError()), -1);
}

// ============================================================================
// 2. FileTooSmall_PartialHeader
// buffer size 48 bytes (< 96) -> returns false, last_error == PWMFError::FileTooSmall (-1)
// ============================================================================
PW_TEST_CASE(PWMFFuzzSuite, FileTooSmall_PartialHeader) {
    PWMFParser parser;
    std::vector<uint8_t> short_buf(48, 0xCC);

    bool ok = parser.ParseMemory(short_buf.data(), short_buf.size());
    PW_CHECK_FALSE(ok);
    PW_CHECK_EQ(parser.GetLastError(), PWMFError::FileTooSmall);
    PW_CHECK_EQ(static_cast<int32_t>(parser.GetLastError()), -1);

    // Also test 1 byte and 95 bytes boundary conditions
    std::vector<uint8_t> one_byte(1, 0xFF);
    PW_CHECK_FALSE(parser.ParseMemory(one_byte.data(), one_byte.size()));
    PW_CHECK_EQ(parser.GetLastError(), PWMFError::FileTooSmall);

    std::vector<uint8_t> boundary_95(95, 0xAA);
    PW_CHECK_FALSE(parser.ParseMemory(boundary_95.data(), boundary_95.size()));
    PW_CHECK_EQ(parser.GetLastError(), PWMFError::FileTooSmall);
}

// ============================================================================
// 3. InvalidMagic_CorruptedMagicWord
// magic set to 0x12345678 or 0xDEADBEEF -> returns false, last_error == PWMFError::InvalidMagic (-3)
// ============================================================================
PW_TEST_CASE(PWMFFuzzSuite, InvalidMagic_CorruptedMagicWord) {
    PWMFHeader hdr{};
    hdr.magic = 0xDEADBEEF; // Corrupted magic word
    hdr.version_major = PWMF_VERSION_MAJOR;
    hdr.header_size = sizeof(PWMFHeader);

    PWMFParser parser1;
    bool ok1 = parser1.ParseMemory(reinterpret_cast<const uint8_t*>(&hdr), sizeof(hdr));
    PW_CHECK_FALSE(ok1);
    PW_CHECK_EQ(parser1.GetLastError(), PWMFError::InvalidMagic);
    PW_CHECK_EQ(static_cast<int32_t>(parser1.GetLastError()), -3);

    // Test alternative corruption 0x12345678
    hdr.magic = 0x12345678;
    PWMFParser parser2;
    bool ok2 = parser2.ParseMemory(reinterpret_cast<const uint8_t*>(&hdr), sizeof(hdr));
    PW_CHECK_FALSE(ok2);
    PW_CHECK_EQ(parser2.GetLastError(), PWMFError::InvalidMagic);
    PW_CHECK_EQ(static_cast<int32_t>(parser2.GetLastError()), -3);

    // Single-bit flip in magic word
    hdr.magic = PWMF_MAGIC ^ 0x00000001;
    PWMFParser parser3;
    bool ok3 = parser3.ParseMemory(reinterpret_cast<const uint8_t*>(&hdr), sizeof(hdr));
    PW_CHECK_FALSE(ok3);
    PW_CHECK_EQ(parser3.GetLastError(), PWMFError::InvalidMagic);
}

// ============================================================================
// 4. UnsupportedVersion_MajorVersionTooHigh
// version_major set to 2 or 99 -> returns false, last_error == PWMFError::UnsupportedVersion (-4)
// ============================================================================
PW_TEST_CASE(PWMFFuzzSuite, UnsupportedVersion_MajorVersionTooHigh) {
    PWMFHeader hdr{};
    hdr.magic = PWMF_MAGIC;
    hdr.version_major = 2; // Incompatible major version 2
    hdr.header_size = sizeof(PWMFHeader);

    PWMFParser parser1;
    bool ok1 = parser1.ParseMemory(reinterpret_cast<const uint8_t*>(&hdr), sizeof(hdr));
    PW_CHECK_FALSE(ok1);
    PW_CHECK_EQ(parser1.GetLastError(), PWMFError::UnsupportedVersion);
    PW_CHECK_EQ(static_cast<int32_t>(parser1.GetLastError()), -4);

    // Major version 99
    hdr.version_major = 99;
    PWMFParser parser2;
    bool ok2 = parser2.ParseMemory(reinterpret_cast<const uint8_t*>(&hdr), sizeof(hdr));
    PW_CHECK_FALSE(ok2);
    PW_CHECK_EQ(parser2.GetLastError(), PWMFError::UnsupportedVersion);
    PW_CHECK_EQ(static_cast<int32_t>(parser2.GetLastError()), -4);

    // Major version 0
    hdr.version_major = 0;
    PWMFParser parser3;
    bool ok3 = parser3.ParseMemory(reinterpret_cast<const uint8_t*>(&hdr), sizeof(hdr));
    PW_CHECK_FALSE(ok3);
    PW_CHECK_EQ(parser3.GetLastError(), PWMFError::UnsupportedVersion);
}

// ============================================================================
// 5. FileTruncated_HeaderSizeFieldInvalid
// header_size set to 64 (< 96) -> returns false, last_error == PWMFError::FileTruncated (-2)
// ============================================================================
PW_TEST_CASE(PWMFFuzzSuite, FileTruncated_HeaderSizeFieldInvalid) {
    PWMFHeader hdr{};
    hdr.magic = PWMF_MAGIC;
    hdr.version_major = PWMF_VERSION_MAJOR;
    hdr.header_size = 64; // Corrupted field: header claims size < 96 bytes

    PWMFParser parser1;
    bool ok1 = parser1.ParseMemory(reinterpret_cast<const uint8_t*>(&hdr), sizeof(hdr));
    PW_CHECK_FALSE(ok1);
    PW_CHECK_EQ(parser1.GetLastError(), PWMFError::FileTruncated);
    PW_CHECK_EQ(static_cast<int32_t>(parser1.GetLastError()), -2);

    // Also test header_size = 0
    hdr.header_size = 0;
    PWMFParser parser2;
    bool ok2 = parser2.ParseMemory(reinterpret_cast<const uint8_t*>(&hdr), sizeof(hdr));
    PW_CHECK_FALSE(ok2);
    PW_CHECK_EQ(parser2.GetLastError(), PWMFError::FileTruncated);

    // header_size = 95
    hdr.header_size = 95;
    PWMFParser parser3;
    bool ok3 = parser3.ParseMemory(reinterpret_cast<const uint8_t*>(&hdr), sizeof(hdr));
    PW_CHECK_FALSE(ok3);
    PW_CHECK_EQ(parser3.GetLastError(), PWMFError::FileTruncated);
}

// ============================================================================
// 6. AlignmentViolation_UnalignedWeightOffset
// weight_data_offset set to 97 (not 64-byte aligned) -> returns false, last_error == PWMFError::AlignmentViolation (-5)
// ============================================================================
PW_TEST_CASE(PWMFFuzzSuite, AlignmentViolation_UnalignedWeightOffset) {
    std::vector<uint8_t> buffer(512, 0);
    PWMFHeader* hdr = reinterpret_cast<PWMFHeader*>(buffer.data());
    hdr->magic = PWMF_MAGIC;
    hdr->version_major = PWMF_VERSION_MAJOR;
    hdr->header_size = sizeof(PWMFHeader);
    hdr->alignment = 64;
    hdr->weight_data_offset = 97; // 97 is not divisible by 64
    hdr->weight_data_length = 0;
    hdr->metadata_offset = 96;
    hdr->metadata_length = 0;
    hdr->tensor_table_offset = 96;
    hdr->tensor_table_length = 0;

    PWMFParser parser;
    bool ok = parser.ParseMemory(buffer.data(), buffer.size());
    PW_CHECK_FALSE(ok);
    PW_CHECK_EQ(parser.GetLastError(), PWMFError::AlignmentViolation);
    PW_CHECK_EQ(static_cast<int32_t>(parser.GetLastError()), -5);

    // Test with alignment=64 and offset=63, 65, 127
    for (uint64_t unaligned_offset : {63ULL, 65ULL, 127ULL, 129ULL}) {
        hdr->weight_data_offset = unaligned_offset;
        PW_CHECK_FALSE(parser.ParseMemory(buffer.data(), buffer.size()));
        PW_CHECK_EQ(parser.GetLastError(), PWMFError::AlignmentViolation);
    }
}

// ============================================================================
// 7. OffsetOutOfBounds_MetadataOffsetPastEOF
// metadata_offset = 2000 with buffer size 1000 -> returns false, last_error == PWMFError::OffsetOutOfBounds (-7)
// ============================================================================
PW_TEST_CASE(PWMFFuzzSuite, OffsetOutOfBounds_MetadataOffsetPastEOF) {
    std::vector<uint8_t> buffer(1000, 0);
    PWMFHeader* hdr = reinterpret_cast<PWMFHeader*>(buffer.data());
    hdr->magic = PWMF_MAGIC;
    hdr->version_major = PWMF_VERSION_MAJOR;
    hdr->header_size = sizeof(PWMFHeader);
    hdr->alignment = 64;
    hdr->weight_data_offset = 128; // Aligned
    hdr->weight_data_length = 0;
    hdr->metadata_offset = 2000; // Past 1000 byte buffer bounds
    hdr->metadata_length = 0;
    hdr->tensor_table_offset = 128;
    hdr->tensor_table_length = 0;

    PWMFParser parser;
    bool ok = parser.ParseMemory(buffer.data(), buffer.size());
    PW_CHECK_FALSE(ok);
    PW_CHECK_EQ(parser.GetLastError(), PWMFError::OffsetOutOfBounds);
    PW_CHECK_EQ(static_cast<int32_t>(parser.GetLastError()), -7);
}

// ============================================================================
// 8. OffsetOutOfBounds_MetadataLengthPastEOF
// metadata_offset = 96, metadata_length = 2000 in buffer size 500 -> returns false, last_error == PWMFError::OffsetOutOfBounds (-7)
// ============================================================================
PW_TEST_CASE(PWMFFuzzSuite, OffsetOutOfBounds_MetadataLengthPastEOF) {
    std::vector<uint8_t> buffer(500, 0);
    PWMFHeader* hdr = reinterpret_cast<PWMFHeader*>(buffer.data());
    hdr->magic = PWMF_MAGIC;
    hdr->version_major = PWMF_VERSION_MAJOR;
    hdr->header_size = sizeof(PWMFHeader);
    hdr->alignment = 64;
    hdr->weight_data_offset = 128;
    hdr->weight_data_length = 0;
    hdr->metadata_offset = 96;
    hdr->metadata_length = 2000; // 96 + 2000 = 2096 > 500
    hdr->tensor_table_offset = 128;
    hdr->tensor_table_length = 0;

    PWMFParser parser;
    bool ok = parser.ParseMemory(buffer.data(), buffer.size());
    PW_CHECK_FALSE(ok);
    PW_CHECK_EQ(parser.GetLastError(), PWMFError::OffsetOutOfBounds);
    PW_CHECK_EQ(static_cast<int32_t>(parser.GetLastError()), -7);
}

// ============================================================================
// 9. OffsetOutOfBounds_TensorTableOffsetPastEOF
// tensor_table_offset = 5000 in buffer size 1000 -> returns false, last_error == PWMFError::OffsetOutOfBounds (-7)
// ============================================================================
PW_TEST_CASE(PWMFFuzzSuite, OffsetOutOfBounds_TensorTableOffsetPastEOF) {
    std::vector<uint8_t> buffer(1000, 0);
    PWMFHeader* hdr = reinterpret_cast<PWMFHeader*>(buffer.data());
    hdr->magic = PWMF_MAGIC;
    hdr->version_major = PWMF_VERSION_MAJOR;
    hdr->header_size = sizeof(PWMFHeader);
    hdr->alignment = 64;
    hdr->weight_data_offset = 128;
    hdr->weight_data_length = 0;
    hdr->metadata_offset = 96;
    hdr->metadata_length = 0; // Metadata valid
    hdr->tensor_table_offset = 5000; // Past 1000 bytes
    hdr->tensor_table_length = 0;

    PWMFParser parser;
    bool ok = parser.ParseMemory(buffer.data(), buffer.size());
    PW_CHECK_FALSE(ok);
    PW_CHECK_EQ(parser.GetLastError(), PWMFError::OffsetOutOfBounds);
    PW_CHECK_EQ(static_cast<int32_t>(parser.GetLastError()), -7);
}

// ============================================================================
// 10. InvalidDescriptor_TableLengthMismatch
// num_tensors = 2, but tensor_table_length != 2 * 112 (e.g. 100) -> returns false, last_error == PWMFError::InvalidDescriptor (-11)
// ============================================================================
PW_TEST_CASE(PWMFFuzzSuite, InvalidDescriptor_TableLengthMismatch) {
    std::vector<uint8_t> buffer(1024, 0);
    PWMFHeader* hdr = reinterpret_cast<PWMFHeader*>(buffer.data());
    hdr->magic = PWMF_MAGIC;
    hdr->version_major = PWMF_VERSION_MAJOR;
    hdr->header_size = sizeof(PWMFHeader);
    hdr->alignment = 64;
    hdr->metadata_offset = 96;
    hdr->metadata_length = 0;
    hdr->num_tensors = 2;
    hdr->tensor_table_offset = 128;
    hdr->tensor_table_length = 100; // Expected: 2 * 112 = 224 bytes
    hdr->weight_data_offset = 384;
    hdr->weight_data_length = 0;

    PWMFParser parser;
    bool ok = parser.ParseMemory(buffer.data(), buffer.size());
    PW_CHECK_FALSE(ok);
    PW_CHECK_EQ(parser.GetLastError(), PWMFError::InvalidDescriptor);
    PW_CHECK_EQ(static_cast<int32_t>(parser.GetLastError()), -11);

    // Also test length = 225 (1 byte off)
    hdr->tensor_table_length = 225;
    PW_CHECK_FALSE(parser.ParseMemory(buffer.data(), buffer.size()));
    PW_CHECK_EQ(parser.GetLastError(), PWMFError::InvalidDescriptor);
}

// ============================================================================
// 11. InvalidDescriptor_TensorRankZero
// desc.ndims = 0 -> returns false, last_error == PWMFError::InvalidDescriptor (-11)
// ============================================================================
PW_TEST_CASE(PWMFFuzzSuite, InvalidDescriptor_TensorRankZero) {
    std::vector<uint8_t> blob = CreateMinimalValidPWMFBuffer();
    PWMFHeader* hdr = reinterpret_cast<PWMFHeader*>(blob.data());
    PW_ASSERT_TRUE(hdr->tensor_table_offset + sizeof(PWMFTensorDescriptor) <= blob.size());

    PWMFTensorDescriptor* desc = reinterpret_cast<PWMFTensorDescriptor*>(
        blob.data() + hdr->tensor_table_offset);
    desc->ndims = 0; // Rank zero is forbidden by specification (1..5)

    PWMFParser parser;
    bool ok = parser.ParseMemory(blob.data(), blob.size());
    PW_CHECK_FALSE(ok);
    PW_CHECK_EQ(parser.GetLastError(), PWMFError::InvalidDescriptor);
    PW_CHECK_EQ(static_cast<int32_t>(parser.GetLastError()), -11);
}

// ============================================================================
// 12. InvalidDescriptor_TensorRankTooLarge
// desc.ndims = 6 -> returns false, last_error == PWMFError::InvalidDescriptor (-11)
// ============================================================================
PW_TEST_CASE(PWMFFuzzSuite, InvalidDescriptor_TensorRankTooLarge) {
    std::vector<uint8_t> blob = CreateMinimalValidPWMFBuffer();
    PWMFHeader* hdr = reinterpret_cast<PWMFHeader*>(blob.data());
    PW_ASSERT_TRUE(hdr->tensor_table_offset + sizeof(PWMFTensorDescriptor) <= blob.size());

    PWMFTensorDescriptor* desc = reinterpret_cast<PWMFTensorDescriptor*>(
        blob.data() + hdr->tensor_table_offset);
    desc->ndims = 6; // Rank 6 exceeds maximum supported rank of 5

    PWMFParser parser;
    bool ok = parser.ParseMemory(blob.data(), blob.size());
    PW_CHECK_FALSE(ok);
    PW_CHECK_EQ(parser.GetLastError(), PWMFError::InvalidDescriptor);
    PW_CHECK_EQ(static_cast<int32_t>(parser.GetLastError()), -11);

    // Also test extreme rank (e.g. 255)
    desc->ndims = 255;
    PW_CHECK_FALSE(parser.ParseMemory(blob.data(), blob.size()));
    PW_CHECK_EQ(parser.GetLastError(), PWMFError::InvalidDescriptor);
}

// ============================================================================
// 13. UnsupportedQuant_InvalidQuantTypeEnum
// desc.quant_type = QuantType(8) (or 99) -> returns false, last_error == PWMFError::UnsupportedQuant (-8)
// ============================================================================
PW_TEST_CASE(PWMFFuzzSuite, UnsupportedQuant_InvalidQuantTypeEnum) {
    std::vector<uint8_t> blob = CreateMinimalValidPWMFBuffer();
    PWMFHeader* hdr = reinterpret_cast<PWMFHeader*>(blob.data());
    PWMFTensorDescriptor* desc = reinterpret_cast<PWMFTensorDescriptor*>(
        blob.data() + hdr->tensor_table_offset);

    // Set quant_type to 8 (beyond max QuantType::FP8_E5M2 = 7)
    desc->quant_type = static_cast<QuantType>(8);

    PWMFParser parser;
    bool ok = parser.ParseMemory(blob.data(), blob.size());
    PW_CHECK_FALSE(ok);
    PW_CHECK_EQ(parser.GetLastError(), PWMFError::UnsupportedQuant);
    PW_CHECK_EQ(static_cast<int32_t>(parser.GetLastError()), -8);

    // Test quant_type = 99 and 255
    desc->quant_type = static_cast<QuantType>(99);
    PW_CHECK_FALSE(parser.ParseMemory(blob.data(), blob.size()));
    PW_CHECK_EQ(parser.GetLastError(), PWMFError::UnsupportedQuant);

    desc->quant_type = static_cast<QuantType>(255);
    PW_CHECK_FALSE(parser.ParseMemory(blob.data(), blob.size()));
    PW_CHECK_EQ(parser.GetLastError(), PWMFError::UnsupportedQuant);
}

// ============================================================================
// 14. OffsetOutOfBounds_TensorDataOffsetPastWeightData
// desc.data_offset + desc.data_bytes > weight_data_length -> returns false, last_error == PWMFError::OffsetOutOfBounds (-7)
// ============================================================================
PW_TEST_CASE(PWMFFuzzSuite, OffsetOutOfBounds_TensorDataOffsetPastWeightData) {
    std::vector<uint8_t> blob = CreateMinimalValidPWMFBuffer();
    PWMFHeader* hdr = reinterpret_cast<PWMFHeader*>(blob.data());
    PWMFTensorDescriptor* desc = reinterpret_cast<PWMFTensorDescriptor*>(
        blob.data() + hdr->tensor_table_offset);

    // Offset points beyond the allocated weight data blob
    desc->data_offset = hdr->weight_data_length + 1024;

    PWMFParser parser;
    bool ok = parser.ParseMemory(blob.data(), blob.size());
    PW_CHECK_FALSE(ok);
    PW_CHECK_EQ(parser.GetLastError(), PWMFError::OffsetOutOfBounds);
    PW_CHECK_EQ(static_cast<int32_t>(parser.GetLastError()), -7);

    // Alternatively, data_offset is within range, but data_bytes overflows weight_data_length
    desc->data_offset = 0;
    desc->data_bytes = hdr->weight_data_length + 1;
    PW_CHECK_FALSE(parser.ParseMemory(blob.data(), blob.size()));
    PW_CHECK_EQ(parser.GetLastError(), PWMFError::OffsetOutOfBounds);
}

// ============================================================================
// 15. PayloadTruncated_WeightDataLengthPastEOF
// weight_data_offset + weight_data_length > size -> returns false, last_error == PWMFError::PayloadTruncated (-6)
// ============================================================================
PW_TEST_CASE(PWMFFuzzSuite, PayloadTruncated_WeightDataLengthPastEOF) {
    std::vector<uint8_t> buffer(500, 0);
    PWMFHeader* hdr = reinterpret_cast<PWMFHeader*>(buffer.data());
    hdr->magic = PWMF_MAGIC;
    hdr->version_major = PWMF_VERSION_MAJOR;
    hdr->header_size = sizeof(PWMFHeader);
    hdr->alignment = 64;
    hdr->metadata_offset = 96;
    hdr->metadata_length = 0;
    hdr->num_tensors = 0;
    hdr->tensor_table_offset = 96;
    hdr->tensor_table_length = 0;
    hdr->weight_data_offset = 128; // 64-byte aligned
    hdr->weight_data_length = 1000; // 128 + 1000 = 1128 > 500 byte buffer

    PWMFParser parser;
    bool ok = parser.ParseMemory(buffer.data(), buffer.size());
    PW_CHECK_FALSE(ok);
    PW_CHECK_EQ(parser.GetLastError(), PWMFError::PayloadTruncated);
    PW_CHECK_EQ(static_cast<int32_t>(parser.GetLastError()), -6);
}

// ============================================================================
// 16. ChecksumMismatch_CorruptedPayload
// 1 bit flipped in weight payload with non-zero header CRC32 -> returns false, last_error == PWMFError::ChecksumMismatch (-9)
// ============================================================================
PW_TEST_CASE(PWMFFuzzSuite, ChecksumMismatch_CorruptedPayload) {
    std::vector<uint8_t> blob = CreateMinimalValidPWMFBuffer();
    const PWMFHeader* hdr = reinterpret_cast<const PWMFHeader*>(blob.data());
    PW_ASSERT_NE(hdr->GetCRC32(), 0U);

    // Ensure pristine buffer validates first
    PWMFParser valid_parser;
    PW_ASSERT_TRUE(valid_parser.ParseMemory(blob.data(), blob.size()));
    PW_ASSERT_EQ(valid_parser.GetLastError(), PWMFError::Success);

    // Corrupt a single bit in weight payload
    size_t payload_byte_index = static_cast<size_t>(hdr->weight_data_offset);
    PW_ASSERT_LT(payload_byte_index, blob.size());
    blob[payload_byte_index] ^= 0x01;

    PWMFParser tampered_parser;
    bool ok = tampered_parser.ParseMemory(blob.data(), blob.size());
    PW_CHECK_FALSE(ok);
    PW_CHECK_EQ(tampered_parser.GetLastError(), PWMFError::ChecksumMismatch);
    PW_CHECK_EQ(static_cast<int32_t>(tampered_parser.GetLastError()), -9);
}

// ============================================================================
// 17. ChecksumMismatch_FlippedHeaderCRC
// header CRC field flipped/mismatched -> returns false, last_error == PWMFError::ChecksumMismatch (-9)
// ============================================================================
PW_TEST_CASE(PWMFFuzzSuite, ChecksumMismatch_FlippedHeaderCRC) {
    std::vector<uint8_t> blob = CreateMinimalValidPWMFBuffer();
    PWMFHeader* hdr = reinterpret_cast<PWMFHeader*>(blob.data());
    uint32_t original_crc = hdr->GetCRC32();
    PW_ASSERT_NE(original_crc, 0U);

    // Invert the most significant bit of the stored CRC32
    hdr->SetCRC32(original_crc ^ 0x80000000U);

    PWMFParser parser;
    bool ok = parser.ParseMemory(blob.data(), blob.size());
    PW_CHECK_FALSE(ok);
    PW_CHECK_EQ(parser.GetLastError(), PWMFError::ChecksumMismatch);
    PW_CHECK_EQ(static_cast<int32_t>(parser.GetLastError()), -9);
}

// ============================================================================
// 18. DequantizerFuzz_ExtremeFloats
// - INT4Block32 with denormal scale (0x0400 = 2^-14): dequantizes to finite values without underflow trap or crash.
// - INT4Block32 with +Inf scale (0x7C00) and -Inf scale (0xFC00): dequantizes to +/-Inf without crash.
// - INT4Block32 with NaN scale (0x7E00): dequantizes to NaN without crash.
// - FP8_E4M3 with 0x7F and 0xFF: dequantizes to NaN without crash.
// - FP8_E5M2 with 0x7C (+Inf), 0xFC (-Inf), 0x7D (NaN): dequantizes to +/-Inf and NaN without crash.
// ============================================================================
PW_TEST_CASE(PWMFFuzzSuite, DequantizerFuzz_ExtremeFloats) {
    // ------------------------------------------------------------------------
    // Part A: INT4Block32 with Denormal / Subnormal Scale (0x0400 = 2^-14, 0x0001 = 2^-24)
    // ------------------------------------------------------------------------
    {
        INT4Block32 block{};
        // Fill nibbles: 0x12, 0x34, 0x56, ...
        for (size_t i = 0; i < 16; ++i) {
            block.qs[i] = static_cast<uint8_t>((i & 0x0F) | (((i + 1) & 0x0F) << 4));
        }
        block.scale_fp16 = 0x0400; // 2^-14
        block.bias_fp16 = 0x0000;  // 0.0f

        float out[32];
        std::memset(out, 0, sizeof(out));
        DequantizeINT4Block32(&block, out, true);

        for (size_t i = 0; i < 32; ++i) {
            PW_CHECK_TRUE(std::isfinite(out[i]));
        }

        // Test with subtract_bias = false
        DequantizeINT4Block32(&block, out, false);
        for (size_t i = 0; i < 32; ++i) {
            PW_CHECK_TRUE(std::isfinite(out[i]));
        }

        // Test with smallest non-zero subnormal scale (0x0001 = 2^-24)
        block.scale_fp16 = 0x0001;
        DequantizeINT4Block32(&block, out, true);
        for (size_t i = 0; i < 32; ++i) {
            PW_CHECK_TRUE(std::isfinite(out[i]));
        }
    }

    // ------------------------------------------------------------------------
    // Part B: INT4Block32 with +Inf (0x7C00) and -Inf (0xFC00) Scale
    // ------------------------------------------------------------------------
    {
        INT4Block32 block{};
        // Set all nibbles to non-zero values (e.g. 7)
        std::memset(block.qs, 0x77, sizeof(block.qs));
        block.bias_fp16 = 0x0000; // bias = 0.0f

        // +Infinity scale
        block.scale_fp16 = 0x7C00;
        float out_pos_inf[32];
        DequantizeINT4Block32(&block, out_pos_inf, true);

        for (size_t i = 0; i < 32; ++i) {
            PW_CHECK_TRUE(std::isinf(out_pos_inf[i]));
            PW_CHECK_GT(out_pos_inf[i], 0.0f);
        }

        // -Infinity scale
        block.scale_fp16 = 0xFC00;
        float out_neg_inf[32];
        DequantizeINT4Block32(&block, out_neg_inf, true);

        for (size_t i = 0; i < 32; ++i) {
            PW_CHECK_TRUE(std::isinf(out_neg_inf[i]));
            PW_CHECK_LT(out_neg_inf[i], 0.0f);
        }
    }

    // ------------------------------------------------------------------------
    // Part C: INT4Block32 with NaN (0x7E00) Scale
    // ------------------------------------------------------------------------
    {
        INT4Block32 block{};
        std::memset(block.qs, 0x33, sizeof(block.qs));
        block.scale_fp16 = 0x7E00; // NaN
        block.bias_fp16 = 0x0000;

        float out_nan[32];
        DequantizeINT4Block32(&block, out_nan, true);

        for (size_t i = 0; i < 32; ++i) {
            PW_CHECK_TRUE(std::isnan(out_nan[i]));
        }
    }

    // ------------------------------------------------------------------------
    // Part D: FP8_E4M3 with 0x7F and 0xFF (NaN values)
    // ------------------------------------------------------------------------
    {
        float val_7f = fp8_e4m3_to_fp32(0x7F);
        PW_CHECK_TRUE(std::isnan(val_7f));

        float val_ff = fp8_e4m3_to_fp32(0xFF);
        PW_CHECK_TRUE(std::isnan(val_ff));

        // Test array dequantization kernel
        uint8_t fp8_e4m3_data[4] = {0x7F, 0xFF, 0x38, 0x00}; // 0x38 = 1.0f, 0x00 = 0.0f
        float fp8_e4m3_out[4];
        DequantizeFP8_E4M3(fp8_e4m3_data, 4, fp8_e4m3_out);

        PW_CHECK_TRUE(std::isnan(fp8_e4m3_out[0]));
        PW_CHECK_TRUE(std::isnan(fp8_e4m3_out[1]));
        PW_CHECK_FALSE(std::isnan(fp8_e4m3_out[2]));
        EXPECT_NEAR(fp8_e4m3_out[2], 1.0f, 1e-5f);
        EXPECT_NEAR(fp8_e4m3_out[3], 0.0f, 1e-5f);
    }

    // ------------------------------------------------------------------------
    // Part E: FP8_E5M2 with 0x7C (+Inf), 0xFC (-Inf), 0x7D (NaN)
    // ------------------------------------------------------------------------
    {
        float val_pos_inf = fp8_e5m2_to_fp32(0x7C);
        PW_CHECK_TRUE(std::isinf(val_pos_inf));
        PW_CHECK_GT(val_pos_inf, 0.0f);

        float val_neg_inf = fp8_e5m2_to_fp32(0xFC);
        PW_CHECK_TRUE(std::isinf(val_neg_inf));
        PW_CHECK_LT(val_neg_inf, 0.0f);

        float val_nan = fp8_e5m2_to_fp32(0x7D);
        PW_CHECK_TRUE(std::isnan(val_nan));

        // Also test negative NaN (0xFD)
        float val_neg_nan = fp8_e5m2_to_fp32(0xFD);
        PW_CHECK_TRUE(std::isnan(val_neg_nan));

        // Test array dequantization kernel
        uint8_t fp8_e5m2_data[4] = {0x7C, 0xFC, 0x7D, 0x3C}; // 0x3C = 1.0f
        float fp8_e5m2_out[4];
        DequantizeFP8_E5M2(fp8_e5m2_data, 4, fp8_e5m2_out);

        PW_CHECK_TRUE(std::isinf(fp8_e5m2_out[0]));
        PW_CHECK_GT(fp8_e5m2_out[0], 0.0f);
        PW_CHECK_TRUE(std::isinf(fp8_e5m2_out[1]));
        PW_CHECK_LT(fp8_e5m2_out[1], 0.0f);
        PW_CHECK_TRUE(std::isnan(fp8_e5m2_out[2]));
        EXPECT_NEAR(fp8_e5m2_out[3], 1.0f, 1e-5f);
    }
}

// ============================================================================
// 19. ParserRobustness_NullPointer
// ParseMemory(nullptr, 100) -> returns false, last_error == PWMFError::FileTooSmall (-1), 0 crash.
// ============================================================================
PW_TEST_CASE(PWMFFuzzSuite, ParserRobustness_NullPointer) {
    PWMFParser parser;

    // Null pointer with non-zero size
    bool ok1 = parser.ParseMemory(nullptr, 100);
    PW_CHECK_FALSE(ok1);
    PW_CHECK_EQ(parser.GetLastError(), PWMFError::FileTooSmall);
    PW_CHECK_EQ(static_cast<int32_t>(parser.GetLastError()), -1);

    // Null pointer with zero size
    bool ok2 = parser.ParseMemory(nullptr, 0);
    PW_CHECK_FALSE(ok2);
    PW_CHECK_EQ(parser.GetLastError(), PWMFError::FileTooSmall);

    // Null pointer with huge size
    bool ok3 = parser.ParseMemory(nullptr, 1024 * 1024);
    PW_CHECK_FALSE(ok3);
    PW_CHECK_EQ(parser.GetLastError(), PWMFError::FileTooSmall);
}

// ============================================================================
// 20. FileIO_NonexistentFile
// Open("/nonexistent/file/path/does_not_exist.pwmf") -> returns false, last_error == PWMFError::IOError (-10), 0 crash.
// ============================================================================
PW_TEST_CASE(PWMFFuzzSuite, FileIO_NonexistentFile) {
    PWMFParser parser;
    const std::string missing_path = "/nonexistent/file/path/does_not_exist_xyz_12345.pwmf";

    bool ok = parser.Open(missing_path);
    PW_CHECK_FALSE(ok);
    PW_CHECK_EQ(parser.GetLastError(), PWMFError::IOError);
    PW_CHECK_EQ(static_cast<int32_t>(parser.GetLastError()), -10);
    PW_CHECK_FALSE(parser.IsLoaded());
}

// ============================================================================
// 21. DequantizerINT4_BatchExtremeFloats
// Exercises DequantizeINT4Block32Batch across ARM NEON SIMD vector pathways
// with mixed denormals, +Inf, -Inf, and NaNs.
// ============================================================================
PW_TEST_CASE(PWMFFuzzSuite, DequantizerINT4_BatchExtremeFloats) {
    constexpr size_t num_blocks = 4;
    INT4Block32 blocks[num_blocks]{};

    // Block 0: Normal finite
    std::memset(blocks[0].qs, 0x22, sizeof(blocks[0].qs));
    blocks[0].scale_fp16 = fp32_to_fp16(0.5f);
    blocks[0].bias_fp16 = 0x0000;

    // Block 1: Denormal scale (0x0400 = 2^-14)
    std::memset(blocks[1].qs, 0x55, sizeof(blocks[1].qs));
    blocks[1].scale_fp16 = 0x0400;
    blocks[1].bias_fp16 = 0x0000;

    // Block 2: Positive Infinity (0x7C00)
    std::memset(blocks[2].qs, 0x88, sizeof(blocks[2].qs));
    blocks[2].scale_fp16 = 0x7C00;
    blocks[2].bias_fp16 = 0x0000;

    // Block 3: Quiet NaN (0x7E00)
    std::memset(blocks[3].qs, 0x44, sizeof(blocks[3].qs));
    blocks[3].scale_fp16 = 0x7E00;
    blocks[3].bias_fp16 = 0x0000;

    float batch_out[num_blocks * 32];
    std::memset(batch_out, 0, sizeof(batch_out));

    // Execute batch dequantization (SIMD / NEON)
    DequantizeINT4Block32Batch(blocks, num_blocks, batch_out, true);

    // Block 0: finite
    for (size_t i = 0; i < 32; ++i) {
        PW_CHECK_TRUE(std::isfinite(batch_out[i]));
    }
    // Block 1: finite denormal
    for (size_t i = 32; i < 64; ++i) {
        PW_CHECK_TRUE(std::isfinite(batch_out[i]));
    }
    // Block 2: +Infinity
    for (size_t i = 64; i < 96; ++i) {
        PW_CHECK_TRUE(std::isinf(batch_out[i]));
        PW_CHECK_GT(batch_out[i], 0.0f);
    }
    // Block 3: NaN
    for (size_t i = 96; i < 128; ++i) {
        PW_CHECK_TRUE(std::isnan(batch_out[i]));
    }
}

// ============================================================================
// 22. ErrorCodeToString_AllVariants
// Verifies PWMFErrorToString returns non-empty descriptive messages for all 12 enums
// ============================================================================
PW_TEST_CASE(PWMFFuzzSuite, ErrorCodeToString_AllVariants) {
    const PWMFError all_errors[] = {
        PWMFError::Success,
        PWMFError::FileTooSmall,
        PWMFError::FileTruncated,
        PWMFError::InvalidMagic,
        PWMFError::UnsupportedVersion,
        PWMFError::AlignmentViolation,
        PWMFError::PayloadTruncated,
        PWMFError::OffsetOutOfBounds,
        PWMFError::UnsupportedQuant,
        PWMFError::ChecksumMismatch,
        PWMFError::IOError,
        PWMFError::InvalidDescriptor
    };

    for (PWMFError err : all_errors) {
        const char* msg = PWMFErrorToString(err);
        PW_ASSERT_TRUE(msg != nullptr);
        PW_CHECK_GT(std::strlen(msg), 0UL);
        PW_CHECK_NE(std::string(msg), "Unknown error");
    }

    // Verify unknown error fallback
    const char* unknown_msg = PWMFErrorToString(static_cast<PWMFError>(999));
    PW_CHECK_EQ(std::string(unknown_msg), "Unknown error");
}

// ============================================================================
// 23. OffsetOutOfBounds_WrappingMetadataOffset
// metadata_offset = UINT64_MAX - 10, metadata_length = 20 -> returns false, PWMFError::OffsetOutOfBounds
// ============================================================================
PW_TEST_CASE(PWMFFuzzSuite, OffsetOutOfBounds_WrappingMetadataOffset) {
    std::vector<uint8_t> buffer(512, 0);
    PWMFHeader* hdr = reinterpret_cast<PWMFHeader*>(buffer.data());
    hdr->magic = PWMF_MAGIC;
    hdr->version_major = PWMF_VERSION_MAJOR;
    hdr->version_minor = PWMF_VERSION_MINOR;
    hdr->header_size = sizeof(PWMFHeader);
    hdr->alignment = 64;
    hdr->metadata_offset = std::numeric_limits<uint64_t>::max() - 10;
    hdr->metadata_length = 20;
    hdr->tensor_table_offset = 128;
    hdr->tensor_table_length = 0;
    hdr->weight_data_offset = 128;
    hdr->weight_data_length = 0;

    PWMFParser parser;
    bool ok = parser.ParseMemory(buffer.data(), buffer.size());
    PW_CHECK_FALSE(ok);
    PW_CHECK_EQ(parser.GetLastError(), PWMFError::OffsetOutOfBounds);
    PW_CHECK_EQ(static_cast<int32_t>(parser.GetLastError()), -7);
}

// ============================================================================
// 24. OffsetOutOfBounds_WrappingTensorTableOffset
// tensor_table_offset = UINT64_MAX - 50, tensor_table_length = 112 -> returns false, PWMFError::OffsetOutOfBounds
// ============================================================================
PW_TEST_CASE(PWMFFuzzSuite, OffsetOutOfBounds_WrappingTensorTableOffset) {
    std::vector<uint8_t> buffer(512, 0);
    PWMFHeader* hdr = reinterpret_cast<PWMFHeader*>(buffer.data());
    hdr->magic = PWMF_MAGIC;
    hdr->version_major = PWMF_VERSION_MAJOR;
    hdr->version_minor = PWMF_VERSION_MINOR;
    hdr->header_size = sizeof(PWMFHeader);
    hdr->alignment = 64;
    hdr->metadata_offset = 96;
    hdr->metadata_length = 0;
    hdr->num_tensors = 1;
    hdr->tensor_table_offset = std::numeric_limits<uint64_t>::max() - 50;
    hdr->tensor_table_length = 112;
    hdr->weight_data_offset = 256;
    hdr->weight_data_length = 0;

    PWMFParser parser;
    bool ok = parser.ParseMemory(buffer.data(), buffer.size());
    PW_CHECK_FALSE(ok);
    PW_CHECK_EQ(parser.GetLastError(), PWMFError::OffsetOutOfBounds);
    PW_CHECK_EQ(static_cast<int32_t>(parser.GetLastError()), -7);
}

// ============================================================================
// 25. PayloadTruncated_WrappingWeightDataOffset
// weight_data_offset = 0xFFFFFFFFFFFFFFC0ULL, weight_data_length = 64 -> returns false, PWMFError::PayloadTruncated
// ============================================================================
PW_TEST_CASE(PWMFFuzzSuite, PayloadTruncated_WrappingWeightDataOffset) {
    std::vector<uint8_t> buffer(512, 0);
    PWMFHeader* hdr = reinterpret_cast<PWMFHeader*>(buffer.data());
    hdr->magic = PWMF_MAGIC;
    hdr->version_major = PWMF_VERSION_MAJOR;
    hdr->version_minor = PWMF_VERSION_MINOR;
    hdr->header_size = sizeof(PWMFHeader);
    hdr->alignment = 64;
    hdr->metadata_offset = 96;
    hdr->metadata_length = 0;
    hdr->num_tensors = 0;
    hdr->tensor_table_offset = 96;
    hdr->tensor_table_length = 0;
    hdr->weight_data_offset = 0xFFFFFFFFFFFFFFC0ULL; // Aligned to 64 bytes
    hdr->weight_data_length = 64;

    PWMFParser parser;
    bool ok = parser.ParseMemory(buffer.data(), buffer.size());
    PW_CHECK_FALSE(ok);
    PW_CHECK_EQ(parser.GetLastError(), PWMFError::PayloadTruncated);
    PW_CHECK_EQ(static_cast<int32_t>(parser.GetLastError()), -6);
}

// ============================================================================
// 26. OffsetOutOfBounds_WrappingTensorDataOffset
// desc.data_offset = UINT64_MAX - 10, desc.data_bytes = 20 -> returns false, PWMFError::OffsetOutOfBounds
// ============================================================================
PW_TEST_CASE(PWMFFuzzSuite, OffsetOutOfBounds_WrappingTensorDataOffset) {
    std::vector<uint8_t> blob = CreateMinimalValidPWMFBuffer();
    PWMFHeader* hdr = reinterpret_cast<PWMFHeader*>(blob.data());
    PW_ASSERT_TRUE(hdr->tensor_table_offset + sizeof(PWMFTensorDescriptor) <= blob.size());

    PWMFTensorDescriptor* desc = reinterpret_cast<PWMFTensorDescriptor*>(
        blob.data() + hdr->tensor_table_offset);

    // Wrapping descriptor offset (UINT64_MAX - 10 + 20 wraps around mod 2^64 to 9 < weight_data_length)
    desc->data_offset = std::numeric_limits<uint64_t>::max() - 10;
    desc->data_bytes = 20;

    PWMFParser parser;
    bool ok = parser.ParseMemory(blob.data(), blob.size());
    PW_CHECK_FALSE(ok);
    PW_CHECK_EQ(parser.GetLastError(), PWMFError::OffsetOutOfBounds);
    PW_CHECK_EQ(static_cast<int32_t>(parser.GetLastError()), -7);
}

// ============================================================================
// 27. UnsupportedVersion_MinorVersionTooHigh
// version_minor = 99 -> returns false, PWMFError::UnsupportedVersion
// ============================================================================
PW_TEST_CASE(PWMFFuzzSuite, UnsupportedVersion_MinorVersionTooHigh) {
    std::vector<uint8_t> blob = CreateMinimalValidPWMFBuffer();
    PWMFHeader* hdr = reinterpret_cast<PWMFHeader*>(blob.data());
    hdr->version_minor = 99; // Exceeds PWMF_VERSION_MINOR (0)

    PWMFParser parser;
    bool ok = parser.ParseMemory(blob.data(), blob.size());
    PW_CHECK_FALSE(ok);
    PW_CHECK_EQ(parser.GetLastError(), PWMFError::UnsupportedVersion);
    PW_CHECK_EQ(static_cast<int32_t>(parser.GetLastError()), -4);

    // Test minor version 1
    hdr->version_minor = 1;
    PW_CHECK_FALSE(parser.ParseMemory(blob.data(), blob.size()));
    PW_CHECK_EQ(parser.GetLastError(), PWMFError::UnsupportedVersion);
}

// ============================================================================
// 28. InvalidDescriptor_CorruptedHeaderReservedPadding
// non-zero byte in header.reserved[4..27] -> returns false, PWMFError::InvalidDescriptor
// ============================================================================
PW_TEST_CASE(PWMFFuzzSuite, InvalidDescriptor_CorruptedHeaderReservedPadding) {
    std::vector<uint8_t> blob = CreateMinimalValidPWMFBuffer();
    PWMFHeader* hdr = reinterpret_cast<PWMFHeader*>(blob.data());

    // Verify pristine buffer loads
    PWMFParser pristine_parser;
    PW_ASSERT_TRUE(pristine_parser.ParseMemory(blob.data(), blob.size()));

    // Test each byte in reserved[4..27] individually
    for (size_t k = 4; k < sizeof(hdr->reserved); ++k) {
        auto mutated = blob;
        PWMFHeader* mut_hdr = reinterpret_cast<PWMFHeader*>(mutated.data());
        mut_hdr->reserved[k] = 0x5A; // non-zero padding byte

        PWMFParser parser;
        bool ok = parser.ParseMemory(mutated.data(), mutated.size());
        PW_CHECK_FALSE(ok);
        PW_CHECK_EQ(parser.GetLastError(), PWMFError::InvalidDescriptor);
        PW_CHECK_EQ(static_cast<int32_t>(parser.GetLastError()), -11);
    }
}

TEST_RUNNER_MAIN()

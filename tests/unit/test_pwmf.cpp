#include "test_runner.h"
#include "playworld/pwmf_format.h"
#include "playworld/tensor.h"
#include <cstring>
#include <vector>
#include <string>

using namespace playworld;

TEST(PWMFSuite, HeaderByteSizeStaticAssert) {
    // Authoritative requirement: WINNING_PROJECT_PLAN §3.1.1 & env_and_test_infra.md §1.3
    ASSERT_EQ(sizeof(PWMFHeader), 96);
    EXPECT_EQ(offsetof(PWMFHeader, magic), 0x00);
    EXPECT_EQ(offsetof(PWMFHeader, version_major), 0x04);
    EXPECT_EQ(offsetof(PWMFHeader, version_minor), 0x06);
    EXPECT_EQ(offsetof(PWMFHeader, header_size), 0x08);
    EXPECT_EQ(offsetof(PWMFHeader, num_tensors), 0x0C);
    EXPECT_EQ(offsetof(PWMFHeader, metadata_offset), 0x10);
    EXPECT_EQ(offsetof(PWMFHeader, metadata_length), 0x18);
    EXPECT_EQ(offsetof(PWMFHeader, tensor_table_offset), 0x20);
    EXPECT_EQ(offsetof(PWMFHeader, tensor_table_length), 0x28);
    EXPECT_EQ(offsetof(PWMFHeader, weight_data_offset), 0x30);
    EXPECT_EQ(offsetof(PWMFHeader, weight_data_length), 0x38);
    EXPECT_EQ(offsetof(PWMFHeader, alignment), 0x40);
    EXPECT_EQ(offsetof(PWMFHeader, reserved), 0x44);
}

TEST(PWMFSuite, TensorDescriptorByteSizeStaticAssert) {
    // Authoritative requirement: WINNING_PROJECT_PLAN §3.1.1
    ASSERT_EQ(sizeof(PWMFTensorDescriptor), 112);
    EXPECT_EQ(offsetof(PWMFTensorDescriptor, name), 0x00);
    EXPECT_EQ(offsetof(PWMFTensorDescriptor, ndims), 0x40);
    EXPECT_EQ(offsetof(PWMFTensorDescriptor, shape), 0x44);
    EXPECT_EQ(offsetof(PWMFTensorDescriptor, quant_type), 0x58);
    EXPECT_EQ(offsetof(PWMFTensorDescriptor, data_offset), 0x5C);
    EXPECT_EQ(offsetof(PWMFTensorDescriptor, data_bytes), 0x64);
    EXPECT_EQ(offsetof(PWMFTensorDescriptor, global_scale), 0x6C);
}

TEST(PWMFSuite, INT4BlockByteSizesStaticAssert) {
    // Authoritative requirement: WINNING_PROJECT_PLAN §3.1.1
    ASSERT_EQ(sizeof(INT4Block32), 20);
    EXPECT_EQ(offsetof(INT4Block32, qs), 0x00);
    EXPECT_EQ(offsetof(INT4Block32, scale_fp16), 0x10);
    EXPECT_EQ(offsetof(INT4Block32, bias_fp16), 0x12);

    ASSERT_EQ(sizeof(INT4Block64), 36);
}

TEST(PWMFSuite, CRC32_StandardVectorValidation) {
    // Authoritative source: IEEE 802.3 standard test vector
    // Input: "123456789" (9 bytes) -> Expected CRC32: 0xCBF43926
    const char* test_vector = "123456789";
    uint32_t crc = ComputeCRC32(reinterpret_cast<const uint8_t*>(test_vector), 9);
    EXPECT_EQ(crc, 0xCBF43926U);

    // Empty buffer check
    EXPECT_EQ(ComputeCRC32(nullptr, 0), 0x00000000U);
}

TEST(PWMFSuite, CRC32_BitFlipSensitivity) {
    // Bit-flip tamper test: flipping any bit in a buffer must change the CRC
    std::vector<uint8_t> buffer(256, 0xAA);
    uint32_t original_crc = ComputeCRC32(buffer.data(), buffer.size());

    buffer[128] ^= 0x01; // flip 1 bit
    uint32_t tampered_crc = ComputeCRC32(buffer.data(), buffer.size());
    EXPECT_NE(original_crc, tampered_crc);
}

TEST(PWMFSuite, CorruptMagicByteRejection) {
    // Authoritative source: env_and_test_infra.md §3.1.1
    PWMFHeader bad_header{};
    bad_header.magic = 0xDEADBEEF; // Bad magic instead of 0x464D5750 ("PWMF")
    bad_header.version_major = 1;
    bad_header.header_size = 96;

    PWMFParser parser;
    bool success = parser.ParseMemory(reinterpret_cast<const uint8_t*>(&bad_header), sizeof(bad_header));
    EXPECT_FALSE(success);
    EXPECT_EQ(parser.GetLastError(), PWMFError::InvalidMagic);
}

TEST(PWMFSuite, TruncatedHeaderRejection) {
    // Providing less than 96 bytes must be rejected as FileTooSmall
    uint8_t short_buffer[48];
    std::memset(short_buffer, 0, sizeof(short_buffer));

    PWMFParser parser;
    bool success = parser.ParseMemory(short_buffer, sizeof(short_buffer));
    EXPECT_FALSE(success);
    EXPECT_EQ(parser.GetLastError(), PWMFError::FileTooSmall);
}

TEST(PWMFSuite, UnsupportedVersionRejection) {
    PWMFHeader bad_version_header{};
    bad_version_header.magic = PWMF_MAGIC;
    bad_version_header.version_major = 99; // Unsupported major version
    bad_version_header.header_size = 96;

    PWMFParser parser;
    bool success = parser.ParseMemory(reinterpret_cast<const uint8_t*>(&bad_version_header), sizeof(bad_version_header));
    EXPECT_FALSE(success);
    EXPECT_EQ(parser.GetLastError(), PWMFError::UnsupportedVersion);
}

TEST(PWMFSuite, SerializationRoundtripSingleTensor) {
    PWMFWriter writer;
    writer.SetMetadata("{\"model_name\": \"test_world\"}");

    std::vector<uint8_t> dummy_payload(1024 * sizeof(INT4Block32), 0x55);
    uint32_t shape[5] = {1, 1024, 32, 0, 0};
    writer.AddTensor("dit.block0.attn_qkv", 3, shape,
                     QuantType::INT4_BLOCK32, dummy_payload.data(), dummy_payload.size(), 1.0f);

    std::vector<uint8_t> blob = writer.Serialize();
    ASSERT_FALSE(blob.empty());

    PWMFParser parser;
    bool parsed = parser.ParseMemory(blob.data(), blob.size());
    ASSERT_TRUE(parsed);
    EXPECT_EQ(parser.GetLastError(), PWMFError::Success);
    EXPECT_EQ(parser.GetNumTensors(), 1);

    const PWMFTensorDescriptor* desc = parser.GetTensorDescriptor(0);
    ASSERT_TRUE(desc != nullptr);
    EXPECT_EQ(std::string(desc->name), "dit.block0.attn_qkv");
    EXPECT_EQ(desc->ndims, 3);
    EXPECT_EQ(desc->shape[0], 1);
    EXPECT_EQ(desc->shape[1], 1024);
    EXPECT_EQ(desc->shape[2], 32);
    EXPECT_EQ(desc->quant_type, QuantType::INT4_BLOCK32);
    EXPECT_EQ(desc->data_bytes, dummy_payload.size());

    const uint8_t* parsed_data = parser.GetTensorData(0);
    ASSERT_TRUE(parsed_data != nullptr);
    EXPECT_EQ(std::memcmp(parsed_data, dummy_payload.data(), dummy_payload.size()), 0);

    // Verify CRC32 verification passes
    EXPECT_TRUE(parser.VerifyCRC32());
}

TEST(PWMFSuite, SerializationRoundtripMultiTensor) {
    PWMFWriter writer;

    // Tensor 1: FP32 embedding [1536, 32]
    std::vector<float> emb_data(1536 * 32, 1.0f);
    uint32_t shape1[5] = {1536, 32, 0, 0, 0};
    writer.AddTensor("embedding.weight", 2, shape1, QuantType::FP32,
                     emb_data.data(), emb_data.size() * sizeof(float));

    // Tensor 2: INT8 symmetric mlp [1536, 6144]
    std::vector<int8_t> int8_data(1536 * 6144, 42);
    uint32_t shape2[5] = {1536, 6144, 0, 0, 0};
    writer.AddTensor("dit.block0.mlp.fc1", 2, shape2, QuantType::INT8_SYMM,
                     int8_data.data(), int8_data.size(), 0.05f);

    // Tensor 3: INT4 block32 attention [1536, 4608]
    std::vector<uint8_t> int4_data(1536 * (4608 / 32) * sizeof(INT4Block32), 0x77);
    uint32_t shape3[5] = {1536, 4608, 0, 0, 0};
    writer.AddTensor("dit.block0.attn_qkv", 2, shape3, QuantType::INT4_BLOCK32,
                     int4_data.data(), int4_data.size(), 1.0f);

    std::vector<uint8_t> blob = writer.Serialize();
    ASSERT_FALSE(blob.empty());

    PWMFParser parser;
    ASSERT_TRUE(parser.ParseMemory(blob.data(), blob.size()));
    EXPECT_EQ(parser.GetNumTensors(), 3);

    // Query by name
    const PWMFTensorDescriptor* d1 = parser.FindTensorDescriptor("embedding.weight");
    ASSERT_TRUE(d1 != nullptr);
    EXPECT_EQ(d1->quant_type, QuantType::FP32);

    const PWMFTensorDescriptor* d2 = parser.FindTensorDescriptor("dit.block0.mlp.fc1");
    ASSERT_TRUE(d2 != nullptr);
    EXPECT_EQ(d2->quant_type, QuantType::INT8_SYMM);
    EXPECT_NEAR(d2->global_scale, 0.05f, 1e-6f);

    const PWMFTensorDescriptor* d3 = parser.FindTensorDescriptor("dit.block0.attn_qkv");
    ASSERT_TRUE(d3 != nullptr);
    EXPECT_EQ(d3->quant_type, QuantType::INT4_BLOCK32);

    // Check payload identity
    const uint8_t* p1 = parser.GetTensorData("embedding.weight");
    ASSERT_TRUE(p1 != nullptr);
    EXPECT_EQ(std::memcmp(p1, emb_data.data(), emb_data.size() * sizeof(float)), 0);

    const uint8_t* p2 = parser.GetTensorData("dit.block0.mlp.fc1");
    ASSERT_TRUE(p2 != nullptr);
    EXPECT_EQ(std::memcmp(p2, int8_data.data(), int8_data.size()), 0);

    EXPECT_TRUE(parser.VerifyCRC32());
}

TEST(PWMFSuite, TamperedPayloadRejectionByCRC) {
    PWMFWriter writer;
    std::vector<uint8_t> dummy(64, 0x11);
    uint32_t shape[5] = {64, 0, 0, 0, 0};
    writer.AddTensor("dummy", 1, shape, QuantType::INT8_SYMM, dummy.data(), dummy.size());

    std::vector<uint8_t> blob = writer.Serialize();
    ASSERT_FALSE(blob.empty());

    // Tamper with the last byte
    blob.back() ^= 0xFF;

    PWMFParser parser;
    bool success = parser.ParseMemory(blob.data(), blob.size());
    // Must reject due to ChecksumMismatch
    EXPECT_FALSE(success);
    EXPECT_EQ(parser.GetLastError(), PWMFError::ChecksumMismatch);
}

TEST_RUNNER_MAIN()

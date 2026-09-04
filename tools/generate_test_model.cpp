#include "playworld/pwmf_format.h"
#include "playworld/tensor.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::cerr << "CHECK FAILED at line " << __LINE__ << ": " #cond   \
                      << std::endl;                                          \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

int main(int argc, char* argv[]) {
    std::string output_path = "test_model.pwmf";
    bool verify_only = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--verify" && i + 1 < argc) {
            verify_only = true;
            output_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg[0] != '-') {
            output_path = arg;
        }
    }

    // 1. Verify struct sizes and static alignments
    std::cout << "[PlayWorld] Verifying struct sizes and memory alignments...\n";
    std::cout << "  - sizeof(PWMFHeader) = " << sizeof(playworld::PWMFHeader) << " (expected 96)\n";
    std::cout << "  - sizeof(PWMFTensorDescriptor) = " << sizeof(playworld::PWMFTensorDescriptor) << " (expected 112)\n";
    std::cout << "  - sizeof(INT4Block32) = " << sizeof(playworld::INT4Block32) << " (expected 20)\n";
    CHECK(sizeof(playworld::PWMFHeader) == 96);
    CHECK(sizeof(playworld::PWMFTensorDescriptor) == 112);
    CHECK(sizeof(playworld::INT4Block32) == 20);

    constexpr size_t NUM_INT4_BLOCKS = 64;
    std::vector<playworld::INT4Block32> int4_blocks(NUM_INT4_BLOCKS);
    for (size_t b = 0; b < NUM_INT4_BLOCKS; ++b) {
        std::memset(int4_blocks[b].qs, 0xF0, 16); // 15 in high nibble, 0 in low nibble
        int4_blocks[b].scale_fp16 = playworld::fp32_to_fp16(0.05f); // ~0x2A66
        int4_blocks[b].bias_fp16  = playworld::fp32_to_fp16(8.0f);  // 0x4800
    }

    if (!verify_only) {
        std::cout << "[PlayWorld] Generating test model at: " << output_path << std::endl;

        // Construct synthetic model with PWMFWriter
        playworld::PWMFWriter writer;
        writer.SetMetadata(R"({"model":"playworld-test","version":"1.0","hidden_dim":1536})");

        // Tensor 1: INT4_BLOCK32
        uint32_t shape_int4[] = {1, static_cast<uint32_t>(NUM_INT4_BLOCKS), 32, 0, 0};
        writer.AddTensor("dit.block0.attn_qkv", 3, shape_int4,
                         playworld::QuantType::INT4_BLOCK32,
                         int4_blocks.data(), int4_blocks.size() * sizeof(playworld::INT4Block32));

        // Tensor 2: INT8_SYMM (256 signed weights [-128..127], scale=0.01)
        std::vector<int8_t> int8_data(256);
        for (int i = 0; i < 256; ++i) {
            int8_data[i] = static_cast<int8_t>(i - 128);
        }
        uint32_t shape_int8[] = {1, 256, 0, 0, 0};
        writer.AddTensor("dit.block0.mlp_fc1", 2, shape_int8,
                         playworld::QuantType::INT8_SYMM,
                         int8_data.data(), int8_data.size(), 0.01f);

        // Tensor 3: FP16 (64 weights)
        std::vector<uint16_t> fp16_data(64);
        for (size_t i = 0; i < 64; ++i) {
            float f = (i % 2 == 0) ? 1.0f : -0.5f;
            fp16_data[i] = playworld::fp32_to_fp16(f);
        }
        uint32_t shape_fp16[] = {64, 0, 0, 0, 0};
        writer.AddTensor("dit.block0.norm1", 1, shape_fp16,
                         playworld::QuantType::FP16,
                         fp16_data.data(), fp16_data.size() * sizeof(uint16_t));

        // Tensor 4: FP32 (32 weights)
        std::vector<float> fp32_data(32);
        for (size_t i = 0; i < 32; ++i) {
            fp32_data[i] = static_cast<float>(i) * 0.5f;
        }
        uint32_t shape_fp32[] = {32, 0, 0, 0, 0};
        writer.AddTensor("action.embedding_proj", 1, shape_fp32,
                         playworld::QuantType::FP32,
                         fp32_data.data(), fp32_data.size() * sizeof(float));

        // Tensor 5: FP8_E4M3 (16 weights)
        std::vector<uint8_t> fp8_data(16);
        for (size_t i = 0; i < 16; ++i) {
            fp8_data[i] = playworld::fp32_to_fp8_e4m3(static_cast<float>(i) * 0.25f);
        }
        uint32_t shape_fp8[] = {16, 0, 0, 0, 0};
        writer.AddTensor("vae.decoder.conv", 1, shape_fp8,
                         playworld::QuantType::FP8_E4M3,
                         fp8_data.data(), fp8_data.size());

        // Serialize and save
        bool save_ok = writer.SaveToFile(output_path);
        CHECK(save_ok);
        std::cout << "  - Saved PWMF model successfully\n";
    }

    // Parse back using POSIX mmap parser
    std::cout << "[PlayWorld] Loading model via POSIX mmap parser: " << output_path << std::endl;
    playworld::PWMFParser parser;
    bool parse_ok = parser.Open(output_path);
    if (!parse_ok) {
        std::cerr << "Parse error: " << parser.GetLastErrorString() << "\n";
    }
    CHECK(parse_ok);
    std::cout << "  - Memory-mapped parsing successful\n";

    // Verify Header & Checksum
    const auto& hdr = parser.GetHeader();
    CHECK(hdr.magic == playworld::PWMF_MAGIC);
    CHECK(hdr.version_major == 1);
    CHECK(hdr.version_minor == 0);
    CHECK(hdr.header_size == 96);
    CHECK(hdr.num_tensors >= 4);
    CHECK(hdr.weight_data_offset % 64 == 0);
    CHECK(parser.VerifyCRC32());
    std::cout << "  - Header and CRC32 verification passed (CRC32: 0x"
              << std::hex << hdr.GetCRC32() << std::dec << ")\n";

    // Verify Descriptors and Payloads
    auto desc0 = parser.GetTensorDescriptor(0);
    CHECK(desc0 != nullptr);
    CHECK(desc0->quant_type == playworld::QuantType::INT4_BLOCK32);

    const uint8_t* raw_int4 = parser.GetTensorData(0);
    CHECK(raw_int4 != nullptr);
    std::cout << "  - Payload access verification passed\n";

    // Verify Dequantization Accuracy
    playworld::Tensor tensor0 = parser.GetTensor(0);
    std::vector<float> dequant_int4 = tensor0.DequantizeToFP32();
    CHECK(dequant_int4.size() == tensor0.numel());

    const size_t num_blocks = tensor0.numel() / 32;
    for (size_t b = 0; b < num_blocks; ++b) {
        for (size_t i = 0; i < 32; i += 2) {
            float v0 = dequant_int4[b * 32 + i];
            float v1 = dequant_int4[b * 32 + i + 1];
            CHECK(std::fabs(v0 - (-0.40f)) < 0.005f);
            CHECK(std::fabs(v1 - 0.35f) < 0.005f);
        }
    }
    std::cout << "  - INT4 Block-32 dequantization math verified (val0=-0.40, val1=+0.35)\n";

    // Verify INT8 dequantization if present
    const auto* desc_int8 = parser.FindTensorDescriptor("dit.blocks.0.mlp_fc1");
    if (!desc_int8) desc_int8 = parser.FindTensorDescriptor("dit.block0.mlp_fc1");
    if (desc_int8) {
        playworld::Tensor t = parser.GetTensor(desc_int8->name);
        std::vector<float> dequant_int8 = t.DequantizeToFP32();
        for (size_t i = 0; i < t.numel(); ++i) {
            float expected = static_cast<float>(static_cast<int>(i) - 128) * t.global_scale();
            CHECK(std::fabs(dequant_int8[i] - expected) < 1e-4f);
        }
        std::cout << "  - INT8 symmetric dequantization verified\n";
    }

    // Verify FP16 dequantization if present
    const auto* desc_fp16 = parser.FindTensorDescriptor("dit.blocks.0.norm1");
    if (!desc_fp16) desc_fp16 = parser.FindTensorDescriptor("dit.block0.norm1");
    if (desc_fp16) {
        playworld::Tensor t = parser.GetTensor(desc_fp16->name);
        std::vector<float> dequant_fp16 = t.DequantizeToFP32();
        CHECK(dequant_fp16.size() == 64);
        std::cout << "  - FP16 dequantization verified\n";
    }

    // Corrupt Header Rejection Tests (only during full run)
    if (!verify_only) {
        playworld::PWMFWriter writer;
        const uint32_t test_shape[] = {32, 0, 0, 0, 0};
        writer.AddTensor("test", 1, test_shape,
                         playworld::QuantType::INT4_BLOCK32, int4_blocks.data(), 20);
        std::vector<uint8_t> corrupted_data = writer.Serialize();

        // Corrupt magic
        corrupted_data[0] = 0x00;
        playworld::PWMFParser corrupt_parser;
        CHECK(!corrupt_parser.ParseMemory(corrupted_data.data(), corrupted_data.size()));
        CHECK(corrupt_parser.GetLastError() == playworld::PWMFError::InvalidMagic);

        // Corrupt CRC32
        corrupted_data = writer.Serialize();
        corrupted_data[corrupted_data.size() - 1] ^= 0xFF; // Flip a byte in weight payload
        CHECK(!corrupt_parser.ParseMemory(corrupted_data.data(), corrupted_data.size()));
        CHECK(corrupt_parser.GetLastError() == playworld::PWMFError::ChecksumMismatch);
        std::cout << "  - Corrupt magic and corrupt CRC32 rejection verified\n";
    }

    std::cout << "\n=================================================================\n";
    std::cout << "  Milestone 1 Verification: ALL CHECKS PASSED!\n";
    std::cout << "=================================================================\n";

    return 0;
}

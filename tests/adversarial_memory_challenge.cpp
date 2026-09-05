#include "test_runner.h"
#include "playworld/pwmf_format.h"
#include "playworld/tensor.h"
#include "playworld/ring_buffer.h"
#include "playworld/voxel_grid.h"
#include "playworld/scheduler.h"

#include <cstring>
#include <vector>
#include <string>
#include <random>

using namespace playworld;

// ============================================================================
// PART 1: Misaligned Reference Binding & Packed Struct Access Challenge
// ============================================================================

TEST(AdversarialMemoryChallenge, MisalignedPackedStructFieldAssertions) {
    // Allocate raw byte buffer and construct packed structs at every possible byte offset (0..7)
    alignas(64) uint8_t raw_buffer[1024];
    std::memset(raw_buffer, 0, sizeof(raw_buffer));

    for (size_t offset = 0; offset < 8; ++offset) {
        // 1. PWMFTensorDescriptor at misaligned offset
        auto* desc = reinterpret_cast<PWMFTensorDescriptor*>(raw_buffer + offset);
        desc->ndims = 3;
        desc->shape[0] = 1;
        desc->shape[1] = 64;
        desc->shape[2] = 32;
        desc->quant_type = QuantType::INT4_BLOCK32;
        desc->data_offset = 0x1234567890ABCDEFULL + offset;
        desc->data_bytes = 0xFEDCBA0987654321ULL + offset;
        desc->global_scale = 1.25f + static_cast<float>(offset);

        // Verify ASSERT/EXPECT macros do NOT trigger UBSan reference binding to unaligned address
        EXPECT_EQ(desc->ndims, 3U);
        EXPECT_EQ(desc->shape[0], 1U);
        EXPECT_EQ(desc->shape[1], 64U);
        EXPECT_EQ(desc->quant_type, QuantType::INT4_BLOCK32);
        EXPECT_EQ(desc->data_offset, 0x1234567890ABCDEFULL + offset);
        EXPECT_EQ(desc->data_bytes, 0xFEDCBA0987654321ULL + offset);
        EXPECT_FLOAT_EQ(desc->global_scale, 1.25f + static_cast<float>(offset));
        EXPECT_GE(desc->data_bytes, 0xFEDCBA0987654321ULL);
        EXPECT_LE(desc->data_offset, 0x1234567890ABCDEFULL + 8);
        EXPECT_NE(desc->data_offset, 0ULL);

        // 2. PlayerActionFrame at misaligned offset
        auto* frame = reinterpret_cast<PlayerActionFrame*>(raw_buffer + offset + sizeof(PWMFTensorDescriptor));
        frame->timestamp_us = 0x9988776655443322ULL + offset;
        frame->frame_index = static_cast<uint32_t>(100 + offset);
        frame->mouse_delta_yaw = 0.75f;
        frame->mouse_delta_pitch = -0.25f;
        frame->keys_pressed = 0x0105;

        EXPECT_EQ(frame->timestamp_us, 0x9988776655443322ULL + offset);
        EXPECT_EQ(frame->frame_index, static_cast<uint32_t>(100 + offset));
        EXPECT_FLOAT_EQ(frame->mouse_delta_yaw, 0.75f);
        EXPECT_FLOAT_EQ(frame->mouse_delta_pitch, -0.25f);
        EXPECT_EQ(frame->keys_pressed, static_cast<uint16_t>(0x0105));
    }
}

// ============================================================================
// PART 2: Unaligned Memory Buffer CRC32 & Parsing Stress
// ============================================================================

TEST(AdversarialMemoryChallenge, UnalignedBufferCRC32AllByteOffsets) {
    // Verify CRC32 works cleanly with zero sanitizer faults across 0..63 byte offsets
    std::vector<uint8_t> base_buf(2048 + 64, 0xAB);
    for (size_t i = 0; i < base_buf.size(); ++i) {
        base_buf[i] = static_cast<uint8_t>((i * 37 + 13) & 0xFF);
    }

    for (size_t offset = 0; offset < 64; ++offset) {
        const uint8_t* ptr = base_buf.data() + offset;
        for (size_t len = 0; len < 128; ++len) {
            uint32_t crc = ComputeCRC32(ptr, len);
            (void)crc;
        }
    }
}

// ============================================================================
// PART 3: INT4 Dequantization Boundary & Buffer Overflow Challenge
// ============================================================================

TEST(AdversarialMemoryChallenge, INT4BlockBatchBoundaryExactSizes) {
    // Verify batch dequantizer and quantizer on arbitrary block counts (1..100)
    for (size_t num_blocks = 1; num_blocks <= 65; ++num_blocks) {
        const size_t num_weights = num_blocks * 32;
        std::vector<float> input_weights(num_weights);
        for (size_t i = 0; i < num_weights; ++i) {
            input_weights[i] = static_cast<float>(i) * 0.05f - 1.5f;
        }

        std::vector<INT4Block32> blocks(num_blocks);
        QuantizeINT4Block32Batch(input_weights.data(), num_weights, blocks.data());

        std::vector<float> output_weights(num_weights, 0.0f);
        DequantizeINT4Block32Batch(blocks.data(), num_blocks, output_weights.data(), true);

        // Verify output is valid (finite, non-NaN)
        for (size_t i = 0; i < num_weights; ++i) {
            EXPECT_FALSE(std::isnan(output_weights[i]));
            EXPECT_FALSE(std::isinf(output_weights[i]));
        }
    }
}

// ============================================================================
// PART 4: Memory Leak & Lifetime Cycle Audit (ASan)
// ============================================================================

TEST(AdversarialMemoryChallenge, RepeatedPWMFParseSerializeCycles_ZeroLeaks) {
    // Repeatedly serialize and parse models to verify clean destructor cleanup
    for (int cycle = 0; cycle < 50; ++cycle) {
        PWMFWriter writer;
        writer.SetMetadata("{\"cycle\": " + std::to_string(cycle) + "}");

        std::vector<uint8_t> dummy_int4(128 * sizeof(INT4Block32), 0x33);
        uint32_t shape[5] = {128, 32, 0, 0, 0};
        writer.AddTensor("layer.test", 2, shape, QuantType::INT4_BLOCK32,
                         dummy_int4.data(), dummy_int4.size(), 0.5f);

        std::vector<uint8_t> blob = writer.Serialize();
        ASSERT_FALSE(blob.empty());

        PWMFParser parser;
        bool ok = parser.ParseMemory(blob.data(), blob.size());
        ASSERT_TRUE(ok);
        EXPECT_EQ(parser.GetNumTensors(), 1);

        Tensor t = parser.GetTensor(0);
        EXPECT_EQ(t.name(), "layer.test");
        EXPECT_EQ(t.size_bytes(), dummy_int4.size());
    }
}

TEST(AdversarialMemoryChallenge, FrustumVoxelGridRapidEvictionCycles_ZeroLeaks) {
    FrustumMemoryGrid grid(64, 0.5f, 15.0f);
    std::vector<uint8_t> dummy(128, 0x42);

    for (int i = 0; i < 500; ++i) {
        CameraPose p{static_cast<float>(i) * 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        grid.StoreLatents(p, dummy.data(), dummy.size(), 4, 1, 32);
        EXPECT_LE(grid.Size(), 64);
    }
    EXPECT_EQ(grid.Size(), 64);
}

TEST_RUNNER_MAIN()

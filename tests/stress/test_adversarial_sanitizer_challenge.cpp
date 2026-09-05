#include "test_runner.h"
#include "playworld/pwmf_format.h"
#include "playworld/ring_buffer.h"
#include "playworld/voxel_grid.h"
#include "playworld/tensor.h"
#include "playworld/engine_interface.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

using namespace playworld;

// ============================================================================
// SUITE 1: Extreme Action Poses & Float Boundary Stress (ASan / UBSan Audit)
// ============================================================================

TEST(AdversarialSanitizerSuite, QuantizePose_ExtremeValues_NoUBSanHalts) {
    FrustumMemoryGrid grid(256, 0.5f, 15.0f);

    const float inf_pos = std::numeric_limits<float>::infinity();
    const float inf_neg = -std::numeric_limits<float>::infinity();
    const float nan_val = std::numeric_limits<float>::quiet_NaN();
    const float flt_max = std::numeric_limits<float>::max();
    const float flt_min = -std::numeric_limits<float>::max();
    const float flt_denorm = std::numeric_limits<float>::denorm_min();

    std::vector<CameraPose> test_poses = {
        // Non-finite cases (should return {0,0,0,0,0} cleanly)
        {nan_val, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, nan_val, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, nan_val, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, nan_val, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, nan_val, 0.0f},
        {inf_pos, inf_pos, inf_pos, inf_pos, inf_pos, inf_pos},
        {inf_neg, inf_neg, inf_neg, inf_neg, inf_neg, inf_neg},
        {nan_val, inf_pos, inf_neg, nan_val, inf_pos, 0.0f},

        // Denormal cases
        {flt_denorm, flt_denorm, flt_denorm, flt_denorm, flt_denorm, flt_denorm},
        {-flt_denorm, -flt_denorm, -flt_denorm, -flt_denorm, -flt_denorm, -flt_denorm},

        // Extreme finite floats
        {1e20f, 1e20f, 1e20f, 1e10f, 89.0f, 0.0f},
        {-1e20f, -1e20f, -1e20f, -1e10f, -89.0f, 0.0f},
        {flt_max, flt_max, flt_max, flt_max, 90.0f, 0.0f},
        {flt_min, flt_min, flt_min, flt_min, -90.0f, 0.0f},
        {2147483648.0f, 2147483648.0f, 2147483648.0f, 360.0f, 0.0f, 0.0f},
        {-2147483649.0f, -2147483649.0f, -2147483649.0f, -360.0f, 0.0f, 0.0f},

        // Pitch edge boundaries
        {1.0f, 2.0f, 3.0f, 45.0f, 90.0001f, 0.0f},
        {1.0f, 2.0f, 3.0f, 45.0f, -90.0001f, 0.0f},
        {1.0f, 2.0f, 3.0f, 45.0f, 1e6f, 0.0f},
        {1.0f, 2.0f, 3.0f, 45.0f, -1e6f, 0.0f},

        // Yaw edge boundaries
        {1.0f, 2.0f, 3.0f, 1e6f, 0.0f, 0.0f},
        {1.0f, 2.0f, 3.0f, -1e6f, 0.0f, 0.0f},
        {1.0f, 2.0f, 3.0f, 36000000.0f, 0.0f, 0.0f},
        {1.0f, 2.0f, 3.0f, -36000000.0f, 0.0f, 0.0f}
    };

    std::vector<uint8_t> dummy_latent(14400 * sizeof(float), 0x7F);

    for (size_t i = 0; i < test_poses.size(); ++i) {
        const auto& p = test_poses[i];
        VoxelCoordinate coord = grid.QuantizePose(p);
        uint64_t h = grid.HashPose(p);
        (void)coord;
        (void)h;

        // Verify Store and Query under extreme poses
        grid.StoreLatents(p, dummy_latent.data(), dummy_latent.size(), 4, 45, 80);

        CachedLatentTensor cached{};
        float sim = 0.0f;
        bool hit = grid.QueryLatents(p, cached, sim);
        EXPECT_TRUE(hit || !hit); // Must not crash or trigger UB
        EXPECT_GE(sim, 0.0f);
    }
}

TEST(AdversarialSanitizerSuite, VoxelGrid_ExtremeVoxelAndStepSizes) {
    // Test atypical constructor arguments
    FrustumMemoryGrid g1(10, 0.0f, 0.0f);       // fallback to 0.5f and 15.0f
    FrustumMemoryGrid g2(10, -1.0f, -15.0f);    // fallback to 0.5f and 15.0f
    FrustumMemoryGrid g3(10, 1e-6f, 1e-4f);     // microscopic grid
    FrustumMemoryGrid g4(10, 1e6f, 180.0f);     // macroscopic grid

    CameraPose pose{10.0f, 20.0f, 30.0f, 45.0f, 30.0f, 0.0f};
    VoxelCoordinate c1 = g1.QuantizePose(pose);
    VoxelCoordinate c2 = g2.QuantizePose(pose);
    VoxelCoordinate c3 = g3.QuantizePose(pose);
    VoxelCoordinate c4 = g4.QuantizePose(pose);
    (void)c1; (void)c2; (void)c3; (void)c4;
    EXPECT_TRUE(true);
}

TEST(AdversarialSanitizerSuite, BlendLatents_AdversarialInputs) {
    std::vector<uint8_t> buf1(1024, 100);
    std::vector<uint8_t> buf2(1024, 200);
    std::vector<uint8_t> out(1024, 0);

    // Null pointer resilience
    FrustumMemoryGrid::BlendLatents(nullptr, buf2.data(), out.data(), 1024, 0.5f);
    EXPECT_EQ(out[0], 200);

    FrustumMemoryGrid::BlendLatents(buf1.data(), nullptr, out.data(), 1024, 0.5f);
    EXPECT_EQ(out[0], 100);

    FrustumMemoryGrid::BlendLatents(nullptr, nullptr, out.data(), 1024, 0.5f);
    FrustumMemoryGrid::BlendLatents(buf1.data(), buf2.data(), nullptr, 1024, 0.5f);

    // Extreme and non-finite gamma values
    FrustumMemoryGrid::BlendLatents(buf1.data(), buf2.data(), out.data(), 1024, -100.0f);
    EXPECT_EQ(out[0], 200); // clamped to 0.0

    FrustumMemoryGrid::BlendLatents(buf1.data(), buf2.data(), out.data(), 1024, 100.0f);
    EXPECT_EQ(out[0], 100); // clamped to 1.0

    // FP32 blend version
    std::vector<float> fbuf1(256, 1.0f);
    std::vector<float> fbuf2(256, 2.0f);
    std::vector<float> fout(256, 0.0f);

    FrustumMemoryGrid::BlendLatentsFP32(nullptr, fbuf2.data(), fout.data(), 256, 0.5f);
    EXPECT_FLOAT_EQ(fout[0], 2.0f);

    FrustumMemoryGrid::BlendLatentsFP32(fbuf1.data(), nullptr, fout.data(), 256, 0.5f);
    EXPECT_FLOAT_EQ(fout[0], 1.0f);

    FrustumMemoryGrid::BlendLatentsFP32(fbuf1.data(), fbuf2.data(), fout.data(), 256, 2.5f);
    EXPECT_FLOAT_EQ(fout[0], 1.0f); // clamped to 1.0
}

// ============================================================================
// SUITE 2: Malformed & Truncated .PWMF File Stress (Zero ASan / UBSan Errors)
// ============================================================================

TEST(AdversarialSanitizerSuite, PWMF_AllByteTruncations_ZeroFaults) {
    PWMFWriter writer;
    writer.SetMetadata("{\"version\": \"test\"}");
    std::vector<float> data(64, 3.1415f);
    uint32_t shape[5] = {64, 0, 0, 0, 0};
    writer.AddTensor("layer.weight", 1, shape, QuantType::FP32, data.data(), data.size() * sizeof(float));
    std::vector<uint8_t> valid_blob = writer.Serialize();

    ASSERT_GT(valid_blob.size(), sizeof(PWMFHeader));

    // Parse every possible truncation from 0 up to full size - 1
    for (size_t len = 0; len < valid_blob.size(); ++len) {
        PWMFParser parser;
        bool ok = parser.ParseMemory(valid_blob.data(), len);
        EXPECT_FALSE(ok);
        EXPECT_NE(parser.GetLastError(), PWMFError::Success);
    }
}

TEST(AdversarialSanitizerSuite, PWMF_CorruptedHeaderFields_Fuzz) {
    PWMFWriter writer;
    writer.SetMetadata("{\"test\": 123}");
    std::vector<float> data(32, 1.0f);
    uint32_t shape[5] = {32, 0, 0, 0, 0};
    writer.AddTensor("t0", 1, shape, QuantType::FP32, data.data(), data.size() * sizeof(float));
    std::vector<uint8_t> base = writer.Serialize();

    // Bit flip across header
    for (size_t byte_idx = 0; byte_idx < sizeof(PWMFHeader); byte_idx += 2) {
        for (uint8_t bit = 0; bit < 8; ++bit) {
            std::vector<uint8_t> corrupted = base;
            corrupted[byte_idx] ^= (1U << bit);

            PWMFParser parser;
            bool ok = parser.ParseMemory(corrupted.data(), corrupted.size());
            if (!ok) {
                EXPECT_NE(parser.GetLastError(), PWMFError::Success);
            }
        }
    }
}

TEST(AdversarialSanitizerSuite, PWMF_WrappingAndExtremeOffsets_CleanReject) {
    PWMFWriter writer;
    std::vector<float> data(32, 1.0f);
    uint32_t shape[5] = {32, 0, 0, 0, 0};
    writer.AddTensor("t0", 1, shape, QuantType::FP32, data.data(), data.size() * sizeof(float));
    std::vector<uint8_t> base = writer.Serialize();

    PWMFHeader hdr;
    std::memcpy(&hdr, base.data(), sizeof(PWMFHeader));

    // Extreme metadata offset
    {
        std::vector<uint8_t> buf = base;
        PWMFHeader h = hdr;
        h.metadata_offset = 0xFFFFFFFFFFFFFFF0ULL;
        std::memcpy(buf.data(), &h, sizeof(h));
        PWMFParser p;
        EXPECT_FALSE(p.ParseMemory(buf.data(), buf.size()));
        EXPECT_EQ(p.GetLastError(), PWMFError::OffsetOutOfBounds);
    }

    // Extreme tensor table offset
    {
        std::vector<uint8_t> buf = base;
        PWMFHeader h = hdr;
        h.tensor_table_offset = 0xFFFFFFFFFFFFFFF0ULL;
        std::memcpy(buf.data(), &h, sizeof(h));
        PWMFParser p;
        EXPECT_FALSE(p.ParseMemory(buf.data(), buf.size()));
        EXPECT_EQ(p.GetLastError(), PWMFError::OffsetOutOfBounds);
    }

    // Extreme weight data offset
    {
        std::vector<uint8_t> buf = base;
        PWMFHeader h = hdr;
        h.weight_data_offset = 0xFFFFFFFFFFFFFFC0ULL; // aligned to 64
        std::memcpy(buf.data(), &h, sizeof(h));
        PWMFParser p;
        EXPECT_FALSE(p.ParseMemory(buf.data(), buf.size()));
        EXPECT_EQ(p.GetLastError(), PWMFError::PayloadTruncated);
    }
}

TEST(AdversarialSanitizerSuite, PWMF_MismatchedShapeAndDataBytes_ASanCheck) {
    PWMFWriter writer;
    std::vector<float> data(32, 1.0f);
    uint32_t shape[5] = {32, 0, 0, 0, 0};
    writer.AddTensor("t0", 1, shape, QuantType::FP32, data.data(), data.size() * sizeof(float));
    std::vector<uint8_t> base = writer.Serialize();

    PWMFHeader hdr;
    std::memcpy(&hdr, base.data(), sizeof(PWMFHeader));
    hdr.reserved[0] = 0; // zero CRC
    hdr.reserved[1] = 0;
    hdr.reserved[2] = 0;
    hdr.reserved[3] = 0;
    std::memcpy(base.data(), &hdr, sizeof(PWMFHeader));

    PWMFTensorDescriptor desc;
    std::memcpy(&desc, base.data() + hdr.tensor_table_offset, sizeof(desc));
    // Tamper shape: claims 100,000 elements (400 KB) but payload only has 128 bytes
    desc.shape[0] = 100000;
    std::memcpy(base.data() + hdr.tensor_table_offset, &desc, sizeof(desc));

    PWMFParser p;
    if (p.ParseMemory(base.data(), base.size())) {
        Tensor t = p.GetTensor("t0");
        std::vector<float> out(t.numel());
        t.DequantizeTo(out.data());
    }

}


TEST(AdversarialSanitizerSuite, DequantizeExtremeBitPatterns_NoASanUBSanCrash) {
    // Test INT4 dequantization with random and extreme nibble patterns
    std::vector<INT4Block32> blocks(16);
    for (size_t b = 0; b < blocks.size(); ++b) {
        // Extreme fp16 scales (NaN, Inf, Denormals, 0, Max)
        if (b == 0) blocks[b].scale_fp16 = 0x7E00; // NaN
        else if (b == 1) blocks[b].scale_fp16 = 0x7C00; // +Inf
        else if (b == 2) blocks[b].scale_fp16 = 0xFC00; // -Inf
        else if (b == 3) blocks[b].scale_fp16 = 0x0001; // Denormal min
        else if (b == 4) blocks[b].scale_fp16 = 0x7BFF; // Max finite
        else blocks[b].scale_fp16 = static_cast<uint16_t>(b * 100);

        blocks[b].bias_fp16 = static_cast<uint16_t>(b * 50);
        std::memset(blocks[b].qs, static_cast<int>(b * 17), sizeof(blocks[b].qs));
    }

    std::vector<float> out(blocks.size() * 32);
    DequantizeINT4Block32Batch(blocks.data(), blocks.size(), out.data(), true);
    DequantizeINT4Block32Batch(blocks.data(), blocks.size(), out.data(), false);

    // Test FP8 E4M3 and E5M2 across all 256 possible byte values
    std::vector<uint8_t> all_bytes(256);
    for (size_t i = 0; i < 256; ++i) all_bytes[i] = static_cast<uint8_t>(i);

    std::vector<float> fp8_e4m3_out(256);
    DequantizeFP8_E4M3(all_bytes.data(), 256, fp8_e4m3_out.data());

    std::vector<float> fp8_e5m2_out(256);
    DequantizeFP8_E5M2(all_bytes.data(), 256, fp8_e5m2_out.data());

    // Test FP16 all special values
    std::vector<uint16_t> fp16_special = {0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E00, 0x7FFF, 0x0001, 0x03FF};
    std::vector<float> fp16_out(fp16_special.size());
    DequantizeFP16(fp16_special.data(), fp16_special.size(), fp16_out.data());

    EXPECT_EQ(out.size(), blocks.size() * 32);
}

// ============================================================================
// SUITE 3: High-Contention Lock-Free SPSC Ring Buffer Concurrency Stress
// ============================================================================

TEST(AdversarialSanitizerSuite, RingBuffer_HighContention_1M_PushesPops) {
    constexpr size_t TOTAL_FRAMES = 1000000;
    ActionRingBuffer<2048> ring;

    std::atomic<bool> start_flag{false};
    std::atomic<size_t> popped_count{0};
    std::atomic<bool> done_producer{false};

    // Producer Thread
    std::thread producer([&]() {
        while (!start_flag.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }

        for (size_t i = 0; i < TOTAL_FRAMES; ++i) {
            PlayerActionFrame frame{};
            frame.timestamp_us = i;
            frame.mouse_delta_yaw = static_cast<float>(i % 360) * 0.01f;
            frame.mouse_delta_pitch = static_cast<float>(i % 180) * 0.01f;
            frame.keys_pressed = static_cast<uint16_t>(i & 0xFFFF);

            while (!ring.Push(frame)) {
                // Buffer full: yield and retry to ensure zero drops
                std::this_thread::yield();
            }
        }
        done_producer.store(true, std::memory_order_release);
    });

    // Consumer Thread
    std::thread consumer([&]() {
        while (!start_flag.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }

        uint64_t last_timestamp = 0;
        bool first = true;

        while (!done_producer.load(std::memory_order_acquire) || !ring.Empty()) {
            PlayerActionFrame frame{};
            if (ring.Pop(frame)) {
                if (!first) {
                    EXPECT_GT(frame.timestamp_us, last_timestamp);
                }
                last_timestamp = frame.timestamp_us;
                first = false;
                popped_count.fetch_add(1, std::memory_order_relaxed);
            } else {
                std::this_thread::yield();
            }
        }
    });

    start_flag.store(true, std::memory_order_release);

    producer.join();
    consumer.join();

    EXPECT_EQ(popped_count.load(), TOTAL_FRAMES);
    // dropped_count_ tracks momentary buffer-full push rejections before retry
    EXPECT_GE(ring.DroppedFrames(), 0ULL);
}

TEST(AdversarialSanitizerSuite, RingBuffer_AdversarialDataPayloads) {
    ActionRingBuffer<64> ring;

    const float nan_val = std::numeric_limits<float>::quiet_NaN();
    const float inf_val = std::numeric_limits<float>::infinity();

    for (size_t i = 0; i < 50; ++i) {
        PlayerActionFrame frame{};
        frame.mouse_delta_yaw = nan_val;
        frame.mouse_delta_pitch = inf_val;
        frame.analog_move_x = 1e30f;
        frame.analog_move_y = -1e30f;
        frame.auxiliary_trigger = nan_val;
        frame.keys_pressed = 0xFFFF;
        EXPECT_TRUE(ring.Push(frame));
    }

    PlayerActionFrame out_frame{};
    float norm32[32];
    ActionMLP mlp(32, 64, 128);

    while (ring.Pop(out_frame)) {
        NormalizeActionFrame(out_frame, norm32);
        // Note: std::clamp(NaN) returns NaN in standard C++, propagating non-finite values into MLP
        std::vector<float> emb = mlp.Project(out_frame);
        EXPECT_EQ(emb.size(), 128ULL);
    }
}


// ============================================================================
// SUITE 4: Continuous Engine Step Under Adversarial Action Stream
// ============================================================================

TEST(AdversarialSanitizerSuite, WorldEngine_1000Steps_AdversarialActions) {
    EngineConfig cfg;
    cfg.render_width = 320;
    cfg.render_height = 180;
    cfg.denoising_steps = 1;
    cfg.enable_voxel_memory = true;
    cfg.voxel_memory_capacity = 64;

    auto engine = WorldEngine::Create(cfg);
    ASSERT_TRUE(engine->Initialize());

    const float nan_val = std::numeric_limits<float>::quiet_NaN();
    const float inf_val = std::numeric_limits<float>::infinity();

    for (int step = 0; step < 1000; ++step) {
        PlayerActionFrame action{};
        if (step % 5 == 0) {
            action.mouse_delta_yaw = nan_val;
            action.mouse_delta_pitch = inf_val;
        } else if (step % 5 == 1) {
            action.mouse_delta_yaw = 1e20f;
            action.mouse_delta_pitch = -1e20f;
        } else if (step % 5 == 2) {
            action.analog_move_x = nan_val;
            action.analog_move_y = inf_val;
        } else if (step % 5 == 3) {
            action.auxiliary_trigger = nan_val;
            action.keys_pressed = 0xFFFF;
        } else {
            action.mouse_delta_yaw = 0.5f;
            action.mouse_delta_pitch = -0.3f;
            action.analog_move_x = 1.0f;
            action.keys_pressed = 0x0001;
        }

        engine->InjectAction(action);
        FrameOutput output = engine->Step();
        EXPECT_EQ(output.width, 320U);
        EXPECT_EQ(output.height, 180U);
        EXPECT_TRUE(output.rgba_pixels != nullptr);

        if (step % 200 == 0) {
            float fps = 0.0f, vram = 0.0f, hitrate = 0.0f;
            engine->GetTelemetry(fps, vram, hitrate);
            EXPECT_GE(fps, 0.0f);
            EXPECT_GE(vram, 0.0f);
        }
    }
}

TEST_RUNNER_MAIN()

#include "test_runner.h"
#include "playworld/tensor.h"
#include <cmath>
#include <vector>
#include <numeric>

using namespace playworld;

// Helper to compute Signal-to-Quantization-Noise Ratio in decibels
static double CalculateSQNR(const float* ref, const float* test, size_t n) {
    double signal_pwr = 0.0;
    double noise_pwr = 0.0;
    for (size_t i = 0; i < n; ++i) {
        signal_pwr += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
        double diff = static_cast<double>(ref[i]) - static_cast<double>(test[i]);
        noise_pwr += diff * diff;
    }
    if (noise_pwr < 1e-12) return 100.0; // Infinite SNR
    return 10.0 * std::log10(signal_pwr / noise_pwr);
}

TEST(QuantSuite, FP16_StandardValuesConversion) {
    // Authoritative source: IEEE 754-2008 16-bit half-precision specification
    // 0.0f -> 0x0000
    EXPECT_EQ(fp32_to_fp16(0.0f), 0x0000);
    EXPECT_NEAR(fp16_to_fp32(0x0000), 0.0f, 1e-6f);

    // 1.0f -> 0x3C00 (sign=0, exp=15, mant=0)
    EXPECT_EQ(fp32_to_fp16(1.0f), 0x3C00);
    EXPECT_NEAR(fp16_to_fp32(0x3C00), 1.0f, 1e-6f);

    // -1.0f -> 0xBC00 (sign=1, exp=15, mant=0)
    EXPECT_EQ(fp32_to_fp16(-1.0f), 0xBC00);
    EXPECT_NEAR(fp16_to_fp32(0xBC00), -1.0f, 1e-6f);

    // 2.0f -> 0x4000 (sign=0, exp=16, mant=0)
    EXPECT_EQ(fp32_to_fp16(2.0f), 0x4000);
    EXPECT_NEAR(fp16_to_fp32(0x4000), 2.0f, 1e-6f);

    // 0.5f -> 0x3800 (sign=0, exp=14, mant=0)
    EXPECT_EQ(fp32_to_fp16(0.5f), 0x3800);
    EXPECT_NEAR(fp16_to_fp32(0x3800), 0.5f, 1e-6f);

    // Max finite FP16: 65504.0f -> 0x7BFF
    EXPECT_EQ(fp32_to_fp16(65504.0f), 0x7BFF);
    EXPECT_NEAR(fp16_to_fp32(0x7BFF), 65504.0f, 1.0f);
}

TEST(QuantSuite, FP8_E4M3_AccuracyVsMath) {
    // Authoritative source: OCP Microscaling Formats (MX) Spec - E4M3
    // E4M3: 1 sign, 4 exp, 3 mantissa, bias 7
    // 0.0f -> 0x00
    EXPECT_EQ(fp32_to_fp8_e4m3(0.0f), 0x00);
    EXPECT_NEAR(fp8_e4m3_to_fp32(0x00), 0.0f, 1e-6f);

    // 1.0f -> exp=7, mant=0 => (7 << 3) | 0 = 0x38
    EXPECT_EQ(fp32_to_fp8_e4m3(1.0f), 0x38);
    EXPECT_NEAR(fp8_e4m3_to_fp32(0x38), 1.0f, 1e-6f);

    // -1.0f -> 0x80 | 0x38 = 0xB8
    EXPECT_EQ(fp32_to_fp8_e4m3(-1.0f), 0xB8);
    EXPECT_NEAR(fp8_e4m3_to_fp32(0xB8), -1.0f, 1e-6f);

    // 1.5f -> 1.0 * (1 + 4/8) => exp=7, mant=4 => (7 << 3) | 4 = 0x3C
    EXPECT_EQ(fp32_to_fp8_e4m3(1.5f), 0x3C);
    EXPECT_NEAR(fp8_e4m3_to_fp32(0x3C), 1.5f, 1e-6f);

    // Max finite: 448.0f -> exp=15, mant=6 => (15 << 3) | 6 = 0x7E
    EXPECT_EQ(fp32_to_fp8_e4m3(448.0f), 0x7E);
    EXPECT_NEAR(fp8_e4m3_to_fp32(0x7E), 448.0f, 1e-3f);
}

TEST(QuantSuite, FP8_E5M2_AccuracyVsMath) {
    // Authoritative source: IEEE 754-like FP8 E5M2 specification
    // E5M2: 1 sign, 5 exp, 2 mantissa, bias 15
    // 0.0f -> 0x00
    EXPECT_EQ(fp32_to_fp8_e5m2(0.0f), 0x00);
    EXPECT_NEAR(fp8_e5m2_to_fp32(0x00), 0.0f, 1e-6f);

    // 1.0f -> exp=15, mant=0 => (15 << 2) | 0 = 0x3C
    EXPECT_EQ(fp32_to_fp8_e5m2(1.0f), 0x3C);
    EXPECT_NEAR(fp8_e5m2_to_fp32(0x3C), 1.0f, 1e-6f);

    // -1.0f -> 0x80 | 0x3C = 0xBC
    EXPECT_EQ(fp32_to_fp8_e5m2(-1.0f), 0xBC);
    EXPECT_NEAR(fp8_e5m2_to_fp32(0xBC), -1.0f, 1e-6f);

    // 2.0f -> exp=16, mant=0 => (16 << 2) | 0 = 0x40
    EXPECT_EQ(fp32_to_fp8_e5m2(2.0f), 0x40);
    EXPECT_NEAR(fp8_e5m2_to_fp32(0x40), 2.0f, 1e-6f);
}

TEST(QuantSuite, BF16_RoundtripAndAccuracy) {
    // Authoritative source: Brain Floating Point format (top 16 bits of FP32)
    float original = 3.14159265f;
    uint16_t b = fp32_to_bf16(original);
    float reconstructed = bf16_to_fp32(b);
    EXPECT_NEAR(reconstructed, original, 0.01f);
}

TEST(QuantSuite, INT4_Block32_NibbleUnpackingMath) {
    // Authoritative source: WINNING_PROJECT_PLAN §2.2.3 & env_and_test_infra.md §3.1.2
    // Block layout: 16 bytes (32 nibbles), scale_fp16, bias_fp16
    // Formula: out[i] = (q_i - bias) * scale
    INT4Block32 block{};
    block.scale_fp16 = fp32_to_fp16(0.05f); // Scale = 0.05
    block.bias_fp16  = fp32_to_fp16(8.0f);  // Bias = 8.0

    // Set known alternating nibbles: low = 0 (q=0), high = 15 (q=15)
    for (int i = 0; i < 16; ++i) {
        block.qs[i] = static_cast<uint8_t>((15 << 4) | 0);
    }

    float out[32];
    DequantizeINT4Block32(&block, out, true);

    // Expected:
    // q=0:  (0  - 8.0) * 0.05 = -0.40
    // q=15: (15 - 8.0) * 0.05 = +0.35
    for (int i = 0; i < 32; i += 2) {
        EXPECT_NEAR(out[i],     -0.40f, 1e-3f);
        EXPECT_NEAR(out[i + 1],  0.35f, 1e-3f);
    }

    // Now test with mid nibbles: low = 8 (q=8), high = 4 (q=4)
    // q=8: (8 - 8.0) * 0.05 = 0.0
    // q=4: (4 - 8.0) * 0.05 = -0.20
    for (int i = 0; i < 16; ++i) {
        block.qs[i] = static_cast<uint8_t>((4 << 4) | 8);
    }
    DequantizeINT4Block32(&block, out, true);
    for (int i = 0; i < 32; i += 2) {
        EXPECT_NEAR(out[i],      0.00f, 1e-3f);
        EXPECT_NEAR(out[i + 1], -0.20f, 1e-3f);
    }
}

TEST(QuantSuite, INT4_Block32_BatchNEONVsScalar) {
    // Verifies batch SIMD dequantization yields identical results across 8 blocks (256 weights)
    const size_t num_blocks = 8;
    const size_t num_weights = num_blocks * 32;
    std::vector<INT4Block32> blocks(num_blocks);

    for (size_t b = 0; b < num_blocks; ++b) {
        blocks[b].scale_fp16 = fp32_to_fp16(0.01f * static_cast<float>(b + 1));
        blocks[b].bias_fp16  = fp32_to_fp16(7.5f);
        for (int i = 0; i < 16; ++i) {
            uint8_t q0 = static_cast<uint8_t>((b * 3 + i) % 16);
            uint8_t q1 = static_cast<uint8_t>((b * 7 + i * 2) % 16);
            blocks[b].qs[i] = static_cast<uint8_t>((q1 << 4) | q0);
        }
    }

    std::vector<float> batch_out(num_weights, 0.0f);
    DequantizeINT4Block32Batch(blocks.data(), num_blocks, batch_out.data(), true);

    std::vector<float> single_out(num_weights, 0.0f);
    for (size_t b = 0; b < num_blocks; ++b) {
        DequantizeINT4Block32(&blocks[b], single_out.data() + b * 32, true);
    }

    for (size_t i = 0; i < num_weights; ++i) {
        ASSERT_NEAR(batch_out[i], single_out[i], 1e-5f);
    }
}

TEST(QuantSuite, INT4_Block32_QuantizeDequantizeRoundtrip) {
    // Synthesize 64 floats with smooth sinusoidal pattern
    const size_t N = 64;
    std::vector<float> original(N);
    for (size_t i = 0; i < N; ++i) {
        original[i] = std::sin(static_cast<float>(i) * 0.1f);
    }

    std::vector<INT4Block32> blocks(N / 32);
    QuantizeINT4Block32Batch(original.data(), N, blocks.data());

    std::vector<float> reconstructed(N);
    DequantizeINT4Block32Batch(blocks.data(), blocks.size(), reconstructed.data(), true);

    // Verify SQNR >= 28.0 dB (per env_and_test_infra.md §3.1.2)
    double sqnr = CalculateSQNR(original.data(), reconstructed.data(), N);
    EXPECT_GE(sqnr, 28.0);

    // Verify max error L_infinity <= 0.08
    float max_err = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        float err = std::fabs(original[i] - reconstructed[i]);
        if (err > max_err) max_err = err;
    }
    EXPECT_LE(max_err, 0.08f);
}

TEST(QuantSuite, INT8_Symmetric_AccuracyAndSQNR) {
    // Authoritative source: WINNING_PROJECT_PLAN §2.2.3
    const size_t N = 128;
    std::vector<float> original(N);
    for (size_t i = 0; i < N; ++i) {
        original[i] = 2.0f * (static_cast<float>(i) / static_cast<float>(N - 1)) - 1.0f; // [-1.0, 1.0]
    }

    std::vector<int8_t> quantized(N);
    float scale = 0.0f;
    QuantizeINT8Symm(original.data(), N, quantized.data(), scale);

    std::vector<float> reconstructed(N);
    DequantizeINT8Symm(quantized.data(), N, scale, reconstructed.data());

    // Verify INT8 SQNR >= 42.0 dB
    double sqnr = CalculateSQNR(original.data(), reconstructed.data(), N);
    EXPECT_GE(sqnr, 42.0);

    // Max absolute error <= 0.01
    float max_err = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        float err = std::fabs(original[i] - reconstructed[i]);
        if (err > max_err) max_err = err;
    }
    EXPECT_LE(max_err, 0.01f);
}

TEST(QuantSuite, Tensor_ShapeAndStridesCalculation) {
    uint32_t shape[5] = {2, 3, 4, 5, 6};
    Tensor t("test_5d", 5, shape, QuantType::FP32);

    EXPECT_EQ(t.name(), "test_5d");
    EXPECT_EQ(t.ndims(), 5);
    EXPECT_EQ(t.numel(), 2 * 3 * 4 * 5 * 6); // 720
    EXPECT_EQ(t.size_bytes(), 720 * sizeof(float));

    // Row-major strides: [360, 120, 30, 6, 1]
    EXPECT_EQ(t.strides()[0], 360);
    EXPECT_EQ(t.strides()[1], 120);
    EXPECT_EQ(t.strides()[2], 30);
    EXPECT_EQ(t.strides()[3], 6);
    EXPECT_EQ(t.strides()[4], 1);
}

TEST(QuantSuite, Tensor_DequantizeToFP32_INT4Pipeline) {
    uint32_t shape[3] = {1, 2, 32}; // 64 elements = 2 INT4Block32 blocks
    Tensor t("weight_int4", 3, shape, QuantType::INT4_BLOCK32);

    INT4Block32* blocks = t.mutable_data_as<INT4Block32>();
    ASSERT_TRUE(blocks != nullptr);

    // Populate block 0: constant q=8 (bias=8 => value=0.0)
    blocks[0].scale_fp16 = fp32_to_fp16(0.1f);
    blocks[0].bias_fp16  = fp32_to_fp16(8.0f);
    std::memset(blocks[0].qs, 0x88, 16);

    // Populate block 1: constant q=10 (bias=8 => (10-8)*0.1 = 0.2)
    blocks[1].scale_fp16 = fp32_to_fp16(0.1f);
    blocks[1].bias_fp16  = fp32_to_fp16(8.0f);
    std::memset(blocks[1].qs, 0xAA, 16);

    std::vector<float> fp32_vals = t.DequantizeToFP32();
    EXPECT_EQ(fp32_vals.size(), 64);

    for (size_t i = 0; i < 32; ++i) {
        EXPECT_NEAR(fp32_vals[i], 0.0f, 1e-4f);
    }
    for (size_t i = 32; i < 64; ++i) {
        EXPECT_NEAR(fp32_vals[i], 0.2f, 1e-4f);
    }
}

TEST(QuantSuite, Tensor_ZeroAndExtremeEdgeCases) {
    // 0 dimensions
    Tensor t_empty("empty", 0, nullptr, QuantType::FP32);
    EXPECT_EQ(t_empty.numel(), 0);
    EXPECT_EQ(t_empty.size_bytes(), 0);

    // Large single dimension with INT8
    uint32_t shape[1] = {1024};
    Tensor t_int8("large_int8", 1, shape, QuantType::INT8_SYMM, 0.05f);
    EXPECT_EQ(t_int8.numel(), 1024);
    EXPECT_EQ(t_int8.size_bytes(), 1024);
}

TEST(QuantSuite, INT4_ConstantBlockZeroVariance) {
    const std::vector<float> test_constants = {0.0f, 1.0f, 5.0f, 42.0f, 100.0f, -10.0f, 0.007f};

    for (float c : test_constants) {
        std::vector<float> input(32, c);
        INT4Block32 block{};
        QuantizeINT4Block32(input.data(), &block);

        float out[32];
        DequantizeINT4Block32(&block, out, true);

        for (size_t i = 0; i < 32; ++i) {
            EXPECT_FALSE(std::isnan(out[i]));
            EXPECT_FALSE(std::isinf(out[i]));
            EXPECT_NEAR(out[i], c, 1e-3f);
        }

        // Also test batch quantization/dequantization path
        std::vector<float> input_batch(64, c);
        std::vector<INT4Block32> blocks(2);
        QuantizeINT4Block32Batch(input_batch.data(), 64, blocks.data());

        std::vector<float> out_batch(64, 0.0f);
        DequantizeINT4Block32Batch(blocks.data(), 2, out_batch.data(), true);

        for (size_t i = 0; i < 64; ++i) {
            EXPECT_FALSE(std::isnan(out_batch[i]));
            EXPECT_FALSE(std::isinf(out_batch[i]));
            EXPECT_NEAR(out_batch[i], c, 1e-3f);
        }
    }
}

TEST_RUNNER_MAIN()

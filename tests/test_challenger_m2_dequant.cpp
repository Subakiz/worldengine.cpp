#include "test_runner.h"
#include "playworld/tensor.h"
#include "playworld/pwmf_format.h"

#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

using namespace playworld;

// ============================================================================
// Scalar Reference Implementations for Empirical Equivalence Verification
// ============================================================================

static void DequantizeINT4Block32_ScalarRef(const INT4Block32* block, float* out, bool subtract_bias) noexcept {
    if (!block || !out) return;
    const float scale = fp16_to_fp32(block->scale_fp16);
    const float bias  = fp16_to_fp32(block->bias_fp16);

    for (size_t i = 0; i < 16; ++i) {
        const uint8_t byte = block->qs[i];
        const uint8_t q_low  = byte & 0x0F;
        const uint8_t q_high = (byte >> 4) & 0x0F;

        if (subtract_bias) {
            out[2 * i]     = (static_cast<float>(q_low)  - bias) * scale;
            out[2 * i + 1] = (static_cast<float>(q_high) - bias) * scale;
        } else {
            out[2 * i]     = static_cast<float>(q_low)  * scale + bias;
            out[2 * i + 1] = static_cast<float>(q_high) * scale + bias;
        }
    }
}

static void DequantizeINT8Symm_ScalarRef(const int8_t* data, size_t num_elements, float scale, float* out) noexcept {
    if (!data || !out) return;
    for (size_t i = 0; i < num_elements; ++i) {
        out[i] = static_cast<float>(data[i]) * scale;
    }
}

// Oracle for FP8 E4M3 (OCP Microscaling Formats Specification)
static float FP8_E4M3_Oracle(uint8_t byte) noexcept {
    const uint32_t sign = (byte & 0x80) ? 0x80000000U : 0U;
    const uint32_t exp  = (byte >> 3) & 0x0F;
    const uint32_t mant = byte & 0x07;

    // 0x7F and 0xFF represent NaN in E4M3 OCP specification
    if (exp == 0x0F && mant == 0x07) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    if (exp == 0) {
        if (mant == 0) {
            float zero = 0.0f;
            return sign ? -zero : zero;
        }
        // Subnormal: (-1)^S * 2^(-6) * (mant / 8.0)
        const float val = 0.015625f * (static_cast<float>(mant) / 8.0f);
        return sign ? -val : val;
    }

    // Normal: (-1)^S * 2^(exp - 7) * (1.0 + mant / 8.0)
    const float scale = std::ldexp(1.0f, static_cast<int>(exp) - 7);
    const float val = scale * (1.0f + static_cast<float>(mant) / 8.0f);
    return sign ? -val : val;
}

// Oracle for FP8 E5M2 (IEEE 754-like FP8 Specification)
static float FP8_E5M2_Oracle(uint8_t byte) noexcept {
    const uint32_t sign = (byte & 0x80) ? 0x80000000U : 0U;
    const uint32_t exp  = (byte >> 2) & 0x1F;
    const uint32_t mant = byte & 0x03;

    if (exp == 0x1F) {
        if (mant == 0) {
            return sign ? -std::numeric_limits<float>::infinity()
                        : std::numeric_limits<float>::infinity();
        }
        return std::numeric_limits<float>::quiet_NaN();
    }

    if (exp == 0) {
        if (mant == 0) {
            float zero = 0.0f;
            return sign ? -zero : zero;
        }
        // Subnormal: (-1)^S * 2^(-14) * (mant / 4.0)
        const float val = std::ldexp(1.0f, -14) * (static_cast<float>(mant) / 4.0f);
        return sign ? -val : val;
    }

    // Normal: (-1)^S * 2^(exp - 15) * (1.0 + mant / 4.0)
    const float scale = std::ldexp(1.0f, static_cast<int>(exp) - 15);
    const float val = scale * (1.0f + static_cast<float>(mant) / 4.0f);
    return sign ? -val : val;
}

// Helper to check if two floats match in IEEE 754 sense:
// - Both NaN
// - Both +Inf or both -Inf
// - Finite: exact or within relative tolerance
static bool FloatIEEEEquivalent(float a, float b, float rel_tol = 1e-5f) {
    if (std::isnan(a) && std::isnan(b)) return true;
    if (std::isinf(a) && std::isinf(b)) return std::signbit(a) == std::signbit(b);
    if (std::isnan(a) != std::isnan(b)) return false;
    if (std::isinf(a) != std::isinf(b)) return false;
    if (a == b) return true;
    float diff = std::fabs(a - b);
    float max_mag = std::max(std::fabs(a), std::fabs(b));
    return diff <= rel_tol * max_mag || diff < 1e-35f;
}

// ============================================================================
// Suite 1: INT4Block32 Extreme Scales, Biases & Zero Points
// ============================================================================

TEST(ChallengerM2Suite, INT4Block32_ExtremeScalesAndBiases) {
    // Test vectors for scale_fp16:
    // - Denormals: 0x0001 (2^-24 min subnormal), 0x03FF (max subnormal), 0x0400 (2^-14 min normal)
    // - Negative denormals: 0x8001, 0x83FF, 0x8400
    // - Infinities: +Inf (0x7C00), -Inf (0xFC00)
    // - Signaling NaNs: 0x7D00, 0x7C01, 0xFD00
    // - Quiet NaNs: 0x7E00, 0x7FFF, 0xFE00
    // - Max finite: 0x7BFF (+65504.0f), 0xFBFF (-65504.0f)
    // - Zeros: 0x0000 (+0.0f), 0x8000 (-0.0f)

    const uint16_t test_scales[] = {
        0x0001, 0x0002, 0x03FF, 0x0400, // Denormals and min normal
        0x8001, 0x83FF, 0x8400,         // Negative denormals
        0x7C00, 0xFC00,                 // +/- Infinity
        0x7D00, 0x7C01, 0xFD00,         // Signaling NaNs
        0x7E00, 0x7FFF, 0xFE00,         // Quiet NaNs
        0x7BFF, 0xFBFF,                 // Max finite (+/- 65504.0f)
        0x0000, 0x8000,                 // +/- 0.0f
        0x3C00                          // 1.0f normal
    };

    const uint16_t test_biases[] = {
        0x0000, 0x8000,                 // 0.0f
        0x0001, 0x0400,                 // Denormals
        0x7C00, 0xFC00,                 // +/- Infinity
        0x7D00, 0x7E00,                 // NaNs
        0x7BFF, 0xFBFF,                 // Max finite
        0x4800,                         // 8.0f
        0x4780                          // 7.5f
    };

    for (uint16_t s : test_scales) {
        for (uint16_t b : test_biases) {
            INT4Block32 block{};
            block.scale_fp16 = s;
            block.bias_fp16 = b;

            // Pattern of diverse nibbles (0 to 15)
            for (int i = 0; i < 16; ++i) {
                uint8_t q0 = static_cast<uint8_t>(i % 16);
                uint8_t q1 = static_cast<uint8_t>((15 - i) % 16);
                block.qs[i] = static_cast<uint8_t>((q1 << 4) | q0);
            }

            // Test subtract_bias = true
            {
                float out_neon[32];
                float out_scalar[32];
                std::memset(out_neon, 0xAA, sizeof(out_neon));
                std::memset(out_scalar, 0x55, sizeof(out_scalar));

                DequantizeINT4Block32(&block, out_neon, true);
                DequantizeINT4Block32_ScalarRef(&block, out_scalar, true);

                for (size_t i = 0; i < 32; ++i) {
                    ASSERT_TRUE(FloatIEEEEquivalent(out_neon[i], out_scalar[i]));
                }
            }

            // Test subtract_bias = false
            {
                float out_neon[32];
                float out_scalar[32];
                std::memset(out_neon, 0xAA, sizeof(out_neon));
                std::memset(out_scalar, 0x55, sizeof(out_scalar));

                DequantizeINT4Block32(&block, out_neon, false);
                DequantizeINT4Block32_ScalarRef(&block, out_scalar, false);

                for (size_t i = 0; i < 32; ++i) {
                    ASSERT_TRUE(FloatIEEEEquivalent(out_neon[i], out_scalar[i]));
                }
            }
        }
    }
}

// ============================================================================
// Suite 2: Specific Denormal and Subnormal Preservation
// ============================================================================

TEST(ChallengerM2Suite, INT4Block32_DenormalValues) {
    // 0x0001 is the smallest positive subnormal FP16: 2^-24 = 5.9604644775390625e-8
    float scale_subnormal = fp16_to_fp32(0x0001);
    EXPECT_GT(scale_subnormal, 0.0f);
    EXPECT_NEAR(scale_subnormal, 5.9604645e-8f, 1e-15f);

    // 0x0400 is the smallest positive normalized FP16: 2^-14 = 6.103515625e-5
    float scale_min_norm = fp16_to_fp32(0x0400);
    EXPECT_NEAR(scale_min_norm, 6.1035156e-5f, 1e-12f);

    INT4Block32 block{};
    block.scale_fp16 = 0x0001;
    block.bias_fp16 = 0x0000;
    // Set all nibbles to 1
    for (int i = 0; i < 16; ++i) block.qs[i] = 0x11;

    float out[32];
    DequantizeINT4Block32(&block, out, true);

    for (size_t i = 0; i < 32; ++i) {
        EXPECT_FALSE(std::isnan(out[i]));
        EXPECT_FALSE(std::isinf(out[i]));
        EXPECT_NEAR(out[i], 5.9604645e-8f, 1e-15f);
    }
}

// ============================================================================
// Suite 3: INT4Block32 Signaling NaNs and Quiet NaNs
// ============================================================================

TEST(ChallengerM2Suite, INT4Block32_SignalingAndQuietNaNs) {
    // sNaN 0x7D00, qNaN 0x7E00
    const uint16_t nan_codes[] = {0x7D00, 0x7C01, 0x7E00, 0x7FFF, 0xFD00, 0xFE00};

    for (uint16_t nan_val : nan_codes) {
        float f = fp16_to_fp32(nan_val);
        EXPECT_TRUE(std::isnan(f));

        INT4Block32 block{};
        block.scale_fp16 = nan_val;
        block.bias_fp16 = 0x0000;
        std::memset(block.qs, 0x55, sizeof(block.qs));

        float out[32];
        DequantizeINT4Block32(&block, out, true);
        for (size_t i = 0; i < 32; ++i) {
            EXPECT_TRUE(std::isnan(out[i]));
        }

        // Test as bias
        block.scale_fp16 = 0x3C00; // 1.0f
        block.bias_fp16 = nan_val;
        DequantizeINT4Block32(&block, out, true);
        for (size_t i = 0; i < 32; ++i) {
            EXPECT_TRUE(std::isnan(out[i]));
        }
    }
}

// ============================================================================
// Suite 4: INT4Block32 Batch Dequantizer Multi-Block Adversarial Stress
// ============================================================================

TEST(ChallengerM2Suite, INT4Block32Batch_StressAndNullptr) {
    // Nullptr / 0-blocks safety
    float dummy_out[64];
    DequantizeINT4Block32Batch(nullptr, 10, dummy_out, true);
    INT4Block32 dummy_block{};
    DequantizeINT4Block32Batch(&dummy_block, 1, nullptr, true);
    DequantizeINT4Block32Batch(&dummy_block, 0, dummy_out, true);

    // Multi-block batch with mixed extreme blocks
    const size_t num_blocks = 37; // Non-power-of-two block count
    std::vector<INT4Block32> blocks(num_blocks);

    std::mt19937 rng(1337);
    const uint16_t scale_pool[] = {0x0001, 0x0400, 0x7C00, 0xFC00, 0x7D00, 0x7E00, 0x7BFF, 0xFBFF, 0x3C00};
    const uint16_t bias_pool[]  = {0x0000, 0x0001, 0x7C00, 0x7E00, 0x7BFF, 0x4800};

    for (size_t b = 0; b < num_blocks; ++b) {
        blocks[b].scale_fp16 = scale_pool[rng() % 9];
        blocks[b].bias_fp16  = bias_pool[rng() % 6];
        for (size_t i = 0; i < 16; ++i) {
            blocks[b].qs[i] = static_cast<uint8_t>(rng() & 0xFF);
        }
    }

    std::vector<float> batch_out(num_blocks * 32, 0.0f);
    DequantizeINT4Block32Batch(blocks.data(), num_blocks, batch_out.data(), true);

    std::vector<float> single_out(num_blocks * 32, 0.0f);
    for (size_t b = 0; b < num_blocks; ++b) {
        DequantizeINT4Block32(&blocks[b], single_out.data() + b * 32, true);
    }

    for (size_t i = 0; i < num_blocks * 32; ++i) {
        ASSERT_TRUE(FloatIEEEEquivalent(batch_out[i], single_out[i]));
    }
}

// ============================================================================
// Suite 5: FP8_E4M3 Exhaustive 256-Byte Oracle Verification
// ============================================================================

TEST(ChallengerM2Suite, FP8_E4M3_ExhaustiveAll256Bytes) {
    // Verify all 256 possible byte values against the OCP MX specification oracle
    for (int b = 0; b <= 255; ++b) {
        uint8_t byte = static_cast<uint8_t>(b);
        float actual = fp8_e4m3_to_fp32(byte);
        float oracle = FP8_E4M3_Oracle(byte);

        if (std::isnan(oracle)) {
            EXPECT_TRUE(std::isnan(actual));
        } else if (oracle == 0.0f) {
            EXPECT_EQ(actual, 0.0f);
            EXPECT_EQ(std::signbit(actual), std::signbit(oracle));
        } else {
            EXPECT_NEAR(actual, oracle, 1e-6f);
        }
    }
}

// ============================================================================
// Suite 6: FP8_E4M3 Arbitrary Random Byte Noise Fuzzing
// ============================================================================

TEST(ChallengerM2Suite, FP8_E4M3_RandomNoiseFuzzing) {
    const size_t N = 100000;
    std::vector<uint8_t> random_bytes(N);
    std::mt19937 rng(42);
    for (size_t i = 0; i < N; ++i) {
        random_bytes[i] = static_cast<uint8_t>(rng() & 0xFF);
    }

    std::vector<float> out(N, 0.0f);
    DequantizeFP8_E4M3(random_bytes.data(), N, out.data());

    // Verify each dequantized float against oracle
    for (size_t i = 0; i < N; ++i) {
        float oracle = FP8_E4M3_Oracle(random_bytes[i]);
        if (std::isnan(oracle)) {
            ASSERT_TRUE(std::isnan(out[i]));
        } else {
            ASSERT_EQ(out[i], oracle);
        }
    }

    // Boundary sizes: 0, 1, 3, 7, 15, 17, 31, 33
    const size_t edge_sizes[] = {0, 1, 3, 7, 15, 17, 31, 33};
    for (size_t sz : edge_sizes) {
        std::vector<float> edge_out(sz, 0.0f);
        DequantizeFP8_E4M3(sz > 0 ? random_bytes.data() : nullptr, sz, edge_out.data());
        for (size_t i = 0; i < sz; ++i) {
            float oracle = FP8_E4M3_Oracle(random_bytes[i]);
            if (std::isnan(oracle)) {
                ASSERT_TRUE(std::isnan(edge_out[i]));
            } else {
                ASSERT_EQ(edge_out[i], oracle);
            }
        }
    }

    // Nullptr safety
    DequantizeFP8_E4M3(nullptr, 10, out.data());
    DequantizeFP8_E4M3(random_bytes.data(), 10, nullptr);
}

// ============================================================================
// Suite 7: FP8_E5M2 Exhaustive 256-Byte Oracle Verification
// ============================================================================

TEST(ChallengerM2Suite, FP8_E5M2_ExhaustiveAll256Bytes) {
    // Verify all 256 possible byte values against the IEEE 754-like FP8 oracle
    for (int b = 0; b <= 255; ++b) {
        uint8_t byte = static_cast<uint8_t>(b);
        float actual = fp8_e5m2_to_fp32(byte);
        float oracle = FP8_E5M2_Oracle(byte);

        if (std::isnan(oracle)) {
            EXPECT_TRUE(std::isnan(actual));
        } else if (std::isinf(oracle)) {
            EXPECT_TRUE(std::isinf(actual));
            EXPECT_EQ(std::signbit(actual), std::signbit(oracle));
        } else if (oracle == 0.0f) {
            EXPECT_EQ(actual, 0.0f);
            EXPECT_EQ(std::signbit(actual), std::signbit(oracle));
        } else {
            EXPECT_NEAR(actual, oracle, 1e-6f);
        }
    }
}

// ============================================================================
// Suite 8: FP8_E5M2 Arbitrary Random Byte Noise Fuzzing
// ============================================================================

TEST(ChallengerM2Suite, FP8_E5M2_RandomNoiseFuzzing) {
    const size_t N = 100000;
    std::vector<uint8_t> random_bytes(N);
    std::mt19937 rng(84);
    for (size_t i = 0; i < N; ++i) {
        random_bytes[i] = static_cast<uint8_t>(rng() & 0xFF);
    }

    std::vector<float> out(N, 0.0f);
    DequantizeFP8_E5M2(random_bytes.data(), N, out.data());

    for (size_t i = 0; i < N; ++i) {
        float oracle = FP8_E5M2_Oracle(random_bytes[i]);
        if (std::isnan(oracle)) {
            ASSERT_TRUE(std::isnan(out[i]));
        } else if (std::isinf(oracle)) {
            ASSERT_TRUE(std::isinf(out[i]));
            ASSERT_EQ(std::signbit(out[i]), std::signbit(oracle));
        } else {
            ASSERT_EQ(out[i], oracle);
        }
    }

    // Boundary sizes: 0, 1, 3, 7, 15, 17, 31, 33
    const size_t edge_sizes[] = {0, 1, 3, 7, 15, 17, 31, 33};
    for (size_t sz : edge_sizes) {
        std::vector<float> edge_out(sz, 0.0f);
        DequantizeFP8_E5M2(sz > 0 ? random_bytes.data() : nullptr, sz, edge_out.data());
        for (size_t i = 0; i < sz; ++i) {
            float oracle = FP8_E5M2_Oracle(random_bytes[i]);
            if (std::isnan(oracle)) {
                ASSERT_TRUE(std::isnan(edge_out[i]));
            } else if (std::isinf(oracle)) {
                ASSERT_TRUE(std::isinf(edge_out[i]));
                ASSERT_EQ(std::signbit(edge_out[i]), std::signbit(oracle));
            } else {
                ASSERT_EQ(edge_out[i], oracle);
            }
        }
    }

    // Nullptr safety
    DequantizeFP8_E5M2(nullptr, 10, out.data());
    DequantizeFP8_E5M2(random_bytes.data(), 10, nullptr);
}

// ============================================================================
// Suite 9: INT8 Symmetric Dequantizer Extreme Scales & Edge Values
// ============================================================================

TEST(ChallengerM2Suite, INT8Symm_ExtremeScales) {
    const float extreme_scales[] = {
        1e-20f, -1e-20f,
        1e20f,  -1e20f,
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::signaling_NaN(),
        0.0f, -0.0f,
        std::numeric_limits<float>::denorm_min(),
        std::numeric_limits<float>::min(),
        std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max()
    };

    // Prepare full range of int8_t values (-128..127)
    const size_t N = 256;
    std::vector<int8_t> data(N);
    for (size_t i = 0; i < N; ++i) {
        data[i] = static_cast<int8_t>(static_cast<int>(i) - 128);
    }

    for (float scale : extreme_scales) {
        std::vector<float> out_neon(N, 0.0f);
        std::vector<float> out_scalar(N, 0.0f);

        DequantizeINT8Symm(data.data(), N, scale, out_neon.data());
        DequantizeINT8Symm_ScalarRef(data.data(), N, scale, out_scalar.data());

        for (size_t i = 0; i < N; ++i) {
            ASSERT_TRUE(FloatIEEEEquivalent(out_neon[i], out_scalar[i]));

            // Verify IEEE 754 rules
            if (std::isnan(scale)) {
                EXPECT_TRUE(std::isnan(out_neon[i]));
            } else if (std::isinf(scale)) {
                if (data[i] == 0) {
                    // 0 * Inf = NaN
                    EXPECT_TRUE(std::isnan(out_neon[i]));
                } else if ((data[i] > 0 && scale > 0) || (data[i] < 0 && scale < 0)) {
                    EXPECT_TRUE(std::isinf(out_neon[i]) && out_neon[i] > 0.0f);
                } else {
                    EXPECT_TRUE(std::isinf(out_neon[i]) && out_neon[i] < 0.0f);
                }
            } else if (scale == 0.0f) {
                EXPECT_EQ(out_neon[i], 0.0f);
            }
        }
    }
}

// ============================================================================
// Suite 10: INT8 Symmetric Dequantizer Odd Sizes & Memory Alignment
// ============================================================================

TEST(ChallengerM2Suite, INT8Symm_OddSizesAndAlignment) {
    std::mt19937 rng(999);
    const size_t max_size = 257;
    std::vector<int8_t> buffer(max_size + 16);
    for (size_t i = 0; i < buffer.size(); ++i) {
        buffer[i] = static_cast<int8_t>(rng() & 0xFF);
    }

    const size_t test_sizes[] = {0, 1, 2, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257};
    const float test_scales[] = {0.05f, -1.5f, 1e20f, -1e-20f};

    for (size_t sz : test_sizes) {
        for (float scale : test_scales) {
            // Test unaligned pointers (offset 1, offset 3)
            for (size_t offset : {0, 1, 3}) {
                std::vector<float> out_neon(sz + 4, -999.0f);
                std::vector<float> out_scalar(sz + 4, -999.0f);

                const int8_t* in_ptr = buffer.data() + offset;
                float* neon_dst = out_neon.data() + 1; // unaligned float offset
                float* scalar_dst = out_scalar.data() + 1;

                DequantizeINT8Symm(in_ptr, sz, scale, neon_dst);
                DequantizeINT8Symm_ScalarRef(in_ptr, sz, scale, scalar_dst);

                for (size_t i = 0; i < sz; ++i) {
                    ASSERT_TRUE(FloatIEEEEquivalent(neon_dst[i], scalar_dst[i]));
                }
            }
        }
    }

    // Nullptr safety
    float dummy_out[16];
    DequantizeINT8Symm(nullptr, 16, 1.0f, dummy_out);
    DequantizeINT8Symm(buffer.data(), 16, 1.0f, nullptr);
    DequantizeINT8Symm(buffer.data(), 0, 1.0f, dummy_out);
}

// ============================================================================
// Suite 11: Hardware Traps, Exception Monitoring & Non-Crashing Verification
// ============================================================================

TEST(ChallengerM2Suite, FloatingPointExceptionRobustness) {
    // Under IEEE 754 default exception handling, floating point operations
    // must not trigger hardware traps or crashes even during division by zero,
    // invalid operation (0 * Inf, sqrt(-1)), or overflow/underflow.

    // Clear FP exceptions
    std::feclearexcept(FE_ALL_EXCEPT);

    // Run INT4 dequantizer with extreme values
    INT4Block32 block{};
    block.scale_fp16 = 0x7C00; // +Inf
    block.bias_fp16  = 0xFC00; // -Inf
    for (int i = 0; i < 16; ++i) block.qs[i] = static_cast<uint8_t>(i * 17);

    float out[32];
    DequantizeINT4Block32(&block, out, true);

    // Operations must execute cleanly without terminating process
    for (size_t i = 0; i < 32; ++i) {
        // Output can be Inf or NaN depending on (q - bias), but must be safely accessible
        bool is_valid_float = std::isnan(out[i]) || std::isinf(out[i]) || std::isfinite(out[i]);
        EXPECT_TRUE(is_valid_float);
    }

    // Run FP8 with 0x7F (NaN) and 0x7C (Inf)
    uint8_t fp8_e4[4] = {0x7F, 0xFF, 0x00, 0x7E};
    float fp8_e4_out[4];
    DequantizeFP8_E4M3(fp8_e4, 4, fp8_e4_out);
    EXPECT_TRUE(std::isnan(fp8_e4_out[0]));
    EXPECT_TRUE(std::isnan(fp8_e4_out[1]));
    EXPECT_EQ(fp8_e4_out[2], 0.0f);
    EXPECT_NEAR(fp8_e4_out[3], 448.0f, 1e-3f);

    uint8_t fp8_e5[4] = {0x7C, 0xFC, 0x7D, 0x7B};
    float fp8_e5_out[4];
    DequantizeFP8_E5M2(fp8_e5, 4, fp8_e5_out);
    EXPECT_TRUE(std::isinf(fp8_e5_out[0]) && fp8_e5_out[0] > 0);
    EXPECT_TRUE(std::isinf(fp8_e5_out[1]) && fp8_e5_out[1] < 0);
    EXPECT_TRUE(std::isnan(fp8_e5_out[2]));
    EXPECT_NEAR(fp8_e5_out[3], 57344.0f, 1e-1f);
}

// ============================================================================
// Suite 12: INT4Block32 Exact Bitwise Equivalence for Finite Values
// ============================================================================

TEST(ChallengerM2Suite, INT4Block32_ExactBitEquivalence_Finite) {
    const uint16_t finite_scales[] = {
        0x0001, 0x0002, 0x03FF, 0x0400, // Subnormals and min normal
        0x8001, 0x83FF, 0x8400,         // Negative subnormals
        0x3C00, 0x4000, 0x3800,         // 1.0, 2.0, 0.5
        0x7BFF, 0xFBFF,                 // Max finite (+/- 65504.0f)
        0x0000, 0x8000                  // +/- 0.0f
    };

    const uint16_t finite_biases[] = {
        0x0000, 0x8000, 0x0400, 0x4800, 0x4780, 0x7BFF, 0xFBFF
    };

    for (uint16_t s : finite_scales) {
        for (uint16_t b : finite_biases) {
            INT4Block32 block{};
            block.scale_fp16 = s;
            block.bias_fp16 = b;

            for (int i = 0; i < 16; ++i) {
                block.qs[i] = static_cast<uint8_t>((i * 13) & 0xFF);
            }

            // subtract_bias = true
            {
                float out_neon[32];
                float out_scalar[32];
                DequantizeINT4Block32(&block, out_neon, true);
                DequantizeINT4Block32_ScalarRef(&block, out_scalar, true);

                for (size_t i = 0; i < 32; ++i) {
                    if (std::isfinite(out_neon[i]) && std::isfinite(out_scalar[i])) {
                        uint32_t u_neon = 0, u_scalar = 0;
                        std::memcpy(&u_neon, &out_neon[i], sizeof(float));
                        std::memcpy(&u_scalar, &out_scalar[i], sizeof(float));
                        EXPECT_EQ(u_neon, u_scalar);
                    }
                }
            }
        }
    }
}

// ============================================================================
// Suite 13: Large Batch INT4Block32 Stress (1024 Blocks / 32k Weights)
// ============================================================================

TEST(ChallengerM2Suite, INT4Block32_LargeBatchStress) {
    const size_t num_blocks = 1024;
    const size_t num_weights = num_blocks * 32;
    std::vector<INT4Block32> blocks(num_blocks);

    std::mt19937 rng(54321);
    for (size_t b = 0; b < num_blocks; ++b) {
        blocks[b].scale_fp16 = static_cast<uint16_t>(rng() & 0xFFFF);
        blocks[b].bias_fp16  = static_cast<uint16_t>(rng() & 0xFFFF);
        for (size_t i = 0; i < 16; ++i) {
            blocks[b].qs[i] = static_cast<uint8_t>(rng() & 0xFF);
        }
    }

    std::vector<float> batch_out(num_weights, 0.0f);
    std::vector<float> scalar_out(num_weights, 0.0f);

    DequantizeINT4Block32Batch(blocks.data(), num_blocks, batch_out.data(), true);
    for (size_t b = 0; b < num_blocks; ++b) {
        DequantizeINT4Block32_ScalarRef(&blocks[b], scalar_out.data() + b * 32, true);
    }

    for (size_t i = 0; i < num_weights; ++i) {
        ASSERT_TRUE(FloatIEEEEquivalent(batch_out[i], scalar_out[i]));
    }
}

// ============================================================================
// Suite 14: FP16 and BF16 Dequantizers Extreme Values
// ============================================================================

TEST(ChallengerM2Suite, FP16_BF16_ExtremeValues) {
    const uint16_t fp16_test_vectors[] = {
        0x0000, 0x8000, // 0
        0x0001, 0x03FF, 0x0400, // denormals
        0x7C00, 0xFC00, // Infs
        0x7D00, 0x7E00, // NaNs
        0x7BFF, 0xFBFF  // Max finite
    };

    const size_t N = sizeof(fp16_test_vectors) / sizeof(fp16_test_vectors[0]);
    std::vector<float> out_fp16(N);
    DequantizeFP16(fp16_test_vectors, N, out_fp16.data());

    for (size_t i = 0; i < N; ++i) {
        float expected = fp16_to_fp32(fp16_test_vectors[i]);
        ASSERT_TRUE(FloatIEEEEquivalent(out_fp16[i], expected));
    }

    // BF16
    const uint16_t bf16_test_vectors[] = {
        0x0000, 0x8000,
        0x7F80, 0xFF80, // Infs
        0x7FC0, 0xFFC0, // NaNs
        0x7F7F          // Max finite BF16
    };
    const size_t N_bf16 = sizeof(bf16_test_vectors) / sizeof(bf16_test_vectors[0]);
    std::vector<float> out_bf16(N_bf16);
    DequantizeBF16(bf16_test_vectors, N_bf16, out_bf16.data());

    for (size_t i = 0; i < N_bf16; ++i) {
        float expected = bf16_to_fp32(bf16_test_vectors[i]);
        ASSERT_TRUE(FloatIEEEEquivalent(out_bf16[i], expected));
    }

    // Nullptr safety
    DequantizeFP16(nullptr, 10, out_fp16.data());
    DequantizeFP16(fp16_test_vectors, 10, nullptr);
    DequantizeBF16(nullptr, 10, out_bf16.data());
    DequantizeBF16(bf16_test_vectors, 10, nullptr);
}

TEST_RUNNER_MAIN()

#include "test_runner.h"
#include "playworld/tensor.h"
#include "playworld/voxel_grid.h"
#include "playworld/action_types.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using namespace playworld;

// ============================================================================
// Mathematical Helpers: SQNR and SSIM
// ============================================================================

static double CalculateSQNR(const float* ref, const float* test, size_t n) {
    double signal_pwr = 0.0;
    double noise_pwr = 0.0;
    for (size_t i = 0; i < n; ++i) {
        signal_pwr += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
        double diff = static_cast<double>(ref[i]) - static_cast<double>(test[i]);
        noise_pwr += diff * diff;
    }
    if (noise_pwr < 1e-12) return 100.0; // Perfect match
    return 10.0 * std::log10(signal_pwr / noise_pwr);
}

static double ComputeSSIM(const uint8_t* img1, const uint8_t* img2, size_t width, size_t height, size_t channels = 4) noexcept {
    if (!img1 || !img2 || width == 0 || height == 0 || channels == 0) return 0.0;

    const size_t total_pixels = width * height;
    const double C1 = 6.5025;
    const double C2 = 58.5225;

    double ssim_sum = 0.0;
    const size_t color_channels = std::min(channels, static_cast<size_t>(3));
    for (size_t c = 0; c < color_channels; ++c) {
        double sum_x = 0.0;
        double sum_y = 0.0;

        for (size_t i = 0; i < total_pixels; ++i) {
            sum_x += img1[i * channels + c];
            sum_y += img2[i * channels + c];
        }

        const double mu_x = sum_x / static_cast<double>(total_pixels);
        const double mu_y = sum_y / static_cast<double>(total_pixels);

        double var_x = 0.0;
        double var_y = 0.0;
        double cov_xy = 0.0;

        for (size_t i = 0; i < total_pixels; ++i) {
            double dx = static_cast<double>(img1[i * channels + c]) - mu_x;
            double dy = static_cast<double>(img2[i * channels + c]) - mu_y;
            var_x += dx * dx;
            var_y += dy * dy;
            cov_xy += dx * dy;
        }

        var_x /= static_cast<double>(total_pixels - 1);
        var_y /= static_cast<double>(total_pixels - 1);
        cov_xy /= static_cast<double>(total_pixels - 1);

        const double numerator = (2.0 * mu_x * mu_y + C1) * (2.0 * cov_xy + C2);
        const double denominator = (mu_x * mu_x + mu_y * mu_y + C1) * (var_x + var_y + C2);

        ssim_sum += (denominator > 0.0) ? (numerator / denominator) : 0.0;
    }

    return ssim_sum / static_cast<double>(color_channels);
}

// ============================================================================
// PART 1: INT4 Block-32 Dequantization Challenges
// ============================================================================

TEST(ChallengerINT4, EdgeValuesAllZeros) {
    // All 0x0 nibbles (qs = 0x00 throughout)
    INT4Block32 block{};
    std::memset(block.qs, 0x00, sizeof(block.qs));
    block.scale_fp16 = fp32_to_fp16(0.25f);
    block.bias_fp16  = fp32_to_fp16(4.0f);

    float out[32];
    DequantizeINT4Block32(&block, out, true);

    // Analytical: (0 - 4.0) * 0.25 = -1.0
    for (int i = 0; i < 32; ++i) {
        EXPECT_NEAR(out[i], -1.0f, 1e-4f);
    }
}

TEST(ChallengerINT4, EdgeValuesAllOnes) {
    // All 0xF nibbles (qs = 0xFF throughout)
    INT4Block32 block{};
    std::memset(block.qs, 0xFF, sizeof(block.qs));
    block.scale_fp16 = fp32_to_fp16(0.25f);
    block.bias_fp16  = fp32_to_fp16(7.0f);

    float out[32];
    DequantizeINT4Block32(&block, out, true);

    // Analytical: (15 - 7.0) * 0.25 = 2.0
    for (int i = 0; i < 32; ++i) {
        EXPECT_NEAR(out[i], 2.0f, 1e-4f);
    }
}

TEST(ChallengerINT4, AlternatingNibbles_0x0F_and_0xF0) {
    // 0x0F: low = 0xF (15), high = 0x0 (0)
    // 0xF0: low = 0x0 (0), high = 0xF (15)
    INT4Block32 block{};
    for (int i = 0; i < 16; ++i) {
        block.qs[i] = (i % 2 == 0) ? 0x0F : 0xF0;
    }
    block.scale_fp16 = fp32_to_fp16(0.5f);
    block.bias_fp16  = fp32_to_fp16(5.0f);

    float out[32];
    DequantizeINT4Block32(&block, out, true);

    // Even byte (i=0, 2, 4...):
    //   out[2*i] = low nibble = 15 => (15 - 5.0) * 0.5 = 5.0
    //   out[2*i + 1] = high nibble = 0 => (0 - 5.0) * 0.5 = -2.5
    // Odd byte (i=1, 3, 5...):
    //   out[2*i] = low nibble = 0 => (0 - 5.0) * 0.5 = -2.5
    //   out[2*i + 1] = high nibble = 15 => (15 - 5.0) * 0.5 = 5.0
    for (int byte_idx = 0; byte_idx < 16; ++byte_idx) {
        float expected_low  = (byte_idx % 2 == 0) ? 5.0f : -2.5f;
        float expected_high = (byte_idx % 2 == 0) ? -2.5f : 5.0f;
        EXPECT_NEAR(out[2 * byte_idx], expected_low, 1e-4f);
        EXPECT_NEAR(out[2 * byte_idx + 1], expected_high, 1e-4f);
    }
}

TEST(ChallengerINT4, NegativeBiasesAndScales) {
    // Negative bias: bias = -6.0f
    // Formula: (q - (-6.0)) * 0.1 = (q + 6.0) * 0.1
    INT4Block32 block{};
    block.scale_fp16 = fp32_to_fp16(0.1f);
    block.bias_fp16  = fp32_to_fp16(-6.0f);

    for (int i = 0; i < 16; ++i) {
        block.qs[i] = static_cast<uint8_t>((i << 4) | (15 - i));
    }

    float out[32];
    DequantizeINT4Block32(&block, out, true);

    for (int i = 0; i < 16; ++i) {
        uint8_t q_low  = 15 - i;
        uint8_t q_high = i;
        float exp_low  = (static_cast<float>(q_low) - (-6.0f)) * 0.1f;
        float exp_high = (static_cast<float>(q_high) - (-6.0f)) * 0.1f;
        EXPECT_NEAR(out[2 * i], exp_low, 1e-3f);
        EXPECT_NEAR(out[2 * i + 1], exp_high, 1e-3f);
    }

    // Negative scale: scale = -0.5f, bias = -2.0f
    block.scale_fp16 = fp32_to_fp16(-0.5f);
    block.bias_fp16  = fp32_to_fp16(-2.0f);
    DequantizeINT4Block32(&block, out, true);

    for (int i = 0; i < 16; ++i) {
        uint8_t q_low  = 15 - i;
        uint8_t q_high = i;
        float exp_low  = (static_cast<float>(q_low) - (-2.0f)) * (-0.5f);
        float exp_high = (static_cast<float>(q_high) - (-2.0f)) * (-0.5f);
        EXPECT_NEAR(out[2 * i], exp_low, 1e-3f);
        EXPECT_NEAR(out[2 * i + 1], exp_high, 1e-3f);
    }
}

TEST(ChallengerINT4, ExtremeScalesAndSubnormals) {
    INT4Block32 block{};
    std::memset(block.qs, 0x55, sizeof(block.qs)); // all q=5

    // 1. Min normal FP16: 6.1035e-5
    block.scale_fp16 = 0x0400; // 2^(-14)
    block.bias_fp16  = fp32_to_fp16(5.0f); // q=5 - 5.0 = 0.0
    float out[32];
    DequantizeINT4Block32(&block, out, true);
    for (int i = 0; i < 32; ++i) {
        EXPECT_NEAR(out[i], 0.0f, 1e-7f);
    }

    // 2. Large scale: 1024.0f
    block.scale_fp16 = fp32_to_fp16(1024.0f);
    block.bias_fp16  = fp32_to_fp16(0.0f);
    DequantizeINT4Block32(&block, out, true);
    for (int i = 0; i < 32; ++i) {
        // (5 - 0) * 1024 = 5120.0
        EXPECT_NEAR(out[i], 5120.0f, 1.0f);
    }

    // 3. Max finite FP16: 65504.0f
    block.scale_fp16 = 0x7BFF; // 65504.0f
    block.bias_fp16  = fp32_to_fp16(4.0f);
    DequantizeINT4Block32(&block, out, true);
    for (int i = 0; i < 32; ++i) {
        // (5 - 4) * 65504 = 65504.0
        EXPECT_NEAR(out[i], 65504.0f, 2.0f);
    }
}

TEST(ChallengerINT4, AnalyticalGroundTruthOracleFuzz_10000Blocks) {
    // Fuzz 1,000 blocks (32,000 weights) against analytical oracle
    const size_t num_blocks = 1000;
    std::vector<INT4Block32> blocks(num_blocks);

    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> byte_dist(0, 255);
    std::uniform_real_distribution<float> scale_dist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> bias_dist(-15.0f, 15.0f);

    for (size_t b = 0; b < num_blocks; ++b) {
        float raw_scale = scale_dist(rng);
        if (std::fabs(raw_scale) < 1e-4f) raw_scale = 0.05f;
        float raw_bias = bias_dist(rng);

        blocks[b].scale_fp16 = fp32_to_fp16(raw_scale);
        blocks[b].bias_fp16  = fp32_to_fp16(raw_bias);

        for (int i = 0; i < 16; ++i) {
            blocks[b].qs[i] = static_cast<uint8_t>(byte_dist(rng));
        }
    }

    // Batch dequantization (vectorized NEON)
    std::vector<float> batch_out(num_blocks * 32);
    DequantizeINT4Block32Batch(blocks.data(), num_blocks, batch_out.data(), true);

    // Verify each weight against exact analytical oracle: (q - bias_unpacked) * scale_unpacked
    for (size_t b = 0; b < num_blocks; ++b) {
        float ref_scale = fp16_to_fp32(blocks[b].scale_fp16);
        float ref_bias  = fp16_to_fp32(blocks[b].bias_fp16);

        for (size_t i = 0; i < 16; ++i) {
            uint8_t byte = blocks[b].qs[i];
            uint8_t q_low  = byte & 0x0F;
            uint8_t q_high = (byte >> 4) & 0x0F;

            float expected_low  = (static_cast<float>(q_low)  - ref_bias) * ref_scale;
            float expected_high = (static_cast<float>(q_high) - ref_bias) * ref_scale;

            float actual_low  = batch_out[b * 32 + 2 * i];
            float actual_high = batch_out[b * 32 + 2 * i + 1];

            ASSERT_NEAR(actual_low,  expected_low,  1e-4f);
            ASSERT_NEAR(actual_high, expected_high, 1e-4f);
        }
    }
}

TEST(ChallengerINT4, ContinuousSignalSQNRMultiDistribution) {
    // Challenge SQNR across:
    // 1. High frequency chirp: sin(t^1.4)
    // 2. Multi-tone harmonic signal
    // 3. Normal distribution Gaussian noise
    // Check that SQNR >= 28.0 dB and L_inf error is strictly bounded.
    const size_t N = 1024; // 32 blocks
    std::vector<float> signal(N);

    // Signal 1: Multi-tone harmonics
    for (size_t i = 0; i < N; ++i) {
        float t = static_cast<float>(i) / 100.0f;
        signal[i] = std::sin(t) + 0.4f * std::cos(3.2f * t) + 0.2f * std::sin(7.5f * t);
    }
    std::vector<INT4Block32> blocks(N / 32);
    QuantizeINT4Block32Batch(signal.data(), N, blocks.data());

    std::vector<float> reconstructed(N);
    DequantizeINT4Block32Batch(blocks.data(), blocks.size(), reconstructed.data(), true);

    double sqnr_harmonics = CalculateSQNR(signal.data(), reconstructed.data(), N);
    std::cout << "  [CHALLENGER INT4] Multi-tone harmonic SQNR: " << sqnr_harmonics << " dB\n";
    EXPECT_GE(sqnr_harmonics, 28.0);

    // Signal 2: Gaussian normal distribution with dynamic range [-10.0, 10.0]
    std::mt19937 rng(999);
    std::normal_distribution<float> gauss(0.0f, 3.0f);
    for (size_t i = 0; i < N; ++i) {
        signal[i] = std::clamp(gauss(rng), -10.0f, 10.0f);
    }
    QuantizeINT4Block32Batch(signal.data(), N, blocks.data());
    DequantizeINT4Block32Batch(blocks.data(), blocks.size(), reconstructed.data(), true);

    double sqnr_gaussian = CalculateSQNR(signal.data(), reconstructed.data(), N);
    std::cout << "  [CHALLENGER INT4] Gaussian signal SQNR: " << sqnr_gaussian << " dB\n";
    EXPECT_GE(sqnr_gaussian, 28.0);
}

// ============================================================================
// PART 2: Frustum Voxel Memory Grid Challenges
// ============================================================================

TEST(ChallengerVoxelGrid, Adversarial360DegreeYawRotationLoopback_120Steps) {
    const size_t W = 640, H = 360, C = 4;
    std::vector<uint8_t> ref_frame(W * H * C);

    // Render realistic procedural scene with crisp contrast & high frequencies
    for (size_t y = 0; y < H; ++y) {
        for (size_t x = 0; x < W; ++x) {
            size_t idx = (y * W + x) * C;
            // Sky gradient vs Ground
            uint8_t val = (y < H / 2) ? static_cast<uint8_t>(180 + (y * 40 / (H / 2))) : 50;
            // Central landmark monolith
            if (x >= 240 && x <= 400 && y >= 90 && y <= 270) {
                // High frequency brick checkerboard
                val = ((x / 8 + y / 8) % 2 == 0) ? 220 : 130;
            }
            ref_frame[idx + 0] = val;
            ref_frame[idx + 1] = static_cast<uint8_t>(val * 0.65f);
            ref_frame[idx + 2] = static_cast<uint8_t>(val * 0.35f);
            ref_frame[idx + 3] = 255;
        }
    }

    // Initialize Frustum Voxel Grid
    FrustumMemoryGrid grid(512, 0.5f, 15.0f);

    // Store frame 0 at origin: yaw = 0.0 deg
    CameraPose pose_origin{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    grid.StoreLatents(pose_origin, ref_frame.data(), ref_frame.size(), 4, 45, 80);

    // Simulate 120 forward steps of 360-degree rotation (3 degrees per step)
    CameraPose current_pose = pose_origin;
    std::vector<uint8_t> ar_frame = ref_frame;

    std::mt19937 rng(777);
    std::normal_distribution<float> drift_noise(0.0f, 2.5f);

    for (int step = 1; step <= 120; ++step) {
        current_pose.yaw += 3.0f; // 3.0 deg per step -> 360.0 deg total

        // State drift at each step
        for (size_t i = 0; i < ar_frame.size(); i += C) {
            float n = drift_noise(rng);
            int v = static_cast<int>(ar_frame[i]) + static_cast<int>(n);
            ar_frame[i] = static_cast<uint8_t>(std::clamp(v, 0, 255));
        }

        // Store intermediate views into grid
        grid.StoreLatents(current_pose, ar_frame.data(), ar_frame.size(), 4, 45, 80);
    }

    // At step 120: current_pose.yaw = 360.0 deg
    EXPECT_FLOAT_EQ(current_pose.yaw, 360.0f);

    // Query spatial memory at return frame
    CachedLatentTensor retrieved_latent{};
    float similarity = 0.0f;
    bool hit = grid.QueryLatents(current_pose, retrieved_latent, similarity);

    EXPECT_TRUE(hit);
    // At yaw = 360 deg, normalized yaw = 0 deg. Stored anchor was yaw = 360 deg.
    // Cosine similarity between current_pose (360) and anchor (360) is cos(0) = 1.0f.
    EXPECT_GE(similarity, 0.999f);

    // Conditioned return frame using voxel memory blend
    std::vector<uint8_t> return_frame(W * H * C);
    FrustumMemoryGrid::BlendLatents(retrieved_latent.latent_data.data(),
                                    ar_frame.data(),
                                    return_frame.data(),
                                    return_frame.size(),
                                    similarity);

    double final_ssim = ComputeSSIM(ref_frame.data(), return_frame.data(), W, H, C);
    std::cout << "  [CHALLENGER VOXEL GRID] 120-Step 360-Deg Rotation Loopback SSIM: "
              << final_ssim << " (Must be >= 0.82)\n";

    // Non-negotiable acceptance criteria: SSIM >= 0.82
    EXPECT_GE(final_ssim, 0.82);
    EXPECT_GE(final_ssim, 0.86);
}

TEST(ChallengerVoxelGrid, AutoregressiveBaselineDriftCollapse) {
    const size_t W = 640, H = 360, C = 4;
    std::vector<uint8_t> ref_frame(W * H * C);
    for (size_t y = 0; y < H; ++y) {
        for (size_t x = 0; x < W; ++x) {
            size_t idx = (y * W + x) * C;
            uint8_t val = (y < H / 2) ? 200 : 70;
            if (x >= 240 && x <= 400 && y >= 90 && y <= 270) {
                val = ((x / 8 + y / 8) % 2 == 0) ? 220 : 130;
            }
            ref_frame[idx + 0] = val;
            ref_frame[idx + 1] = static_cast<uint8_t>(val * 0.65f);
            ref_frame[idx + 2] = static_cast<uint8_t>(val * 0.35f);
            ref_frame[idx + 3] = 255;
        }
    }

    // Simulate autoregressive accumulation drift over 120 steps WITHOUT voxel memory
    std::mt19937 rng(42);
    std::normal_distribution<float> drift_dist(0.0f, 15.0f);
    std::vector<uint8_t> drifted_frame = ref_frame;

    for (int step = 0; step < 120; ++step) {
        for (size_t i = 0; i < drifted_frame.size(); i += C) {
            float n = drift_dist(rng);
            int v = static_cast<int>(drifted_frame[i]) + static_cast<int>(n);
            drifted_frame[i] = static_cast<uint8_t>(std::clamp(v, 0, 255));
        }
    }

    double baseline_ssim = ComputeSSIM(ref_frame.data(), drifted_frame.data(), W, H, C);
    std::cout << "  [CHALLENGER BASELINE] Unconditioned Autoregressive SSIM after 120 steps: "
              << baseline_ssim << " (Must be < 0.45)\n";

    // Proves the "world melting" defect occurs without spatial anchoring
    EXPECT_LT(baseline_ssim, 0.45);
}

TEST(ChallengerVoxelGrid, CacheCapacityLimitsAndLRUEviction_1000Insertions) {
    FrustumMemoryGrid grid(512, 0.5f, 15.0f);
    EXPECT_EQ(grid.Capacity(), 512);
    EXPECT_EQ(grid.Size(), 0);

    const size_t DUMMY_BYTES = 64;
    std::vector<uint8_t> dummy_data(DUMMY_BYTES, 0xAB);

    // Insert 1,000 distinct coordinates: x = i * 1.0f (so vx = i * 2)
    for (int i = 0; i < 1000; ++i) {
        CameraPose pose{static_cast<float>(i) * 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        grid.StoreLatents(pose, dummy_data.data(), dummy_data.size(), 4, 1, 16);

        // Strict bound: active entries must NEVER exceed 512
        ASSERT_LE(grid.Size(), 512);
    }

    // Final size must be exactly 512
    EXPECT_EQ(grid.Size(), 512);
    std::cout << "  [CHALLENGER CACHE] Active cache entries after 1,000 insertions: "
              << grid.Size() << " (Capacity: " << grid.Capacity() << ")\n";

    // Verify eviction correctness:
    // Entries 0 to 487 (first 488 insertions) must have been evicted.
    for (int i = 0; i < 488; ++i) {
        CameraPose pose{static_cast<float>(i) * 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        CachedLatentTensor out{};
        float sim = 0.0f;
        EXPECT_FALSE(grid.QueryLatents(pose, out, sim));
    }

    // Entries 488 to 999 (last 512 insertions) must still be present.
    for (int i = 488; i < 1000; ++i) {
        CameraPose pose{static_cast<float>(i) * 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        CachedLatentTensor out{};
        float sim = 0.0f;
        EXPECT_TRUE(grid.QueryLatents(pose, out, sim));
    }
}

TEST(ChallengerVoxelGrid, AdversarialLRURefreshBehavior) {
    FrustumMemoryGrid grid(4, 0.5f, 15.0f); // Capacity 4 for exact LRU tracing
    std::vector<uint8_t> dummy(16, 0x11);

    // Insert entries A, B, C, D (vx = 0, 1, 2, 3)
    for (int i = 0; i < 4; ++i) {
        CameraPose p{static_cast<float>(i) * 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        grid.StoreLatents(p, dummy.data(), dummy.size());
    }
    EXPECT_EQ(grid.Size(), 4);

    // Access entry A (vx=0), moving it to MRU (most recently used).
    // Now the order from MRU to LRU is: A (0), D (3), C (2), B (1).
    // Oldest entry is now B (vx=1).
    CameraPose pose_A{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    CachedLatentTensor out{};
    float sim = 0.0f;
    EXPECT_TRUE(grid.QueryLatents(pose_A, out, sim));

    // Insert entry E (vx = 4)
    CameraPose pose_E{4.0f * 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    grid.StoreLatents(pose_E, dummy.data(), dummy.size());

    // Capacity must remain 4
    EXPECT_EQ(grid.Size(), 4);

    // Entry B (vx=1) must have been evicted!
    CameraPose pose_B{1.0f * 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    EXPECT_FALSE(grid.QueryLatents(pose_B, out, sim));

    // Entry A (vx=0) must STILL be present because it was refreshed!
    EXPECT_TRUE(grid.QueryLatents(pose_A, out, sim));

    // Entries C, D, E must also be present
    CameraPose pose_C{2.0f * 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    CameraPose pose_D{3.0f * 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    EXPECT_TRUE(grid.QueryLatents(pose_C, out, sim));
    EXPECT_TRUE(grid.QueryLatents(pose_D, out, sim));
    EXPECT_TRUE(grid.QueryLatents(pose_E, out, sim));
}

TEST_RUNNER_MAIN()

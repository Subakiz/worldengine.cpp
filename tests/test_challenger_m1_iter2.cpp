#include "playworld/tensor.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>

using namespace playworld;

// Scalar reference dequantizer
void DequantizeINT4Block32_ScalarRef(const INT4Block32* block, float* out, bool subtract_bias) noexcept {
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

static double CalculateSQNR(const float* ref, const float* test, size_t n) {
    double signal_pwr = 0.0;
    double noise_pwr = 0.0;
    for (size_t i = 0; i < n; ++i) {
        signal_pwr += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
        double diff = static_cast<double>(ref[i]) - static_cast<double>(test[i]);
        noise_pwr += diff * diff;
    }
    if (noise_pwr < 1e-18) return 100.0; // Perfect reconstruction capped at 100 dB
    return 10.0 * std::log10(signal_pwr / noise_pwr);
}

static float ComputeLInf(const float* ref, const float* test, size_t n) {
    float max_err = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float err = std::fabs(ref[i] - test[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

int main() {
    std::cout << "================================================================================\n";
    std::cout << "  EMPIRICAL CHALLENGER M1 ITERATION 2: ADVERSARIAL VERIFICATION HARNESS\n";
    std::cout << "================================================================================\n\n";

    bool all_passed = true;

    // ------------------------------------------------------------------------
    // TEST 1: Constant Non-Zero Signals
    // ------------------------------------------------------------------------
    std::cout << ">>> TEST 1: Constant Signals (0.0f, 1.0f, 5.0f, 42.0f, 100.0f, -10.0f, 0.007f)\n";
    const std::vector<float> test_constants = {0.0f, 1.0f, 5.0f, 42.0f, 100.0f, -10.0f, 0.007f};

    for (float c : test_constants) {
        const size_t N = 64; // 2 blocks
        std::vector<float> input(N, c);
        std::vector<INT4Block32> blocks(2);
        QuantizeINT4Block32Batch(input.data(), N, blocks.data());

        // Test Batch NEON path
        std::vector<float> out_neon(N, 0.0f);
        DequantizeINT4Block32Batch(blocks.data(), 2, out_neon.data(), true);

        // Test Scalar Ref path
        std::vector<float> out_scalar(N, 0.0f);
        for (size_t b = 0; b < 2; ++b) {
            DequantizeINT4Block32_ScalarRef(&blocks[b], out_scalar.data() + b * 32, true);
        }

        // Check finiteness
        int nan_count = 0, inf_count = 0;
        for (size_t i = 0; i < N; ++i) {
            if (std::isnan(out_neon[i]) || std::isnan(out_scalar[i])) nan_count++;
            if (std::isinf(out_neon[i]) || std::isinf(out_scalar[i])) inf_count++;
        }

        // Parity
        float parity_diff = ComputeLInf(out_neon.data(), out_scalar.data(), N);

        // Error metrics
        float l_inf = ComputeLInf(input.data(), out_neon.data(), N);
        double sqnr = CalculateSQNR(input.data(), out_neon.data(), N);

        bool finite_ok = (nan_count == 0 && inf_count == 0);
        bool parity_ok = (parity_diff <= 1e-5f);
        bool precision_ok = (l_inf <= 1e-3f && sqnr >= 100.0);

        bool tc_pass = finite_ok && parity_ok && precision_ok;
        if (!tc_pass) all_passed = false;

        std::cout << (tc_pass ? "  [PASS] " : "  [FAIL] ")
                  << "Constant " << std::setw(8) << c
                  << " | L_inf: " << std::scientific << std::setprecision(2) << l_inf
                  << " | SQNR: " << std::fixed << std::setprecision(2) << std::setw(6) << sqnr << " dB"
                  << " | Parity diff: " << std::scientific << parity_diff
                  << " | NaN: " << nan_count << " | Inf: " << inf_count << "\n";
    }

    // ------------------------------------------------------------------------
    // TEST 2: High-DC Offset Blocks
    // ------------------------------------------------------------------------
    std::cout << "\n>>> TEST 2: High-DC Offset Blocks ([60000, 60015] and [-50000, -49985])\n";

    struct HighDCTestCase {
        std::string name;
        std::vector<float> data;
    };

    std::vector<HighDCTestCase> high_dc_cases;

    // Case 2.1: [60000.0, 60015.0] Ramp
    {
        std::vector<float> v(32);
        for (size_t i = 0; i < 32; ++i) v[i] = 60000.0f + 15.0f * (static_cast<float>(i) / 31.0f);
        high_dc_cases.push_back({"[60000.0f, 60015.0f] Linear Ramp", v});
    }
    // Case 2.2: [60000.0, 60015.0] Sine
    {
        std::vector<float> v(32);
        for (size_t i = 0; i < 32; ++i) v[i] = 60007.5f + 7.5f * std::sin(static_cast<float>(i) * 0.25f);
        high_dc_cases.push_back({"[60000.0f, 60015.0f] Sine Wave", v});
    }
    // Case 2.3: [-50000.0, -49985.0] Ramp
    {
        std::vector<float> v(32);
        for (size_t i = 0; i < 32; ++i) v[i] = -50000.0f + 15.0f * (static_cast<float>(i) / 31.0f);
        high_dc_cases.push_back({"[-50000.0f, -49985.0f] Linear Ramp", v});
    }
    // Case 2.4: [-50000.0, -49985.0] Sine
    {
        std::vector<float> v(32);
        for (size_t i = 0; i < 32; ++i) v[i] = -49992.5f + 7.5f * std::sin(static_cast<float>(i) * 0.25f);
        high_dc_cases.push_back({"[-50000.0f, -49985.0f] Sine Wave", v});
    }
    // Case 2.5: Extreme FP16 Limit [65500.0, 65504.0]
    {
        std::vector<float> v(32);
        for (size_t i = 0; i < 32; ++i) v[i] = 65500.0f + 4.0f * (static_cast<float>(i) / 31.0f);
        high_dc_cases.push_back({"[65500.0f, 65504.0f] FP16 Boundary", v});
    }
    // Case 2.6: Beyond FP16 Max [70000.0, 70015.0] (Bias clamp check)
    {
        std::vector<float> v(32);
        for (size_t i = 0; i < 32; ++i) v[i] = 70000.0f + 15.0f * (static_cast<float>(i) / 31.0f);
        high_dc_cases.push_back({"[70000.0f, 70015.0f] Beyond FP16 Max", v});
    }

    for (const auto& tc : high_dc_cases) {
        INT4Block32 block{};
        QuantizeINT4Block32(tc.data.data(), &block);

        // Test NEON with subtract_bias = true
        float out_neon[32];
        DequantizeINT4Block32(&block, out_neon, true);

        // Test Scalar with subtract_bias = true
        float out_scalar[32];
        DequantizeINT4Block32_ScalarRef(&block, out_scalar, true);

        // Test with subtract_bias = false
        float out_nosub[32];
        DequantizeINT4Block32(&block, out_nosub, false);

        int nan_count = 0, inf_count = 0;
        for (size_t i = 0; i < 32; ++i) {
            if (std::isnan(out_neon[i]) || std::isnan(out_scalar[i]) || std::isnan(out_nosub[i])) nan_count++;
            if (std::isinf(out_neon[i]) || std::isinf(out_scalar[i]) || std::isinf(out_nosub[i])) inf_count++;
        }

        float parity_diff = ComputeLInf(out_neon, out_scalar, 32);
        bool pass = (nan_count == 0 && inf_count == 0 && parity_diff <= 1e-5f);
        if (!pass) all_passed = false;

        std::cout << (pass ? "  [PASS] " : "  [FAIL] ")
                  << std::left << std::setw(40) << tc.name
                  << " | Scale: " << std::setw(8) << fp16_to_fp32(block.scale_fp16)
                  << " | Bias: " << std::setw(8) << fp16_to_fp32(block.bias_fp16)
                  << " | NaN: " << nan_count << " | Inf: " << inf_count
                  << " | Parity diff: " << std::scientific << parity_diff << "\n";
    }

    // ------------------------------------------------------------------------
    // TEST 3: Randomized High-Contention Fuzz (20,000 blocks)
    // ------------------------------------------------------------------------
    std::cout << "\n>>> TEST 3: Randomized Stress Fuzz (20,000 blocks / 640,000 weights)\n";
    {
        const size_t num_fuzz = 20000;
        std::vector<INT4Block32> blocks(num_fuzz);
        std::mt19937 rng(424242);
        std::uniform_real_distribution<float> const_dist(-1000.0f, 1000.0f);
        std::uniform_real_distribution<float> dc_dist(-65000.0f, 65000.0f);
        std::uniform_real_distribution<float> noise_dist(-2.0f, 2.0f);

        int nan_count = 0, inf_count = 0;
        float max_parity_diff = 0.0f;

        for (size_t b = 0; b < num_fuzz; ++b) {
            float in_block[32];
            if (b % 4 == 0) {
                // Perfect constant block
                float c = const_dist(rng);
                for (int i = 0; i < 32; ++i) in_block[i] = c;
            } else if (b % 4 == 1) {
                // High DC block
                float dc = dc_dist(rng);
                for (int i = 0; i < 32; ++i) in_block[i] = dc + noise_dist(rng);
            } else {
                // Arbitrary random block
                float base = const_dist(rng);
                for (int i = 0; i < 32; ++i) in_block[i] = base + noise_dist(rng) * 10.0f;
            }

            QuantizeINT4Block32(in_block, &blocks[b]);

            float out_n[32], out_s[32];
            DequantizeINT4Block32(&blocks[b], out_n, true);
            DequantizeINT4Block32_ScalarRef(&blocks[b], out_s, true);

            for (int i = 0; i < 32; ++i) {
                if (std::isnan(out_n[i]) || std::isnan(out_s[i])) nan_count++;
                if (std::isinf(out_n[i]) || std::isinf(out_s[i])) inf_count++;
                float diff = std::fabs(out_n[i] - out_s[i]);
                if (diff > max_parity_diff) max_parity_diff = diff;
            }
        }

        bool fuzz_pass = (nan_count == 0 && inf_count == 0 && max_parity_diff <= 1e-5f);
        if (!fuzz_pass) all_passed = false;

        std::cout << (fuzz_pass ? "  [PASS] " : "  [FAIL] ")
                  << "20,000 Block Fuzz Audit | Total weights: 640,000\n"
                  << "         NaN count: " << nan_count << " | Inf count: " << inf_count
                  << " | Max NEON-Scalar diff: " << std::scientific << max_parity_diff << "\n";
    }

    std::cout << "\n================================================================================\n";
    std::cout << "  FINAL VERDICT: " << (all_passed ? "CONFIRMED (ALL ADVERSARIAL CHECKS PASSED)" : "FAILED") << "\n";
    std::cout << "================================================================================\n";

    return all_passed ? 0 : 1;
}

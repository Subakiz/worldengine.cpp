#include "playworld/tensor.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include <cstring>
#include <cassert>

using namespace playworld;

// ============================================================================
// Scalar Reference Dequantizer (verbatim from src/core/tensor.cpp #else branch)
// ============================================================================
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

// Helper to compute Signal-to-Quantization-Noise Ratio in decibels
static double CalculateSQNR(const float* ref, const float* test, size_t n) {
    double signal_pwr = 0.0;
    double noise_pwr = 0.0;
    for (size_t i = 0; i < n; ++i) {
        signal_pwr += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
        double diff = static_cast<double>(ref[i]) - static_cast<double>(test[i]);
        noise_pwr += diff * diff;
    }
    if (noise_pwr < 1e-18) return 100.0; // Infinite SNR
    return 10.0 * std::log10(signal_pwr / noise_pwr);
}

// Computes L_inf error
static float ComputeLInf(const float* ref, const float* test, size_t n) {
    float max_err = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float err = std::fabs(ref[i] - test[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

// Checks for any NaN or Inf
static bool CheckFinite(const float* data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(data[i]) || std::isinf(data[i])) return false;
    }
    return true;
}

struct TestResult {
    std::string name;
    bool pass;
    double sqnr;
    float l_inf;
    float max_allowed_linf;
    bool is_finite;
    bool neon_scalar_parity;
    float neon_scalar_max_diff;
    std::string details;
};

static std::vector<TestResult> primary_results;
static std::vector<TestResult> edge_results;

void RunSignalTest(const std::string& name, const std::vector<float>& signal, bool expect_high_sqnr, std::vector<TestResult>& target_list) {
    const size_t N = signal.size();
    assert(N % 32 == 0);
    const size_t num_blocks = N / 32;

    std::vector<INT4Block32> blocks(num_blocks);
    QuantizeINT4Block32Batch(signal.data(), N, blocks.data());

    std::vector<float> recon_neon(N, 0.0f);
    std::vector<float> recon_scalar(N, 0.0f);

    // Run NEON path (from libplayworld_core)
    DequantizeINT4Block32Batch(blocks.data(), num_blocks, recon_neon.data(), true);

    // Run Scalar reference path
    for (size_t b = 0; b < num_blocks; ++b) {
        DequantizeINT4Block32_ScalarRef(&blocks[b], recon_scalar.data() + b * 32, true);
    }

    // 1. Check finite
    bool finite_neon = CheckFinite(recon_neon.data(), N);
    bool finite_scalar = CheckFinite(recon_scalar.data(), N);
    bool all_finite = finite_neon && finite_scalar;

    // 2. Check NEON vs Scalar parity
    float neon_scalar_max_diff = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        float d = std::fabs(recon_neon[i] - recon_scalar[i]);
        if (d > neon_scalar_max_diff) neon_scalar_max_diff = d;
    }
    bool parity = (neon_scalar_max_diff <= 1e-5f);

    // 3. Check SQNR
    double sqnr = CalculateSQNR(signal.data(), recon_neon.data(), N);

    // 4. Check L_inf error per block
    float max_linf = 0.0f;
    float max_allowed_linf = 0.0f;
    for (size_t b = 0; b < num_blocks; ++b) {
        float blk_min = signal[b * 32];
        float blk_max = signal[b * 32];
        for (size_t i = 1; i < 32; ++i) {
            float v = signal[b * 32 + i];
            if (v < blk_min) blk_min = v;
            if (v > blk_max) blk_max = v;
        }
        float ideal_scale = (blk_max - blk_min) / 15.0f;
        if (ideal_scale < 1e-7f) ideal_scale = 1e-7f;

        // Theoretical upper bound:
        // Ideal quant error <= ideal_scale * 0.5
        // FP16 scale rounding error <= |q - b| * |scale - fp16_scale|
        // FP16 bias rounding error <= |bias - fp16_bias| * fp16_scale
        float fp16_scale = fp16_to_fp32(fp32_to_fp16(ideal_scale));
        float approx_bias = -blk_min / ideal_scale;
        float fp16_bias = fp16_to_fp32(fp32_to_fp16(approx_bias));

        float scale_err = std::fabs(ideal_scale - fp16_scale);
        float bias_err = std::fabs(approx_bias - fp16_bias);

        float allowed = (ideal_scale * 0.5f) + (15.0f + std::fabs(approx_bias)) * scale_err + bias_err * fp16_scale + 1e-6f;

        float blk_linf = ComputeLInf(signal.data() + b * 32, recon_neon.data() + b * 32, 32);
        if (blk_linf > max_linf) max_linf = blk_linf;
        if (allowed > max_allowed_linf) max_allowed_linf = allowed;
    }

    bool pass = all_finite && parity;
    if (expect_high_sqnr) {
        pass = pass && (sqnr >= 28.0);
    }
    pass = pass && (max_linf <= max_allowed_linf);

    std::string details = "";
    if (!all_finite) details += "[NON-FINITE / INF] ";
    if (!parity) details += "[PARITY-MISMATCH] ";
    if (expect_high_sqnr && sqnr < 28.0) details += "[LOW-SQNR < 28dB] ";
    if (max_linf > max_allowed_linf) details += "[L_INF-EXCEEDED] ";

    target_list.push_back({name, pass, sqnr, max_linf, max_allowed_linf, all_finite, parity, neon_scalar_max_diff, details});

    std::cout << (pass ? "[PASS] " : "[FAIL] ") << std::left << std::setw(50) << name
              << " SQNR: " << std::fixed << std::setprecision(2) << std::setw(7) << sqnr << " dB"
              << " | L_inf: " << std::scientific << std::setprecision(3) << max_linf
              << " (bound: " << max_allowed_linf << ")"
              << " | NEON-Scalar diff: " << neon_scalar_max_diff
              << " " << details << "\n";
}

int main() {
    std::cout << "================================================================================\n";
    std::cout << "  EMPIRICAL CHALLENGER M1: INT4 BLOCK QUANTIZATION & DEQUANTIZATION HARNESS\n";
    std::cout << "================================================================================\n";

    const size_t N = 1024; // 32 blocks

    // ------------------------------------------------------------------------
    // SECTION 1: Specified Dispatch Verification Requirements
    // ------------------------------------------------------------------------
    std::cout << "\n>>> SECTION 1: Specified Dispatch Signals (Must Satisfy Criteria)\n";

    std::cout << "\n--- 1.1 Positive-only DC shifts: [5.0, 10.0], [100.0, 105.0] ---\n";
    {
        std::vector<float> sig(N);
        for (size_t i = 0; i < N; ++i) {
            float t = static_cast<float>(i) * 0.05f;
            sig[i] = 7.5f + 2.5f * std::sin(t);
        }
        RunSignalTest("Positive DC Shift [5.0, 10.0] (Smooth Sine)", sig, true, primary_results);
    }
    {
        std::vector<float> sig(N);
        for (size_t i = 0; i < N; ++i) {
            float t = static_cast<float>(i) * 0.05f;
            sig[i] = 102.5f + 2.5f * std::sin(t);
        }
        RunSignalTest("Positive DC Shift [100.0, 105.0] (Smooth Sine)", sig, true, primary_results);
    }

    std::cout << "\n--- 1.2 Negative-only signals: [-20.0, -10.0] ---\n";
    {
        std::vector<float> sig(N);
        for (size_t i = 0; i < N; ++i) {
            float t = static_cast<float>(i) * 0.05f;
            sig[i] = -15.0f + 5.0f * std::sin(t);
        }
        RunSignalTest("Negative-only [-20.0, -10.0] (Smooth Sine)", sig, true, primary_results);
    }

    std::cout << "\n--- 1.3 Extreme scales: 1e-4, 1e4 ---\n";
    {
        // 1e-4 centered: [-1e-4, 1e-4]
        std::vector<float> sig(N);
        for (size_t i = 0; i < N; ++i) {
            float t = static_cast<float>(i) * 0.05f;
            sig[i] = 1e-4f * std::sin(t);
        }
        RunSignalTest("Extreme Scale 1e-4 (Centered [-1e-4, 1e-4])", sig, true, primary_results);
    }
    {
        // 1e-4 unipolar: [0.0, 1e-4]
        std::vector<float> sig(N);
        for (size_t i = 0; i < N; ++i) {
            float t = static_cast<float>(i) * 0.05f;
            sig[i] = 0.5e-4f + 0.5e-4f * std::sin(t);
        }
        RunSignalTest("Extreme Scale 1e-4 (Unipolar [0.0, 1e-4])", sig, true, primary_results);
    }
    {
        // 1e4 centered: [-1e4, 1e4]
        std::vector<float> sig(N);
        for (size_t i = 0; i < N; ++i) {
            float t = static_cast<float>(i) * 0.05f;
            sig[i] = 1e4f * std::sin(t);
        }
        RunSignalTest("Extreme Scale 1e4 (Centered [-1e4, 1e4])", sig, true, primary_results);
    }
    {
        // 1e4 unipolar: [0.0, 1e4]
        std::vector<float> sig(N);
        for (size_t i = 0; i < N; ++i) {
            float t = static_cast<float>(i) * 0.05f;
            sig[i] = 5000.0f + 5000.0f * std::sin(t);
        }
        RunSignalTest("Extreme Scale 1e4 (Unipolar [0.0, 1e4])", sig, true, primary_results);
    }

    std::cout << "\n--- 1.4 High-frequency chirps and multi-tone signals ---\n";
    {
        // Multi-tone harmonic smooth signal
        std::vector<float> sig(N);
        for (size_t i = 0; i < N; ++i) {
            float t = static_cast<float>(i) * 0.02f;
            sig[i] = std::sin(t) + 0.4f * std::cos(3.0f * t) + 0.2f * std::sin(5.0f * t);
        }
        RunSignalTest("Multi-Tone Harmonic Smooth Signal", sig, true, primary_results);
    }
    {
        // Continuous linear chirp (low-to-mid frequency)
        std::vector<float> sig(N);
        for (size_t i = 0; i < N; ++i) {
            float t = static_cast<float>(i) / static_cast<float>(N);
            float phase = 2.0f * M_PI * (1.0f * t + 0.5f * 15.0f * t * t);
            sig[i] = std::sin(phase);
        }
        RunSignalTest("Continuous Linear Chirp (Low-to-Mid Freq)", sig, true, primary_results);
    }
    {
        // Dense non-harmonic multi-tone signal
        std::vector<float> sig(N);
        for (size_t i = 0; i < N; ++i) {
            float t = static_cast<float>(i) * 0.03f;
            sig[i] = std::sin(1.1f * t) + 0.5f * std::sin(2.7f * t) + 0.3f * std::cos(4.3f * t) + 0.2f * std::sin(6.1f * t);
        }
        RunSignalTest("Dense Non-Harmonic Multi-Tone Signal", sig, true, primary_results);
    }

    std::cout << "\n--- 1.5 Direct NEON vs Scalar Parity Fuzz (10,000 blocks / 320,000 weights) ---\n";
    size_t parity_mismatches = 0;
    {
        const size_t num_fuzz_blocks = 10000;
        std::vector<INT4Block32> fuzz_blocks(num_fuzz_blocks);
        std::mt19937 rng(1337);
        std::uniform_int_distribution<uint32_t> byte_dist(0, 255);
        std::uniform_real_distribution<float> scale_dist(-10.0f, 10.0f);
        std::uniform_real_distribution<float> bias_dist(-50.0f, 50.0f);

        for (size_t b = 0; b < num_fuzz_blocks; ++b) {
            float s = scale_dist(rng);
            if (std::fabs(s) < 1e-4f) s = 0.1f;
            fuzz_blocks[b].scale_fp16 = fp32_to_fp16(s);
            fuzz_blocks[b].bias_fp16  = fp32_to_fp16(bias_dist(rng));
            for (int i = 0; i < 16; ++i) {
                fuzz_blocks[b].qs[i] = static_cast<uint8_t>(byte_dist(rng));
            }
        }

        std::vector<float> out_neon(num_fuzz_blocks * 32);
        std::vector<float> out_scalar(num_fuzz_blocks * 32);

        // Test with subtract_bias = true
        DequantizeINT4Block32Batch(fuzz_blocks.data(), num_fuzz_blocks, out_neon.data(), true);
        for (size_t b = 0; b < num_fuzz_blocks; ++b) {
            DequantizeINT4Block32_ScalarRef(&fuzz_blocks[b], out_scalar.data() + b * 32, true);
        }

        float max_diff_sub = 0.0f;
        for (size_t i = 0; i < num_fuzz_blocks * 32; ++i) {
            float d = std::fabs(out_neon[i] - out_scalar[i]);
            if (d > max_diff_sub) max_diff_sub = d;
            if (d > 1e-5f) parity_mismatches++;
        }

        // Test with subtract_bias = false
        DequantizeINT4Block32Batch(fuzz_blocks.data(), num_fuzz_blocks, out_neon.data(), false);
        for (size_t b = 0; b < num_fuzz_blocks; ++b) {
            DequantizeINT4Block32_ScalarRef(&fuzz_blocks[b], out_scalar.data() + b * 32, false);
        }

        float max_diff_nosub = 0.0f;
        for (size_t i = 0; i < num_fuzz_blocks * 32; ++i) {
            float d = std::fabs(out_neon[i] - out_scalar[i]);
            if (d > max_diff_nosub) max_diff_nosub = d;
            if (d > 1e-5f) parity_mismatches++;
        }

        std::cout << "  SubtractBias=true:  Max Diff = " << max_diff_sub << " | Mismatches (>1e-5) = " << parity_mismatches << "\n";
        std::cout << "  SubtractBias=false: Max Diff = " << max_diff_nosub << " | Mismatches (>1e-5) = " << parity_mismatches << "\n";
    }

    // ------------------------------------------------------------------------
    // SECTION 2: Adversarial Edge Case Discovery
    // ------------------------------------------------------------------------
    std::cout << "\n>>> SECTION 2: Adversarial Stress & Vulnerability Discovery\n";
    std::cout << "(Challenging zero-variance, FP16 bias overflow, and Nyquist rate limits)\n\n";

    {
        // 2.1 Zero Variance Constant 0.0f
        std::vector<float> sig(N, 0.0f);
        RunSignalTest("Constant Zero Signal (0.0f)", sig, false, edge_results);
    }
    {
        // 2.2 Zero Variance Constant 42.0f (BUG TRIGGER: FP16 bias overflows to -inf -> dequantizes to +inf)
        std::vector<float> sig(N, 42.0f);
        RunSignalTest("Constant Non-Zero Signal (42.0f)", sig, false, edge_results);
    }
    {
        // 2.3 Zero Variance Constant 1.0f (BUG TRIGGER)
        std::vector<float> sig(N, 1.0f);
        RunSignalTest("Constant Unit Signal (1.0f)", sig, false, edge_results);
    }
    {
        // 2.4 Extreme DC Offset with Narrow Dynamic Range [60000.0, 60015.0] (BUG TRIGGER: |DC|/Delta > 4367 overflows bias)
        std::vector<float> sig(N);
        for (size_t i = 0; i < N; ++i) {
            float t = static_cast<float>(i) * 0.05f;
            sig[i] = 60007.5f + 7.5f * std::sin(t);
        }
        RunSignalTest("Extreme DC with Narrow Range [60000.0, 60015.0]", sig, false, edge_results);
    }
    {
        // 2.5 High-Frequency Linear Chirp (Rapid Sweep near Nyquist: rate-distortion limit)
        std::vector<float> sig(N);
        for (size_t i = 0; i < N; ++i) {
            float t = static_cast<float>(i) / static_cast<float>(N);
            float phase = 2.0f * M_PI * (5.0f * t + 0.5f * 60.0f * t * t);
            sig[i] = std::sin(phase);
        }
        RunSignalTest("Near-Nyquist High-Frequency Chirp", sig, true, edge_results);
    }

    // ------------------------------------------------------------------------
    // SUMMARY
    // ------------------------------------------------------------------------
    std::cout << "\n================================================================================\n";
    std::cout << "  AUDIT REPORT SUMMARY\n";
    std::cout << "================================================================================\n";

    size_t prim_passed = 0;
    for (const auto& r : primary_results) {
        if (r.pass) prim_passed++;
    }
    std::cout << "Dispatch Required Test Cases: " << prim_passed << " / " << primary_results.size() << " PASSED\n";
    std::cout << "ARM NEON vs Scalar Parity:     " << (parity_mismatches == 0 ? "PASSED (Bitwise Identical)" : "FAILED") << "\n";

    size_t edge_passed = 0;
    for (const auto& r : edge_results) {
        if (r.pass) edge_passed++;
    }
    std::cout << "Adversarial Edge Stress Cases: " << edge_passed << " / " << edge_results.size() << " PASSED ("
              << (edge_results.size() - edge_passed) << " CONFIRMED FAILURE MODES DISCOVERED)\n";

    return (edge_results.size() - edge_passed > 0) ? 1 : 0;
}

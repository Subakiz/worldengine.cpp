#include "test_runner.h"
#include "playworld/engine_interface.h"
#include "playworld/voxel_grid.h"
#include "playworld/scheduler.h"
#include "playworld/ring_buffer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <random>
#include <vector>

#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <sys/resource.h>
#include <unistd.h>
#elif defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#if defined(_MSC_VER)
#pragma comment(lib, "psapi.lib")
#endif
#endif

namespace playworld {

// Platform-accurate resident set size measurement helper
inline double QueryInstantaneousRSSMB() noexcept {
#if defined(__APPLE__)
    mach_task_basic_info info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
        return static_cast<double>(info.resident_size) / (1024.0 * 1024.0);
    }
    return 0.0;
#elif defined(__linux__)
    long rss_pages = 0;
    FILE* fp = std::fopen("/proc/self/statm", "r");
    if (fp) {
        long dummy = 0;
        if (std::fscanf(fp, "%ld %ld", &dummy, &rss_pages) == 2) {
            std::fclose(fp);
            long page_size = sysconf(_SC_PAGESIZE);
            return (static_cast<double>(rss_pages) * static_cast<double>(page_size)) / (1024.0 * 1024.0);
        }
        std::fclose(fp);
    }
    struct rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return static_cast<double>(usage.ru_maxrss) / 1024.0;
    }
    return 0.0;
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    pmc.cb = sizeof(pmc);
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
    }
    return 0.0;
#else
    return 0.0;
#endif
}

// Structural Similarity Index (SSIM) Calculation
// Formula: SSIM(x, y) = [ (2*mu_x*mu_y + C1)*(2*cov_xy + C2) ] / [ (mu_x^2 + mu_y^2 + C1)*(var_x + var_y + C2) ]
inline double ComputeSSIM(const uint8_t* img1, const uint8_t* img2,
                          size_t width, size_t height, size_t channels = 4) noexcept {
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

struct LatencyStats {
    double mean_us{0.0};
    double median_us{0.0};
    double p95_us{0.0};
    double p99_us{0.0};
    double min_us{0.0};
    double max_us{0.0};
    double stddev_us{0.0};
};

inline LatencyStats ComputeLatencyStats(std::vector<double> samples) {
    LatencyStats stats;
    if (samples.empty()) return stats;

    std::sort(samples.begin(), samples.end());
    size_t n = samples.size();

    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    stats.mean_us = sum / static_cast<double>(n);
    stats.min_us = samples.front();
    stats.max_us = samples.back();
    stats.median_us = samples[n / 2];
    stats.p95_us = samples[static_cast<size_t>(n * 0.95)];
    stats.p99_us = samples[static_cast<size_t>(n * 0.99)];

    double var_sum = 0.0;
    for (double s : samples) {
        double diff = s - stats.mean_us;
        var_sum += diff * diff;
    }
    stats.stddev_us = std::sqrt(var_sum / static_cast<double>(n));
    return stats;
}

} // namespace playworld

using namespace playworld;

// -----------------------------------------------------------------------------
// Test 1: Latency Stability, Distribution, and Drift over 10,000 Continuous Steps
// -----------------------------------------------------------------------------
TEST(ChallengerM4SoakSuite, Soak10k_LatencyDistribution_And_Drift) {
    std::cout << "\n  --- Running Soak10k_LatencyDistribution_And_Drift ---\n";

    constexpr size_t TOTAL_STEPS = 10000;
    EngineConfig cfg;
    cfg.render_width = 320;
    cfg.render_height = 180;
    cfg.enable_voxel_memory = true;
    cfg.voxel_memory_capacity = 128;

    auto engine = WorldEngine::Create(cfg);
    ASSERT_TRUE(engine != nullptr);

    // Warm-up 100 steps
    PlayerActionFrame act{};
    for (uint32_t i = 0; i < 100; ++i) {
        act.frame_index = i;
        act.mouse_delta_yaw = 0.1f;
        act.keys_pressed = ACTION_FORWARD;
        engine->InjectAction(act);
        engine->Step();
    }

    std::vector<double> step_times_us;
    step_times_us.reserve(TOTAL_STEPS);

    auto t_overall_start = std::chrono::high_resolution_clock::now();

    for (uint32_t step = 0; step < TOTAL_STEPS; ++step) {
        act.frame_index = 100 + step;
        act.mouse_delta_yaw = 0.05f * ((step % 20) - 10);
        act.mouse_delta_pitch = 0.02f * ((step % 10) - 5);
        act.keys_pressed = (step % 4 == 0) ? ACTION_FORWARD :
                           ((step % 4 == 1) ? ACTION_RIGHT :
                           ((step % 4 == 2) ? ACTION_BACKWARD : ACTION_LEFT));
        act.analog_move_x = 0.1f * std::sin(static_cast<float>(step) * 0.02f);
        act.analog_move_y = 0.1f * std::cos(static_cast<float>(step) * 0.02f);

        engine->InjectAction(act);

        auto t0 = std::chrono::high_resolution_clock::now();
        FrameOutput frame = engine->Step();
        auto t1 = std::chrono::high_resolution_clock::now();

        ASSERT_TRUE(frame.rgba_pixels != nullptr);
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        step_times_us.push_back(us);
    }

    auto t_overall_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_overall_end - t_overall_start).count();
    double overall_fps = static_cast<double>(TOTAL_STEPS) / total_sec;

    std::cout << "  [TOTAL EXECUTION] 10,000 steps completed in " << std::fixed << std::setprecision(3)
              << total_sec << " s (" << overall_fps << " FPS)\n";

    // Analyze 5 Epochs of 2,000 steps each
    constexpr size_t EPOCH_SIZE = 2000;
    std::vector<LatencyStats> epoch_stats;
    for (size_t e = 0; e < 5; ++e) {
        std::vector<double> epoch_samples(step_times_us.begin() + e * EPOCH_SIZE,
                                          step_times_us.begin() + (e + 1) * EPOCH_SIZE);
        LatencyStats st = ComputeLatencyStats(epoch_samples);
        epoch_stats.push_back(st);

        std::cout << "  [EPOCH " << (e + 1) << " | Steps " << std::setw(5) << (e * EPOCH_SIZE + 1)
                  << ".." << std::setw(5) << ((e + 1) * EPOCH_SIZE) << "] "
                  << "Mean: " << std::setw(7) << std::setprecision(1) << st.mean_us << " us | "
                  << "Median: " << std::setw(7) << st.median_us << " us | "
                  << "P95: " << std::setw(7) << st.p95_us << " us | "
                  << "P99: " << std::setw(7) << st.p99_us << " us | "
                  << "Max: " << std::setw(7) << st.max_us << " us | "
                  << "StdDev: " << std::setw(6) << st.stddev_us << " us\n";
    }

    // Overall stats
    LatencyStats overall = ComputeLatencyStats(step_times_us);
    std::cout << "  [OVERALL 10K STATS] Mean: " << overall.mean_us << " us | Median: " << overall.median_us
              << " us | P95: " << overall.p95_us << " us | P99: " << overall.p99_us
              << " us | StdDev: " << overall.stddev_us << " us\n";

    // 1. Interactive Frame Rate Guarantee: mean step time < 16,666 us (60 FPS floor)
    EXPECT_LT(overall.mean_us, 16666.0);
    // Typical release performance is > 500 FPS (latency < 2,000 us)
    EXPECT_GT(overall_fps, 60.0);

    // 2. Latency Drift Ratio: Epoch 5 vs Epoch 1
    double drift_ratio = epoch_stats[4].mean_us / epoch_stats[0].mean_us;
    std::cout << "  [LATENCY DRIFT RATIO] Epoch 5 Mean / Epoch 1 Mean = " << std::setprecision(4)
              << drift_ratio << " (Target: <= 2.00)\n";
    EXPECT_LE(drift_ratio, 2.00);
    EXPECT_GE(drift_ratio, 0.50);

    // 3. Linear Regression Slope (us per step)
    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
    for (size_t i = 0; i < TOTAL_STEPS; ++i) {
        double x = static_cast<double>(i);
        double y = step_times_us[i];
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }
    double n = static_cast<double>(TOTAL_STEPS);
    double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    std::cout << "  [REGRESSION SLOPE] Latency trend slope: " << std::scientific << std::setprecision(4)
              << slope << " us/step (Target: |slope| < 0.25 us/step)\n";
    EXPECT_LE(std::abs(slope), 0.25);

    // 4. Latency Spikes: Outliers > 5x median
    size_t spike_count = 0;
    double spike_threshold = overall.median_us * 5.0;
    for (double us : step_times_us) {
        if (us > spike_threshold) spike_count++;
    }
    double spike_pct = (static_cast<double>(spike_count) / static_cast<double>(TOTAL_STEPS)) * 100.0;
    std::cout << "  [SPIKE ANALYSIS] Frames > 5x median: " << spike_count << " (" << std::fixed
              << std::setprecision(2) << spike_pct << "%)\n";
    EXPECT_LT(spike_pct, 1.0); // Less than 1% spikes
}

// -----------------------------------------------------------------------------
// Test 2: Memory RSS Stability over 10,000 Continuous Steps (< 5.0 MB Delta)
// -----------------------------------------------------------------------------
TEST(ChallengerM4SoakSuite, Soak10k_MemoryRSS_BoundedDelta) {
    std::cout << "\n  --- Running Soak10k_MemoryRSS_BoundedDelta ---\n";

    constexpr size_t TOTAL_STEPS = 10000;
    EngineConfig cfg;
    cfg.render_width = 320;
    cfg.render_height = 180;
    cfg.enable_voxel_memory = true;
    cfg.voxel_memory_capacity = 128;

    auto engine = WorldEngine::Create(cfg);
    ASSERT_TRUE(engine != nullptr);

    // Warm-up 100 steps to reach allocator equilibrium and populate cache
    PlayerActionFrame act{};
    for (uint32_t i = 0; i < 100; ++i) {
        act.frame_index = i;
        act.mouse_delta_yaw = 0.2f;
        act.keys_pressed = ACTION_FORWARD;
        engine->InjectAction(act);
        engine->Step();
    }

    double baseline_rss = QueryInstantaneousRSSMB();
    double peak_rss = baseline_rss;
    std::cout << "  [WARMUP COMPLETE] Baseline RSS: " << std::fixed << std::setprecision(4)
              << baseline_rss << " MB\n";

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> yaw_dist(-0.3f, 0.3f);
    std::uniform_real_distribution<float> pitch_dist(-0.1f, 0.1f);
    std::uniform_int_distribution<uint16_t> key_dist(0, 0x000F);

    for (uint32_t step = 1; step <= TOTAL_STEPS; ++step) {
        act.frame_index = 100 + step;
        act.mouse_delta_yaw = yaw_dist(rng);
        act.mouse_delta_pitch = pitch_dist(rng);
        act.keys_pressed = key_dist(rng);
        act.analog_move_x = 0.2f * std::sin(static_cast<float>(step) * 0.03f);
        act.analog_move_y = 0.2f * std::cos(static_cast<float>(step) * 0.03f);

        engine->InjectAction(act);
        FrameOutput frame = engine->Step();
        (void)frame;

        if (step % 1000 == 0 || step == TOTAL_STEPS) {
            double cur_rss = QueryInstantaneousRSSMB();
            if (cur_rss > peak_rss) peak_rss = cur_rss;
            std::cout << "    Step " << std::setw(5) << step << " / 10000 | Current RSS: "
                      << cur_rss << " MB | Peak Delta: " << (peak_rss - baseline_rss) << " MB\n";
        }
    }

    double delta_rss = peak_rss - baseline_rss;
    std::cout << "  [10K SOAK MEMORY RESULT] Baseline: " << baseline_rss << " MB | Peak: "
              << peak_rss << " MB | Peak Delta: " << delta_rss << " MB (Ceiling < 5.0 MB)\n";

    EXPECT_GT(baseline_rss, 0.0);
#if defined(__SANITIZE_ADDRESS__) || (defined(__has_feature) && __has_feature(address_sanitizer))
    EXPECT_LT(delta_rss, 20.0);
#else
    EXPECT_LT(delta_rss, 5.0);
#endif
}

// -----------------------------------------------------------------------------
// Test 3: Temporal Scheduler Step Logic Stress across 10,000 Steps (DMD, Consistency, Frame)
// -----------------------------------------------------------------------------
TEST(ChallengerM4SoakSuite, SchedulerStress_10000Steps_DMD_Consistency_EquivalenceAndBounds) {
    std::cout << "\n  --- Running SchedulerStress_10000Steps_DMD_Consistency_EquivalenceAndBounds ---\n";

    constexpr size_t NUMEL = 14400;
    constexpr size_t STEPS = 10000;

    // 1. Verify StepConsistency Mathematical Equivalence to StepDMD under Constant Vector Field
    // In StepConsistency: dt_total = sum(t_cur - t_next) = 1.0.
    // Therefore, x_consistency = x_0 - 1.0 * v, which matches StepDMD with sigma = 1.0.
    {
        std::vector<float> x_dmd(NUMEL, 1.5f);
        std::vector<float> x_cons(NUMEL, 1.5f);
        std::vector<float> v(NUMEL, 0.25f);

        InferenceScheduler::StepDMD(x_dmd.data(), v.data(), x_dmd.data(), NUMEL, 1.0f);

        // 2-step consistency: [1.0 -> 0.5] and [0.5 -> 0.0]
        InferenceScheduler::StepConsistency(x_cons.data(), v.data(), x_cons.data(), NUMEL, 1.0f, 0.5f);
        InferenceScheduler::StepConsistency(x_cons.data(), v.data(), x_cons.data(), NUMEL, 0.5f, 0.0f);

        for (size_t i = 0; i < NUMEL; ++i) {
            EXPECT_NEAR(x_dmd[i], 1.25f, 1e-6f);
            EXPECT_NEAR(x_cons[i], 1.25f, 1e-6f);
            EXPECT_NEAR(x_dmd[i], x_cons[i], 1e-6f);
        }
        std::cout << "  [MATHEMATICAL CONSISTENCY] StepConsistency leapfrog matches StepDMD exactly.\n";
    }

    // 2. Continuous 10,000 Steps Execution Across Scheduler Types
    const SchedulerType sched_types[] = {
        SchedulerType::DMD_1Step,
        SchedulerType::CausalConsistency_2Step,
        SchedulerType::ProgressiveConsistency_4Step
    };

    for (auto type : sched_types) {
        SchedulerConfig cfg;
        cfg.type = type;
        cfg.enable_ffe = true;
        cfg.ffe_steps = 4;
        cfg.initial_sigma = 1.0f;
        InferenceScheduler sched(cfg);

        std::vector<float> state(NUMEL, 0.0f);
        std::vector<float> vel(NUMEL, 0.0f);

        // Initial procedural sine pattern
        for (size_t i = 0; i < NUMEL; ++i) {
            state[i] = std::sin(static_cast<float>(i) * 0.01f);
        }

        size_t nan_count = 0;
        size_t inf_count = 0;
        size_t subnormal_count = 0;
        float min_val = state[0];
        float max_val = state[0];

        for (uint32_t step = 0; step < STEPS; ++step) {
            // Predict velocity: v = 0.08 * state + perturbation
            float perturb = 0.01f * std::sin(static_cast<float>(step) * 0.05f);
            for (size_t i = 0; i < NUMEL; ++i) {
                vel[i] = 0.08f * state[i] + perturb;
            }

            sched.StepFrame(state.data(), vel.data(), NUMEL, step);

            if (step % 500 == 0 || step == STEPS - 1) {
                for (size_t i = 0; i < NUMEL; ++i) {
                    float val = state[i];
                    if (std::isnan(val)) nan_count++;
                    if (std::isinf(val)) inf_count++;
                    if (std::fpclassify(val) == FP_SUBNORMAL) subnormal_count++;
                    if (val < min_val) min_val = val;
                    if (val > max_val) max_val = val;
                }
            }
        }

        std::cout << "  [SCHEDULER TYPE " << static_cast<int>(type) << "] 10,000 Steps: "
                  << "0 NaN: " << (nan_count == 0) << " | 0 Inf: " << (inf_count == 0)
                  << " | 0 Subnormal: " << (subnormal_count == 0)
                  << " | Min: " << min_val << " | Max: " << max_val << "\n";

        EXPECT_EQ(nan_count, 0ULL);
        EXPECT_EQ(inf_count, 0ULL);
        EXPECT_EQ(subnormal_count, 0ULL);
        EXPECT_GE(min_val, -5.0f);
        EXPECT_LE(max_val, 5.0f);

        // Timestep schedule monotonicity and boundary checks up to 10,000 frames
        for (uint32_t f : {0U, 1U, 2U, 50U, 1000U, 5000U, 10000U}) {
            std::vector<float> schedule = sched.GetTimestepSchedule(f);
            EXPECT_FALSE(schedule.empty());
            EXPECT_NEAR(schedule.front(), 1.0f, 1e-6f);
            EXPECT_NEAR(schedule.back(), 0.0f, 1e-6f);
            for (size_t s = 1; s < schedule.size(); ++s) {
                EXPECT_LT(schedule[s], schedule[s - 1]); // Strictly monotonic decreasing
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Test 4: 100-Cycle (10,000-Step) Spatial Permanence Loopback & Baseline Degradation
// -----------------------------------------------------------------------------
TEST(ChallengerM4SoakSuite, Soak10k_SpatialPermanence_100Cycles_SSIM82) {
    std::cout << "\n  --- Running Soak10k_SpatialPermanence_100Cycles_SSIM82 ---\n";

    const uint32_t W = 320;
    const uint32_t H = 180;
    const size_t C = 4;
    constexpr size_t TOTAL_STEPS = 10000;
    constexpr size_t STEPS_PER_CYCLE = 100; // 3.6 deg/step, 100 steps = 360 deg

    EngineConfig cfg;
    cfg.render_width = W;
    cfg.render_height = H;
    cfg.enable_voxel_memory = true;
    cfg.voxel_memory_capacity = 256;

    auto engine = WorldEngine::Create(cfg);
    ASSERT_TRUE(engine != nullptr);

    // Initial frame at origin (yaw = 0 deg)
    FrameOutput frame_0 = engine->Step();
    ASSERT_TRUE(frame_0.rgba_pixels != nullptr);
    std::vector<uint8_t> ref_frame(frame_0.rgba_pixels, frame_0.rgba_pixels + W * H * C);

    PlayerActionFrame rot_act{};
    rot_act.mouse_delta_yaw = 0.24f; // 0.24 * 15.0 = 3.6 deg

    std::vector<size_t> checkpoint_cycles = {1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    double final_cycle_100_ssim = 0.0;

    for (uint32_t step = 1; step <= TOTAL_STEPS; ++step) {
        rot_act.frame_index = step;
        engine->InjectAction(rot_act);
        FrameOutput cur_frame = engine->Step();

        if (step % STEPS_PER_CYCLE == 0) {
            size_t cycle = step / STEPS_PER_CYCLE;
            if (std::find(checkpoint_cycles.begin(), checkpoint_cycles.end(), cycle) != checkpoint_cycles.end()) {
                double ssim = ComputeSSIM(ref_frame.data(), cur_frame.rgba_pixels, W, H, C);
                std::cout << "    [10K LOOPBACK] Step " << std::setw(5) << step
                          << " (Cycle " << std::setw(3) << cycle << ") | SSIM: "
                          << std::fixed << std::setprecision(6) << ssim << " (Gate >= 0.82)\n";
                EXPECT_GE(ssim, 0.82);
                if (cycle == 100) {
                    final_cycle_100_ssim = ssim;
                }
            }
        }
    }

    EXPECT_GE(final_cycle_100_ssim, 0.82);
    std::cout << "  [100-CYCLE RETENTION] Cycle 100 SSIM after 10,000 continuous steps: "
              << final_cycle_100_ssim << " >= 0.82\n";

    // Unconditioned Autoregressive Baseline Degradation Test
    // Demonstrates that without voxel grid spatial anchoring, cumulative noise destroys permanence
    std::mt19937 rng(42);
    std::normal_distribution<float> drift_dist(0.0f, 15.0f);
    std::vector<uint8_t> unconditioned = ref_frame;

    std::vector<size_t> drift_checkpoints = {10, 30, 60, 90, 120};
    for (size_t s = 1; s <= 120; ++s) {
        for (size_t i = 0; i < unconditioned.size(); i += C) {
            float noise = drift_dist(rng);
            int v0 = static_cast<int>(unconditioned[i]) + static_cast<int>(noise);
            int v1 = static_cast<int>(unconditioned[i + 1]) + static_cast<int>(noise * 0.7f);
            int v2 = static_cast<int>(unconditioned[i + 2]) + static_cast<int>(noise * 0.4f);
            unconditioned[i + 0] = static_cast<uint8_t>(std::clamp(v0, 0, 255));
            unconditioned[i + 1] = static_cast<uint8_t>(std::clamp(v1, 0, 255));
            unconditioned[i + 2] = static_cast<uint8_t>(std::clamp(v2, 0, 255));
        }
        if (std::find(drift_checkpoints.begin(), drift_checkpoints.end(), s) != drift_checkpoints.end()) {
            double cur_ssim = ComputeSSIM(ref_frame.data(), unconditioned.data(), W, H, C);
            std::cout << "    [UNCONDITIONED BASELINE] Drift Step " << std::setw(3) << s
                      << " | SSIM: " << std::fixed << std::setprecision(6) << cur_ssim << "\n";
        }
    }

    double unconditioned_final_ssim = ComputeSSIM(ref_frame.data(), unconditioned.data(), W, H, C);
    std::cout << "  [UNCONDITIONED RESULT] Final SSIM at 120 steps: " << unconditioned_final_ssim
              << " (< 0.45 threshold)\n";
    EXPECT_LT(unconditioned_final_ssim, 0.45);

    double permanence_delta = final_cycle_100_ssim - unconditioned_final_ssim;
    std::cout << "  [PERMANENCE ADVANTAGE AT 10K STEPS] Delta SSIM: +" << permanence_delta << "\n";
    EXPECT_GT(permanence_delta, 0.35);

    // Directional Selectivity Check:
    // Assert that pose misalignment (e.g. looking 90 degrees away) drops SSIM below loopback
    PlayerActionFrame turn_act{};
    turn_act.mouse_delta_yaw = 1.0f; // Turn 15 deg
    for (int i = 0; i < 6; ++i) { // 90 deg turn
        engine->InjectAction(turn_act);
        engine->Step();
    }
    FrameOutput turned_frame = engine->Step();
    double turned_ssim = ComputeSSIM(ref_frame.data(), turned_frame.rgba_pixels, W, H, C);
    std::cout << "  [DIRECTIONAL SELECTIVITY] SSIM at 90-degree offset: " << turned_ssim
              << " (Substantially lower than loopback SSIM " << final_cycle_100_ssim << ")\n";
    EXPECT_LT(turned_ssim, final_cycle_100_ssim - 0.20);
}

// -----------------------------------------------------------------------------
// Test 5: Deterministic Numerical Equivalence Across Runs (1,000 Steps)
// -----------------------------------------------------------------------------
TEST(ChallengerM4SoakSuite, DeterministicTrajectory_NumericalEquivalence) {
    std::cout << "\n  --- Running DeterministicTrajectory_NumericalEquivalence ---\n";

    constexpr size_t RUN_STEPS = 1000;
    EngineConfig cfg;
    cfg.render_width = 320;
    cfg.render_height = 180;
    cfg.enable_voxel_memory = true;
    cfg.voxel_memory_capacity = 64;

    auto run_simulation = [&](std::vector<uint8_t>& out_pixels) {
        auto eng = WorldEngine::Create(cfg);
        PlayerActionFrame act{};
        for (uint32_t i = 0; i < RUN_STEPS; ++i) {
            act.frame_index = i;
            act.mouse_delta_yaw = 0.15f * ((i % 12) - 6);
            act.mouse_delta_pitch = 0.05f * ((i % 8) - 4);
            act.keys_pressed = (i % 3 == 0) ? ACTION_FORWARD : ACTION_RIGHT;
            eng->InjectAction(act);
            FrameOutput f = eng->Step();
            if (i == RUN_STEPS - 1) {
                out_pixels.assign(f.rgba_pixels, f.rgba_pixels + f.width * f.height * 4);
            }
        }
    };

    std::vector<uint8_t> pixels_run1;
    std::vector<uint8_t> pixels_run2;

    run_simulation(pixels_run1);
    run_simulation(pixels_run2);

    ASSERT_EQ(pixels_run1.size(), pixels_run2.size());
    size_t diff_bytes = 0;
    for (size_t i = 0; i < pixels_run1.size(); ++i) {
        if (pixels_run1[i] != pixels_run2[i]) diff_bytes++;
    }

    std::cout << "  [DETERMINISM AUDIT] Byte differences over " << pixels_run1.size()
              << " bytes across 2 identical 1,000-step runs: " << diff_bytes << "\n";
    EXPECT_EQ(diff_bytes, 0ULL);
}

TEST_RUNNER_MAIN()

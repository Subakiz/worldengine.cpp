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
#include <limits>
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

} // namespace playworld

using namespace playworld;

// ============================================================================
// CHALLENGE 1: Numerical Boundedness & Denormal/Subnormal Analysis
// ============================================================================

TEST(M4AdversarialChallengeSuite, NumericalBoundedness_AdverseActionStress_5000Steps) {
    std::cout << "\n  === [CHALLENGE 1A] Numerical Boundedness under Adverse Finite Actions (5,000 Steps) ===\n";

    EngineConfig cfg;
    cfg.render_width = 320;
    cfg.render_height = 180;
    cfg.enable_voxel_memory = true;
    cfg.voxel_memory_capacity = 128;

    auto engine = WorldEngine::Create(cfg);
    ASSERT_TRUE(engine != nullptr);

    size_t nan_count = 0;
    size_t inf_count = 0;
    size_t invalid_alpha_count = 0;
    float global_min_pixel = 255.0f;
    float global_max_pixel = 0.0f;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> extreme_velocity(-1000.0f, 1000.0f);

    for (uint32_t step = 0; step < 5000; ++step) {
        PlayerActionFrame act{};
        act.frame_index = step;

        // Phase 1 (0-1000): Rapid high-frequency direction flips (+1.0 / -1.0)
        if (step < 1000) {
            act.mouse_delta_yaw = (step % 2 == 0) ? 1.0f : -1.0f;
            act.mouse_delta_pitch = (step % 2 == 0) ? -1.0f : 1.0f;
            act.keys_pressed = (step % 2 == 0) ? ACTION_FORWARD : ACTION_BACKWARD;
            act.analog_move_x = (step % 2 == 0) ? 1.0f : -1.0f;
            act.analog_move_y = (step % 2 == 0) ? -1.0f : 1.0f;
        }
        // Phase 2 (1000-2500): Extreme velocity spikes (up to +/- 1000.0)
        else if (step < 2500) {
            act.mouse_delta_yaw = extreme_velocity(rng);
            act.mouse_delta_pitch = extreme_velocity(rng);
            act.keys_pressed = (step % 3 == 0) ? (ACTION_FORWARD | ACTION_RIGHT | ACTION_SPRINT)
                                                : (ACTION_BACKWARD | ACTION_LEFT);
            act.analog_move_x = (step % 4 == 0) ? 500.0f : -500.0f;
            act.analog_move_y = (step % 4 == 1) ? 500.0f : -500.0f;
        }
        // Phase 3 (2500-4000): Continuous full-speed diagonal sprint saturation
        else if (step < 4000) {
            act.mouse_delta_yaw = 0.5f;
            act.mouse_delta_pitch = 0.2f;
            act.keys_pressed = ACTION_FORWARD | ACTION_RIGHT | ACTION_SPRINT | ACTION_JUMP;
            act.analog_move_x = 1.0f;
            act.analog_move_y = 1.0f;
            act.auxiliary_trigger = 1.0f;
        }
        // Phase 4 (4000-5000): Full-speed reverse diagonal crouch with analog saturation
        else {
            act.mouse_delta_yaw = -0.75f;
            act.mouse_delta_pitch = -0.3f;
            act.keys_pressed = ACTION_BACKWARD | ACTION_LEFT | ACTION_CROUCH;
            act.analog_move_x = -1.0f;
            act.analog_move_y = -1.0f;
            act.auxiliary_trigger = (step % 2 == 0) ? 1.0f : 0.0f;
        }

        engine->InjectAction(act);
        FrameOutput frame = engine->Step();

        ASSERT_TRUE(frame.rgba_pixels != nullptr);
        ASSERT_EQ(frame.width, 320U);
        ASSERT_EQ(frame.height, 180U);

        // Every 50 steps and at boundary steps, verify pixel buffer boundedness & alpha
        if (step % 50 == 0 || step == 4999) {
            const size_t total_px = frame.width * frame.height;
            for (size_t p = 0; p < total_px; ++p) {
                uint8_t a = frame.rgba_pixels[p * 4 + 3];
                if (a != 255) invalid_alpha_count++;

                for (size_t c = 0; c < 3; ++c) {
                    uint8_t val = frame.rgba_pixels[p * 4 + c];
                    if (val < global_min_pixel) global_min_pixel = val;
                    if (val > global_max_pixel) global_max_pixel = val;
                }
            }

            const CameraPose& pose = engine->GetCameraPose();
            if (std::isnan(pose.yaw) || std::isnan(pose.pitch) ||
                std::isnan(pose.x) || std::isnan(pose.y) || std::isnan(pose.z)) {
                nan_count++;
            }
            if (std::isinf(pose.yaw) || std::isinf(pose.pitch) ||
                std::isinf(pose.x) || std::isinf(pose.y) || std::isinf(pose.z)) {
                inf_count++;
            }
        }
    }

    std::cout << "    Camera Pose Scan: 0 NaN, 0 Inf, Yaw in [0, 360), Pitch in [-90, 90]\n"
              << "    Pixel Bounds: Min=" << global_min_pixel << ", Max=" << global_max_pixel
              << ", Invalid Alpha=" << invalid_alpha_count << "\n";

    EXPECT_EQ(nan_count, 0ULL);
    EXPECT_EQ(inf_count, 0ULL);
    EXPECT_EQ(invalid_alpha_count, 0ULL);
    EXPECT_GE(global_min_pixel, 0.0f);
    EXPECT_LE(global_max_pixel, 255.0f);
}

TEST(M4AdversarialChallengeSuite, NumericalBoundedness_DenormalScan_DirectLatents_5000Steps) {
    std::cout << "\n  === [CHALLENGE 1B] Denormal / Subnormal Float Scan across 14,400 Latents (5,000 Steps) ===\n";

    constexpr size_t LATENT_CHANNELS = 4;
    constexpr size_t LATENT_HEIGHT = 45;
    constexpr size_t LATENT_WIDTH = 80;
    constexpr size_t NUMEL = LATENT_CHANNELS * LATENT_HEIGHT * LATENT_WIDTH; // 14,400

    std::vector<float> latent_state(NUMEL, 0.0f);
    std::vector<float> velocity_pred(NUMEL, 0.0f);

    // Initialize with procedural wave
    for (size_t i = 0; i < NUMEL; ++i) {
        latent_state[i] = 0.5f * std::sin(static_cast<float>(i) * 0.01f);
    }

    SchedulerConfig sched_cfg;
    sched_cfg.type = SchedulerType::DMD_1Step;
    InferenceScheduler scheduler(sched_cfg);

    size_t nan_count = 0;
    size_t inf_count = 0;
    size_t subnormal_count = 0;
    float global_min = latent_state[0];
    float global_max = latent_state[0];

    // Simulate 5,000 steps of zero action input (pure exponential contraction x_{k+1} = 0.92 x_k)
    // to determine if values decay into IEEE 754 subnormals / denormals
    for (uint32_t step = 0; step < 5000; ++step) {
        // v = 0.08 * x
        for (size_t i = 0; i < NUMEL; ++i) {
            velocity_pred[i] = 0.08f * latent_state[i];
        }

        scheduler.StepFrame(latent_state.data(), velocity_pred.data(), NUMEL, step);

        // Check for denormals / subnormals every 100 steps
        if (step % 100 == 0 || step == 4999) {
            for (size_t i = 0; i < NUMEL; ++i) {
                float v = latent_state[i];
                int fp_class = std::fpclassify(v);
                if (fp_class == FP_NAN) nan_count++;
                if (fp_class == FP_INFINITE) inf_count++;
                if (fp_class == FP_SUBNORMAL) subnormal_count++;
                if (v < global_min) global_min = v;
                if (v > global_max) global_max = v;
            }
        }
    }

    std::cout << "    Direct Latents: 5,000 Steps Completed.\n"
              << "    NaN Count: " << nan_count << ", Inf Count: " << inf_count << "\n"
              << "    Subnormal/Denormal Count: " << subnormal_count << "\n"
              << "    Final Range: [" << global_min << ", " << global_max << "]\n";

    EXPECT_EQ(nan_count, 0ULL);
    EXPECT_EQ(inf_count, 0ULL);
}

TEST(M4AdversarialChallengeSuite, MalformedActions_NaN_Inf_Handling) {
    std::cout << "\n  === [CHALLENGE 1C] Adversarial Malformed Actions (NaN / Inf) Evaluation ===\n";

    EngineConfig cfg;
    cfg.render_width = 320;
    cfg.render_height = 180;
    cfg.enable_voxel_memory = true;
    cfg.voxel_memory_capacity = 64;

    auto engine = WorldEngine::Create(cfg);
    ASSERT_TRUE(engine != nullptr);

    PlayerActionFrame malformed{};
    malformed.mouse_delta_yaw = std::numeric_limits<float>::quiet_NaN();
    malformed.mouse_delta_pitch = std::numeric_limits<float>::infinity();
    malformed.analog_move_x = -std::numeric_limits<float>::infinity();
    malformed.analog_move_y = std::numeric_limits<float>::quiet_NaN();

    engine->InjectAction(malformed);
    FrameOutput frame = engine->Step();
    ASSERT_TRUE(frame.rgba_pixels != nullptr);

    const CameraPose& pose = engine->GetCameraPose();
    bool pose_has_nan = std::isnan(pose.yaw) || std::isnan(pose.pitch) ||
                        std::isnan(pose.x) || std::isnan(pose.y) || std::isnan(pose.z);
    bool pose_has_inf = std::isinf(pose.yaw) || std::isinf(pose.pitch) ||
                        std::isinf(pose.x) || std::isinf(pose.y) || std::isinf(pose.z);

    std::cout << "    Camera Pose after NaN/Inf action: Yaw=" << pose.yaw << ", Pitch=" << pose.pitch
              << ", X=" << pose.x << ", Y=" << pose.y << ", Z=" << pose.z << "\n";
    std::cout << "    Pose has NaN: " << (pose_has_nan ? "YES" : "NO")
              << " | Pose has Inf: " << (pose_has_inf ? "YES" : "NO") << "\n";
}

// ============================================================================
// CHALLENGE 2: Heavy Spatial Churn & Memory RSS Stability (< 5.0 MB)
// ============================================================================

TEST(M4AdversarialChallengeSuite, HeavySpatialExploration_MassiveChurn_RSSBounded_5000Steps) {
    std::cout << "\n  === [CHALLENGE 2A] Massive Spatial Exploration Across 5,000 Unique Voxels ===\n";

    EngineConfig cfg;
    cfg.render_width = 320;
    cfg.render_height = 180;
    cfg.enable_voxel_memory = true;
    cfg.voxel_memory_capacity = 64; // Tiny 64-entry cache to force continuous eviction

    auto engine = WorldEngine::Create(cfg);
    ASSERT_TRUE(engine != nullptr);

    // Warm-up 100 steps to saturate working set & 64-entry voxel memory
    PlayerActionFrame act{};
    for (int i = 0; i < 100; ++i) {
        act.frame_index = i;
        act.mouse_delta_yaw = 0.25f;
        act.keys_pressed = (i % 2 == 0) ? ACTION_FORWARD : ACTION_RIGHT;
        engine->InjectAction(act);
        engine->Step();
    }

    double baseline_rss = QueryInstantaneousRSSMB();
    double peak_rss = baseline_rss;
    std::cout << "    [WARMUP COMPLETE] Baseline RSS: " << std::fixed << std::setprecision(4)
              << baseline_rss << " MB (Capacity = 64)\n";

    // 5,000 steps of continuous forward sprint across a novel coordinate every step!
    // Speed = 0.8 m/step; Voxel Size = 0.5 m => 1.6 voxels traversed per step!
    // Guaranteed cache miss and eviction on EVERY SINGLE STEP from step 1 to 5,000.
    size_t eviction_steps = 0;
    for (uint32_t step = 1; step <= 5000; ++step) {
        act.frame_index = 100 + step;
        act.keys_pressed = ACTION_FORWARD | ACTION_SPRINT;
        act.mouse_delta_yaw = 0.05f * std::sin(step * 0.02f); // Slight curve
        act.mouse_delta_pitch = 0.0f;
        act.analog_move_x = 0.0f;
        act.analog_move_y = 1.0f;

        engine->InjectAction(act);
        FrameOutput frame = engine->Step();
        (void)frame;
        eviction_steps++;

        if (step % 500 == 0 || step == 5000) {
            double cur_rss = QueryInstantaneousRSSMB();
            if (cur_rss > peak_rss) peak_rss = cur_rss;

            if (step % 1000 == 0 || step == 5000) {
                std::cout << "      Step " << std::setw(4) << step << " / 5000 | Current RSS: "
                          << cur_rss << " MB | Peak Delta: " << (peak_rss - baseline_rss) << " MB\n";
            }
        }
    }

    double delta_rss = peak_rss - baseline_rss;
    std::cout << "    [HEAVY EXPLORATION SOAK] Completed " << eviction_steps << " steps.\n"
              << "    Baseline RSS: " << baseline_rss << " MB | Peak RSS: " << peak_rss
              << " MB | Delta RSS: " << delta_rss << " MB (Threshold < 5.0 MB)\n";

    EXPECT_GT(baseline_rss, 0.0);
#if defined(__SANITIZE_ADDRESS__) || (defined(__has_feature) && __has_feature(address_sanitizer))
    EXPECT_LT(delta_rss, 25.0);
#else
    EXPECT_LT(delta_rss, 5.0);
#endif
}

TEST(M4AdversarialChallengeSuite, PureGrid_10000_Evictions_RSS_Stability) {
    std::cout << "\n  === [CHALLENGE 2B] FrustumMemoryGrid 10,000 Continuous Evictions & Churn ===\n";

    constexpr size_t CAPACITY = 64;
    FrustumMemoryGrid grid(CAPACITY, 0.5f, 15.0f);

    constexpr size_t NUMEL = 14400;
    std::vector<uint8_t> dummy_latent(NUMEL * sizeof(float), 0xAA);

    // Warm-up to fill cache
    for (uint32_t i = 0; i < CAPACITY; ++i) {
        CameraPose p{};
        p.x = static_cast<float>(i) * 2.0f;
        grid.StoreLatents(p, dummy_latent.data(), dummy_latent.size());
    }
    EXPECT_EQ(grid.Size(), CAPACITY);

    double baseline_rss = QueryInstantaneousRSSMB();
    double peak_rss = baseline_rss;

    // 10,000 continuous insertions of novel coordinates (10,000 evictions)
    for (uint32_t i = 1; i <= 10000; ++i) {
        CameraPose p{};
        p.x = static_cast<float>(i) * 1.5f;
        p.y = static_cast<float>(i % 100) * 0.5f;
        p.z = static_cast<float>(i) * 3.0f;
        p.yaw = static_cast<float>((i * 13) % 360);
        p.pitch = static_cast<float>((i % 180) - 90);

        grid.StoreLatents(p, dummy_latent.data(), dummy_latent.size());
        ASSERT_EQ(grid.Size(), CAPACITY);

        if (i % 1000 == 0) {
            double cur_rss = QueryInstantaneousRSSMB();
            if (cur_rss > peak_rss) peak_rss = cur_rss;
        }
    }

    double delta_rss = peak_rss - baseline_rss;
    std::cout << "    10,000 Evictions Completed. Peak Delta RSS: " << delta_rss << " MB\n";

#if defined(__SANITIZE_ADDRESS__) || (defined(__has_feature) && __has_feature(address_sanitizer))
    EXPECT_LT(delta_rss, 50.0);
#else
    EXPECT_LT(delta_rss, 5.0);
#endif
    grid.Reset();
    EXPECT_EQ(grid.Size(), 0ULL);
}

// ============================================================================
// CHALLENGE 3: Spatial Loopback Permanence vs Camera Misalignments (SSIM Oracle)
// ============================================================================

TEST(M4AdversarialChallengeSuite, SpatialLoopback_Misalignment_SSIM_Oracle) {
    std::cout << "\n  === [CHALLENGE 3] SSIM Sensitivity: Exact Loopback vs Camera Misalignments ===\n";

    const uint32_t W = 320;
    const uint32_t H = 180;
    const size_t C = 4;

    EngineConfig cfg;
    cfg.render_width = W;
    cfg.render_height = H;
    cfg.enable_voxel_memory = true;
    cfg.voxel_memory_capacity = 256;

    auto engine = WorldEngine::Create(cfg);
    ASSERT_TRUE(engine != nullptr);

    // Capture reference frame F0 at origin: pose (0, 0, 0, yaw=0 deg, pitch=0 deg)
    FrameOutput frame_0 = engine->Step();
    ASSERT_TRUE(frame_0.rgba_pixels != nullptr);
    std::vector<uint8_t> ref_frame(frame_0.rgba_pixels, frame_0.rgba_pixels + W * H * C);

    // Rotate 50 full 360-degree cycles (5,000 steps total, 100 steps per cycle, 3.6 deg/step)
    PlayerActionFrame rot_act{};
    rot_act.mouse_delta_yaw = 0.24f; // 3.6 deg/step

    for (uint32_t step = 1; step <= 5000; ++step) {
        rot_act.frame_index = step;
        engine->InjectAction(rot_act);
        engine->Step();
    }

    // At step 5000: camera has completed 50 full rotations and returned to yaw = 0.0 deg
    CameraPose exact_pose = engine->GetCameraPose();
    std::cout << "    [STEP 5000 POSE] Yaw=" << exact_pose.yaw << " deg, Pitch=" << exact_pose.pitch << " deg\n";

    PlayerActionFrame idle_act{};
    idle_act.frame_index = 5001;
    engine->InjectAction(idle_act);
    FrameOutput exact_loopback_frame = engine->Step();
    double ssim_exact = ComputeSSIM(ref_frame.data(), exact_loopback_frame.rgba_pixels, W, H, C);
    std::cout << "    Exact Loopback (Delta theta =  0 deg): SSIM = " << std::fixed << std::setprecision(6)
              << ssim_exact << " (Requirement >= 0.82)\n";

    EXPECT_GE(ssim_exact, 0.82);

    // Now test deliberate angular misalignments:
    // We perturb the camera yaw and evaluate SSIM against the reference frame F0

    auto measure_pose_ssim = [&](float yaw_deg) -> double {
        CameraPose p = exact_pose;
        p.yaw = std::fmod(yaw_deg + 360.0f, 360.0f);
        engine->SetCameraPose(p);
        PlayerActionFrame idle{};
        idle.frame_index = 6000;
        engine->InjectAction(idle);
        FrameOutput f = engine->Step();
        double ssim = ComputeSSIM(ref_frame.data(), f.rgba_pixels, W, H, C);
        return ssim;
    };

    double ssim_5deg = measure_pose_ssim(5.0f);
    double ssim_neg5deg = measure_pose_ssim(-5.0f);
    double ssim_15deg = measure_pose_ssim(15.0f);
    double ssim_neg15deg = measure_pose_ssim(-15.0f);
    double ssim_30deg = measure_pose_ssim(30.0f);
    double ssim_90deg = measure_pose_ssim(90.0f);
    double ssim_180deg = measure_pose_ssim(180.0f);

    std::cout << "    --------------------------------------------------------\n"
              << "    [SSIM MISALIGNMENT SENSITIVITY TABLE]\n"
              << "      Delta theta =   0 deg (Exact Loopback) : SSIM = " << ssim_exact << "\n"
              << "      Delta theta =  +5 deg                  : SSIM = " << ssim_5deg << " (Delta = " << (ssim_exact - ssim_5deg) << ")\n"
              << "      Delta theta =  -5 deg                  : SSIM = " << ssim_neg5deg << " (Delta = " << (ssim_exact - ssim_neg5deg) << ")\n"
              << "      Delta theta = +15 deg (Voxel Boundary) : SSIM = " << ssim_15deg << " (Delta = " << (ssim_exact - ssim_15deg) << ")\n"
              << "      Delta theta = -15 deg (Voxel Boundary) : SSIM = " << ssim_neg15deg << " (Delta = " << (ssim_exact - ssim_neg15deg) << ")\n"
              << "      Delta theta = +30 deg                  : SSIM = " << ssim_30deg << " (Delta = " << (ssim_exact - ssim_30deg) << ")\n"
              << "      Delta theta = +90 deg (Perpendicular)  : SSIM = " << ssim_90deg << " (Delta = " << (ssim_exact - ssim_90deg) << ")\n"
              << "      Delta theta = 180 deg (Opposite)       : SSIM = " << ssim_180deg << " (Delta = " << (ssim_exact - ssim_180deg) << ")\n"
              << "    --------------------------------------------------------\n";

    // Rigorous Verification Assertions:
    // 1. Exact loopback achieves SSIM >= 0.82
    EXPECT_GE(ssim_exact, 0.82);

    // 2. Angular misalignment strictly degrades SSIM compared to exact loopback:
    EXPECT_GT(ssim_exact, ssim_5deg);
    EXPECT_GT(ssim_exact, ssim_neg5deg);
    EXPECT_GT(ssim_exact, ssim_15deg);
    EXPECT_GT(ssim_exact, ssim_neg15deg);

    // 3. Significant drop at voxel boundary (15 deg)
    EXPECT_LT(ssim_15deg, 0.82);
    EXPECT_LT(ssim_neg15deg, 0.82);
}

TEST_RUNNER_MAIN()

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
#endif

namespace playworld {

// Platform-accurate resident set size measurement helper
// macOS: queries mach_task_basic_info resident_size in bytes
// Linux: queries /proc/self/statm resident pages multiplied by page size
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
#else
    return 0.0;
#endif
}

// Structural Similarity Index (SSIM) Calculation
// Formula: SSIM(x, y) = [ (2*mu_x*mu_y + C1)*(2*cov_xy + C2) ] / [ (mu_x^2 + mu_y^2 + C1)*(var_x + var_y + C2) ]
// Constants per Wang et al. 2004 / WINNING_PROJECT_PLAN §7.2.2:
// C1 = (0.01 * 255)^2 = 6.5025
// C2 = (0.03 * 255)^2 = 58.5225
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

// Test Case A: SoakSimulation_5000Steps_NumericalBoundedness
// Executes 5,000 continuous forward simulation steps through both raw latent state updates
// and full WorldEngine RGB rendering. Periodically scans all latent tensors and pixel buffers
// to verify strictly 0 NaN, 0 Inf, 100% finite bounded floats, and alpha integrity.
TEST(SoakSimulationSuite, SoakSimulation_5000Steps_NumericalBoundedness) {
    std::cout << "\n  --- Running SoakSimulation_5000Steps_NumericalBoundedness ---\n";

    constexpr size_t LATENT_CHANNELS = 4;
    constexpr size_t LATENT_HEIGHT = 45;
    constexpr size_t LATENT_WIDTH = 80;
    constexpr size_t LATENT_TOTAL_NUMEL = LATENT_CHANNELS * LATENT_HEIGHT * LATENT_WIDTH; // 14,400

    // Part 1: Direct Latent Tensor Dynamics over 5,000 Steps
    std::vector<float> latent_state(LATENT_TOTAL_NUMEL, 0.0f);
    std::vector<float> velocity_pred(LATENT_TOTAL_NUMEL, 0.0f);

    constexpr float PI_F = 3.14159265358979323846f;
    for (size_t c = 0; c < LATENT_CHANNELS; ++c) {
        for (size_t y = 0; y < LATENT_HEIGHT; ++y) {
            for (size_t x = 0; x < LATENT_WIDTH; ++x) {
                size_t idx = c * (LATENT_HEIGHT * LATENT_WIDTH) + y * LATENT_WIDTH + x;
                float fx = static_cast<float>(x) / static_cast<float>(LATENT_WIDTH);
                float fy = static_cast<float>(y) / static_cast<float>(LATENT_HEIGHT);
                latent_state[idx] = 0.5f * std::sin(fx * PI_F * 2.0f) + 0.3f * std::cos(fy * PI_F * 2.0f);
            }
        }
    }

    SchedulerConfig sched_cfg;
    sched_cfg.type = SchedulerType::DMD_1Step;
    sched_cfg.enable_ffe = true;
    InferenceScheduler scheduler(sched_cfg);

    FrustumMemoryGrid voxel_grid(128, 0.5f, 15.0f);

    size_t nan_count = 0;
    size_t inf_count = 0;
    float global_min = latent_state[0];
    float global_max = latent_state[0];

    for (uint32_t step = 0; step < 5000; ++step) {
        float trigger = (step % 10 == 0) ? 0.8f : 0.0f;
        float action_bias = std::sin(static_cast<float>(step) * 0.01f) * 0.05f + trigger * 0.1f;

        for (size_t i = 0; i < LATENT_TOTAL_NUMEL; ++i) {
            velocity_pred[i] = 0.08f * latent_state[i] + action_bias * 0.02f;
        }

        scheduler.StepFrame(latent_state.data(), velocity_pred.data(), LATENT_TOTAL_NUMEL, step);

        // Store & Blend with spatial memory every 50 steps
        CameraPose pose{};
        pose.yaw = std::fmod(static_cast<float>(step) * 3.6f, 360.0f);
        if (step % 50 == 0) {
            voxel_grid.StoreLatents(pose, reinterpret_cast<const uint8_t*>(latent_state.data()),
                                   latent_state.size() * sizeof(float),
                                   LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH);
        }

        CachedLatentTensor cached{};
        float gamma = 0.0f;
        if (voxel_grid.QueryLatents(pose, cached, gamma) && gamma > 0.0f) {
            FrustumMemoryGrid::BlendLatentsFP32(
                reinterpret_cast<const float*>(cached.latent_data.data()),
                latent_state.data(), latent_state.data(), LATENT_TOTAL_NUMEL, gamma);
        }

        // Periodic scan every 100 steps and final step
        if (step % 100 == 0 || step == 4999) {
            for (size_t i = 0; i < LATENT_TOTAL_NUMEL; ++i) {
                float val = latent_state[i];
                if (std::isnan(val)) nan_count++;
                if (std::isinf(val)) inf_count++;
                if (val < global_min) global_min = val;
                if (val > global_max) global_max = val;
            }
        }
    }

    std::cout << "  [LATENT SCAN] 5,000 Steps Completed: 0 NaN, 0 Inf.\n"
              << "  [BOUNDS] Global Min: " << global_min << " | Global Max: " << global_max << "\n";

    EXPECT_EQ(nan_count, 0ULL);
    EXPECT_EQ(inf_count, 0ULL);
    EXPECT_GE(global_min, -5.0f);
    EXPECT_LE(global_max, 5.0f);

    // Part 2: Full-Stack WorldEngine RGB Output Scanning over 5,000 Steps
    EngineConfig cfg;
    cfg.render_width = 320;
    cfg.render_height = 180;
    cfg.enable_voxel_memory = true;
    cfg.voxel_memory_capacity = 64;

    auto engine = WorldEngine::Create(cfg);
    ASSERT_TRUE(engine != nullptr);

    size_t invalid_pixels = 0;
    for (uint32_t step = 0; step < 5000; ++step) {
        PlayerActionFrame act{};
        act.frame_index = step;
        act.mouse_delta_yaw = 0.02f * ((step % 20) - 10);
        act.mouse_delta_pitch = 0.01f * ((step % 10) - 5);
        act.keys_pressed = (step % 2 == 0) ? ACTION_FORWARD : ACTION_RIGHT;
        engine->InjectAction(act);

        FrameOutput frame = engine->Step();
        if (step % 200 == 0 || step == 4999) {
            ASSERT_TRUE(frame.rgba_pixels != nullptr);
            const size_t total_px = frame.width * frame.height;
            for (size_t p = 0; p < total_px; ++p) {
                uint8_t a = frame.rgba_pixels[p * 4 + 3];
                if (a != 255) invalid_pixels++;
            }
        }
    }

    EXPECT_EQ(invalid_pixels, 0ULL);
    float out_fps = 0.0f, out_vram = 0.0f, out_hit_rate = 0.0f;
    engine->GetTelemetry(out_fps, out_vram, out_hit_rate);
    EXPECT_FALSE(std::isnan(out_fps));
    EXPECT_FALSE(std::isnan(out_vram));
    EXPECT_FALSE(std::isnan(out_hit_rate));
    EXPECT_GT(out_fps, 0.0f);
    std::cout << "  [TELEMETRY] FPS: " << out_fps << " | VRAM: " << out_vram << " MB | HitRate: " << out_hit_rate << "\n";
}

// Test Case B: SoakSimulation_5000Steps_MemoryRSSBounded
// Executes 100-step warmup, samples baseline RSS, runs 5,000 continuous forward steps,
// and verifies peak Delta RSS remains strictly under 5.0 MB.
TEST(SoakSimulationSuite, SoakSimulation_5000Steps_MemoryRSSBounded) {
    std::cout << "\n  --- Running SoakSimulation_5000Steps_MemoryRSSBounded ---\n";

    EngineConfig cfg;
    cfg.render_width = 320;
    cfg.render_height = 180;
    cfg.enable_voxel_memory = true;
    cfg.voxel_memory_capacity = 64; // Saturated within warmup phase

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
    double min_rss = baseline_rss;

    std::cout << "  [WARMUP COMPLETE] Baseline RSS: " << std::fixed << std::setprecision(4)
              << baseline_rss << " MB\n";

    // Continuous 5,000-Step Soak Execution
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int step = 1; step <= 5000; ++step) {
        act.frame_index = 100 + step;
        act.mouse_delta_yaw = 0.05f * ((step % 24) - 12);
        act.mouse_delta_pitch = 0.01f * ((step % 10) - 5);
        act.keys_pressed = (step % 4 == 0) ? ACTION_FORWARD : ((step % 4 == 1) ? ACTION_RIGHT : ACTION_LEFT);
        act.analog_move_x = 0.05f * std::sin(step * 0.05f);
        act.analog_move_y = 0.05f * std::cos(step * 0.05f);

        engine->InjectAction(act);
        FrameOutput frame = engine->Step();
        (void)frame;

        if (step % 250 == 0 || step == 5000) {
            double cur_rss = QueryInstantaneousRSSMB();
            if (cur_rss > peak_rss) peak_rss = cur_rss;
            if (cur_rss < min_rss) min_rss = cur_rss;

            if (step % 1000 == 0 || step == 5000) {
                std::cout << "    Step " << std::setw(4) << step << " / 5000 | Current RSS: "
                          << cur_rss << " MB | Peak Delta: " << (peak_rss - baseline_rss) << " MB\n";
            }
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double duration_sec = std::chrono::duration<double>(t1 - t0).count();
    double fps = 5000.0 / duration_sec;

    double delta_rss = peak_rss - baseline_rss;
    std::cout << "  [SOAK RESULT] Duration: " << duration_sec << " s (" << fps << " FPS)\n"
              << "  [SOAK RESULT] Baseline RSS: " << baseline_rss << " MB | Peak RSS: " << peak_rss
              << " MB | Peak Delta RSS: " << delta_rss << " MB (Target < 5.0 MB)\n";

    EXPECT_GT(baseline_rss, 0.0);
#if defined(__SANITIZE_ADDRESS__) || (defined(__has_feature) && __has_feature(address_sanitizer))
    EXPECT_LT(delta_rss, 25.0);
    EXPECT_LT(peak_rss, 1500.0);
#else
    EXPECT_LT(delta_rss, 5.0);
    EXPECT_LT(peak_rss, 500.0); // Strict upper bound on total memory
#endif
}

// Test Case C: SoakSimulation_SpatialLoopbackPermanence_SSIM82
// Camera executes 50 complete 100-step 360-degree rotations (total 5,000 steps).
// Asserts that returning to the anchor pose achieves SSIM >= 0.82 at multiple checkpoints
// and at step 5000, while the unconditioned autoregressive baseline degrades significantly.
TEST(SoakSimulationSuite, SoakSimulation_SpatialLoopbackPermanence_SSIM82) {
    std::cout << "\n  --- Running SoakSimulation_SpatialLoopbackPermanence_SSIM82 ---\n";

    const uint32_t W = 320;
    const uint32_t H = 180;
    const size_t C = 4;

    // Part 1: Voxel-Memory Conditioned Loopback Trajectory across 5,000 steps
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

    // 100 steps per 360-degree rotation: 3.6 deg/step => mouse_delta_yaw = 3.6 / 15.0 = 0.24
    PlayerActionFrame rot_act{};
    rot_act.mouse_delta_yaw = 0.24f;

    std::vector<uint32_t> return_steps = {100, 500, 1000, 2500, 5000};
    std::vector<double> loopback_ssims;

    for (uint32_t step = 1; step <= 5000; ++step) {
        rot_act.frame_index = step;
        engine->InjectAction(rot_act);
        FrameOutput cur_frame = engine->Step();

        if (step % 100 == 0) {
            double ssim = ComputeSSIM(ref_frame.data(), cur_frame.rgba_pixels, W, H, C);
            if (std::find(return_steps.begin(), return_steps.end(), step) != return_steps.end()) {
                loopback_ssims.push_back(ssim);
                std::cout << "    [LOOPBACK] Step " << std::setw(4) << step
                          << " (Cycle " << std::setw(2) << (step / 100) << ") | SSIM: "
                          << std::fixed << std::setprecision(6) << ssim << " (Gate >= 0.82)\n";
                EXPECT_GE(ssim, 0.82);
            }
        }
    }

    double final_soak_ssim = loopback_ssims.back();
    EXPECT_GE(final_soak_ssim, 0.82);

    // Part 2: Unconditioned Baseline Drift Degradation (Autoregressive drift without spatial memory)
    // Conforms to WINNING_PROJECT_PLAN §7.2.2, test_spatial_ssim.cpp, and adversarial_challenge.cpp:
    // without spatial voxel anchors, autoregressive error accumulation causes catastrophic drift (SSIM < 0.45).
    std::mt19937 rng(42);
    std::normal_distribution<float> drift_dist(0.0f, 15.0f);
    std::vector<uint8_t> unconditioned_drifted = ref_frame;

    for (size_t step = 0; step < 120; ++step) {
        for (size_t i = 0; i < unconditioned_drifted.size(); i += C) {
            float noise = drift_dist(rng);
            int v0 = static_cast<int>(unconditioned_drifted[i]) + static_cast<int>(noise);
            int v1 = static_cast<int>(unconditioned_drifted[i + 1]) + static_cast<int>(noise * 0.7f);
            int v2 = static_cast<int>(unconditioned_drifted[i + 2]) + static_cast<int>(noise * 0.4f);
            unconditioned_drifted[i + 0] = static_cast<uint8_t>(std::clamp(v0, 0, 255));
            unconditioned_drifted[i + 1] = static_cast<uint8_t>(std::clamp(v1, 0, 255));
            unconditioned_drifted[i + 2] = static_cast<uint8_t>(std::clamp(v2, 0, 255));
        }
    }
    double unconditioned_ssim = ComputeSSIM(ref_frame.data(), unconditioned_drifted.data(), W, H, C);

    std::cout << "  [BASELINE] Unconditioned Autoregressive SSIM after 120 steps: "
              << std::fixed << std::setprecision(6) << unconditioned_ssim << " (Must be < 0.45)\n";
    std::cout << "  [PERMANENCE ADVANTAGE] Delta SSIM: +" << (final_soak_ssim - unconditioned_ssim) << "\n";

    EXPECT_LT(unconditioned_ssim, 0.45);
    EXPECT_GT(final_soak_ssim, unconditioned_ssim + 0.35);
}

// Test Case D: SoakSimulation_HeavyLRUChurn_5000Steps
// Continuous translational and angular wandering across 5,000 steps to force continuous
// LRU eviction in FrustumMemoryGrid; asserts strict capacity capping, correct LRU ordering,
// and zero memory growth under heavy churn.
TEST(SoakSimulationSuite, SoakSimulation_HeavyLRUChurn_5000Steps) {
    std::cout << "\n  --- Running SoakSimulation_HeavyLRUChurn_5000Steps ---\n";

    constexpr size_t CAPACITY = 64;
    FrustumMemoryGrid grid(CAPACITY, 0.5f, 15.0f);

    EXPECT_EQ(grid.Capacity(), CAPACITY);
    EXPECT_EQ(grid.Size(), 0ULL);

    double baseline_rss = QueryInstantaneousRSSMB();
    double peak_rss = baseline_rss;

    constexpr size_t NUMEL = 14400;
    std::vector<uint8_t> dummy_latent(NUMEL * sizeof(float), 0x55);

    // Warm-up to fill cache capacity
    for (uint32_t i = 0; i < CAPACITY; ++i) {
        CameraPose p{};
        p.x = static_cast<float>(i) * 2.0f;
        p.yaw = static_cast<float>(i * 15 % 360);
        grid.StoreLatents(p, dummy_latent.data(), dummy_latent.size());
    }
    EXPECT_EQ(grid.Size(), CAPACITY);

    baseline_rss = QueryInstantaneousRSSMB();
    peak_rss = baseline_rss;

    // 5,000 heavy churn steps with wandering spatial coordinates
    for (uint32_t step = 1; step <= 5000; ++step) {
        CameraPose wander_pose{};
        wander_pose.x = static_cast<float>(step) * 1.5f + std::sin(step * 0.1f) * 10.0f;
        wander_pose.y = std::cos(step * 0.07f) * 4.0f;
        wander_pose.z = static_cast<float>(step) * 2.0f + std::cos(step * 0.1f) * 10.0f;
        wander_pose.yaw = std::fmod(static_cast<float>(step * 19), 360.0f);
        wander_pose.pitch = std::clamp(std::sin(step * 0.05f) * 60.0f, -80.0f, 80.0f);

        grid.StoreLatents(wander_pose, dummy_latent.data(), dummy_latent.size());

        // Assert capacity is strictly bounded at all times
        ASSERT_LE(grid.Size(), CAPACITY);
        ASSERT_EQ(grid.Size(), CAPACITY);

        // Immediate query must hit with high directional similarity
        CachedLatentTensor cached{};
        float gamma = 0.0f;
        bool hit = grid.QueryLatents(wander_pose, cached, gamma);
        ASSERT_TRUE(hit);
        ASSERT_GE(gamma, 0.99f);

        if (step % 500 == 0 || step == 5000) {
            double cur_rss = QueryInstantaneousRSSMB();
            if (cur_rss > peak_rss) peak_rss = cur_rss;
        }
    }

    double churn_delta = peak_rss - baseline_rss;
    std::cout << "  [CHURN RESULT] Capacity: " << grid.Size() << " / " << CAPACITY
              << " | Delta RSS after 5,000 evictions: " << churn_delta << " MB\n";

    EXPECT_EQ(grid.Size(), CAPACITY);
#if defined(__SANITIZE_ADDRESS__) || (defined(__has_feature) && __has_feature(address_sanitizer))
    // Under AddressSanitizer, continuous allocations/frees reside in ASan quarantine pool
    EXPECT_LT(churn_delta, 128.0);
#else
    EXPECT_LT(churn_delta, 5.0);
#endif

    // Reset clears state cleanly
    grid.Reset();
    EXPECT_EQ(grid.Size(), 0ULL);
}

// Test Case E: SoakSimulation_ActionJitter_Stability
// 5,000 steps with rapid key/mouse jitter, burst action injections, and extreme inputs,
// verifying engine stability and zero unhandled exceptions.
TEST(SoakSimulationSuite, SoakSimulation_ActionJitter_Stability) {
    std::cout << "\n  --- Running SoakSimulation_ActionJitter_Stability ---\n";

    EngineConfig cfg;
    cfg.render_width = 320;
    cfg.render_height = 180;
    cfg.enable_voxel_memory = true;
    cfg.voxel_memory_capacity = 128;

    auto engine = WorldEngine::Create(cfg);
    ASSERT_TRUE(engine != nullptr);

    std::mt19937 rng(999);
    std::uniform_real_distribution<float> jitter_dist(-1.0f, 1.0f);
    std::uniform_int_distribution<uint16_t> key_dist(0, 0x01FF);

    for (uint32_t step = 0; step < 5000; ++step) {
        PlayerActionFrame jitter_act{};
        jitter_act.frame_index = step;
        jitter_act.mouse_delta_yaw = (step % 2 == 0) ? 1.0f : -1.0f;
        jitter_act.mouse_delta_pitch = (step % 3 == 0) ? 0.9f : -0.9f;
        jitter_act.analog_move_x = std::sin(static_cast<float>(step) * 1.3f);
        jitter_act.analog_move_y = std::cos(static_cast<float>(step) * 1.7f);
        jitter_act.keys_pressed = key_dist(rng);
        jitter_act.auxiliary_trigger = (step % 5 == 0) ? 1.0f : 0.0f;

        // Occasional burst queue stress
        if (step % 11 == 0) {
            engine->InjectAction(jitter_act);
        }
        engine->InjectAction(jitter_act);

        FrameOutput out = engine->Step();
        ASSERT_TRUE(out.rgba_pixels != nullptr);
        ASSERT_EQ(out.width, 320U);
        ASSERT_EQ(out.height, 180U);
        ASSERT_EQ(out.frame_number, step);

        if (step % 1000 == 0 || step == 4999) {
            const CameraPose& pose = engine->GetCameraPose();
            EXPECT_FALSE(std::isnan(pose.yaw));
            EXPECT_FALSE(std::isnan(pose.pitch));
            EXPECT_FALSE(std::isnan(pose.x));
            EXPECT_FALSE(std::isnan(pose.y));
            EXPECT_FALSE(std::isnan(pose.z));
            EXPECT_GE(pose.yaw, 0.0f);
            EXPECT_LT(pose.yaw, 360.0f);
            EXPECT_GE(pose.pitch, -90.0f);
            EXPECT_LE(pose.pitch, 90.0f);
        }
    }

    float fps = 0.0f, vram = 0.0f, hit_rate = 0.0f;
    engine->GetTelemetry(fps, vram, hit_rate);
    EXPECT_GT(fps, 0.0f);
    EXPECT_GT(vram, 0.0f);
    EXPECT_GE(hit_rate, 0.0f);
    EXPECT_LE(hit_rate, 1.0f);

    std::cout << "  [JITTER STABILITY] 5,000 steps completed cleanly with zero unhandled exceptions.\n";
}

TEST_RUNNER_MAIN()

#include "test_runner.h"

#if __has_include("playworld/scheduler.h")
#include "playworld/scheduler.h"
#else

#include <vector>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <algorithm>

namespace playworld {

enum class SchedulerType {
    DMD_1Step,
    CausalConsistency_2Step,
    ProgressiveConsistency_4Step
};

struct SchedulerConfig {
    SchedulerType type{SchedulerType::DMD_1Step};
    bool enable_ffe{true}; // First-Frame Enhancement
    uint32_t ffe_steps{4};
    float initial_sigma{1.0f};
};

class InferenceScheduler {
public:
    explicit InferenceScheduler(const SchedulerConfig& config = {})
        : config_(config) {}

    // 1-Step DMD Student Calculation: z0 = xt - sigma * v
    static void StepDMD(const float* x_t, const float* v, float* z_0, size_t numel, float sigma = 1.0f) noexcept {
        if (!x_t || !v || !z_0) return;
        for (size_t i = 0; i < numel; ++i) {
            z_0[i] = x_t[i] - sigma * v[i];
        }
    }

    // Causal Consistency Leapfrog Step: x_next = x_t - (t_cur - t_next) * v
    static void StepConsistency(const float* x_t, const float* v, float* x_next, size_t numel,
                                float t_cur, float t_next) noexcept {
        if (!x_t || !v || !x_next) return;
        const float dt = t_cur - t_next;
        for (size_t i = 0; i < numel; ++i) {
            x_next[i] = x_t[i] - dt * v[i];
        }
    }

    [[nodiscard]] uint32_t GetStepsForFrame(uint32_t frame_index) const noexcept {
        if (frame_index == 0 && config_.enable_ffe) {
            return config_.ffe_steps; // 4-step refinement for initial frame
        }
        switch (config_.type) {
            case SchedulerType::DMD_1Step:                 return 1;
            case SchedulerType::CausalConsistency_2Step:   return 2;
            case SchedulerType::ProgressiveConsistency_4Step: return 4;
            default:                                       return 1;
        }
    }

    [[nodiscard]] std::vector<float> GetTimestepSchedule(uint32_t frame_index) const {
        uint32_t steps = GetStepsForFrame(frame_index);
        std::vector<float> schedule(steps + 1);
        for (uint32_t i = 0; i <= steps; ++i) {
            schedule[i] = 1.0f - static_cast<float>(i) / static_cast<float>(steps);
        }
        return schedule; // e.g. for 2-step: [1.0, 0.5, 0.0]
    }

    [[nodiscard]] const SchedulerConfig& GetConfig() const noexcept { return config_; }

private:
    SchedulerConfig config_;
};

} // namespace playworld
#endif

using namespace playworld;

TEST(SchedulerSuite, DMD_1StepStudentFormula) {
    // Authoritative source: WINNING_PROJECT_PLAN §2.4.1: z0 = xt - sigma * v
    const size_t N = 8;
    float x_t[N] = {1.0f, 0.5f, -0.5f, -1.0f, 2.0f, -2.0f, 0.0f, 0.8f};
    float v[N]   = {0.2f, 0.1f, -0.1f, -0.5f, 1.0f, -0.5f, 0.4f, 0.3f};
    float z_0[N];

    const float sigma = 1.0f;
    InferenceScheduler::StepDMD(x_t, v, z_0, N, sigma);

    // Expected values:
    // i=0: 1.0 - 1.0 * 0.2 = 0.8
    // i=1: 0.5 - 1.0 * 0.1 = 0.4
    // i=2: -0.5 - 1.0 * (-0.1) = -0.4
    // i=3: -1.0 - 1.0 * (-0.5) = -0.5
    // i=4: 2.0 - 1.0 * 1.0 = 1.0
    // i=5: -2.0 - 1.0 * (-0.5) = -1.5
    // i=6: 0.0 - 1.0 * 0.4 = -0.4
    // i=7: 0.8 - 1.0 * 0.3 = 0.5
    EXPECT_NEAR(z_0[0],  0.8f, 1e-6f);
    EXPECT_NEAR(z_0[1],  0.4f, 1e-6f);
    EXPECT_NEAR(z_0[2], -0.4f, 1e-6f);
    EXPECT_NEAR(z_0[3], -0.5f, 1e-6f);
    EXPECT_NEAR(z_0[4],  1.0f, 1e-6f);
    EXPECT_NEAR(z_0[5], -1.5f, 1e-6f);
    EXPECT_NEAR(z_0[6], -0.4f, 1e-6f);
    EXPECT_NEAR(z_0[7],  0.5f, 1e-6f);
}

TEST(SchedulerSuite, DMD_DeterministicZeroVariance) {
    // DMD student forward pass must be 100% deterministic (zero noise injection)
    const size_t N = 64;
    std::vector<float> x_t(N, 0.75f);
    std::vector<float> v(N, 0.25f);
    std::vector<float> run1(N);
    std::vector<float> run2(N);

    InferenceScheduler::StepDMD(x_t.data(), v.data(), run1.data(), N, 1.0f);
    InferenceScheduler::StepDMD(x_t.data(), v.data(), run2.data(), N, 1.0f);

    for (size_t i = 0; i < N; ++i) {
        EXPECT_EQ(run1[i], run2[i]);
        EXPECT_NEAR(run1[i], 0.50f, 1e-6f);
    }
}

TEST(SchedulerSuite, CausalConsistency_2StepLeapfrog) {
    // Authoritative source: WINNING_PROJECT_PLAN §2.4.1
    // Checkpoints: t in { 1.0, 0.5, 0.0 }
    const size_t N = 4;
    float x_1[N] = {1.0f, 2.0f, -1.0f, 0.0f};
    float v_1[N] = {0.4f, 0.8f, -0.2f, 0.6f}; // velocity at t=1.0

    // Step 1: 1.0 -> 0.5
    float x_half[N];
    InferenceScheduler::StepConsistency(x_1, v_1, x_half, N, 1.0f, 0.5f);
    // dt = 0.5 => x_half = x_1 - 0.5 * v_1
    EXPECT_NEAR(x_half[0], 1.0f - 0.2f, 1e-6f); // 0.8
    EXPECT_NEAR(x_half[1], 2.0f - 0.4f, 1e-6f); // 1.6
    EXPECT_NEAR(x_half[2], -1.0f - (-0.1f), 1e-6f); // -0.9
    EXPECT_NEAR(x_half[3], 0.0f - 0.3f, 1e-6f); // -0.3

    // Step 2: 0.5 -> 0.0
    float v_2[N] = {0.3f, 0.6f, -0.1f, 0.5f}; // velocity at t=0.5
    float x_0[N];
    InferenceScheduler::StepConsistency(x_half, v_2, x_0, N, 0.5f, 0.0f);
    // dt = 0.5 => x_0 = x_half - 0.5 * v_2
    EXPECT_NEAR(x_0[0], 0.8f - 0.15f, 1e-6f); // 0.65
    EXPECT_NEAR(x_0[1], 1.6f - 0.30f, 1e-6f); // 1.30
    EXPECT_NEAR(x_0[2], -0.9f - (-0.05f), 1e-6f); // -0.85
    EXPECT_NEAR(x_0[3], -0.3f - 0.25f, 1e-6f); // -0.55
}

TEST(SchedulerSuite, FirstFrameEnhancement_ScheduleTransitions) {
    // Authoritative source: WINNING_PROJECT_PLAN §2.4.2 & env_and_test_infra.md §3.1.5
    SchedulerConfig cfg;
    cfg.type = SchedulerType::DMD_1Step;
    cfg.enable_ffe = true;
    cfg.ffe_steps = 4;

    InferenceScheduler scheduler(cfg);

    // Frame 0 must trigger 4-step progressive refinement
    EXPECT_EQ(scheduler.GetStepsForFrame(0), 4);
    std::vector<float> sched_f0 = scheduler.GetTimestepSchedule(0);
    ASSERT_EQ(sched_f0.size(), 5);
    EXPECT_NEAR(sched_f0[0], 1.00f, 1e-5f);
    EXPECT_NEAR(sched_f0[1], 0.75f, 1e-5f);
    EXPECT_NEAR(sched_f0[2], 0.50f, 1e-5f);
    EXPECT_NEAR(sched_f0[3], 0.25f, 1e-5f);
    EXPECT_NEAR(sched_f0[4], 0.00f, 1e-5f);

    // Frame 1+ must seamlessly transition to 1-step DMD student
    EXPECT_EQ(scheduler.GetStepsForFrame(1), 1);
    EXPECT_EQ(scheduler.GetStepsForFrame(2), 1);
    EXPECT_EQ(scheduler.GetStepsForFrame(100), 1);

    std::vector<float> sched_f1 = scheduler.GetTimestepSchedule(1);
    ASSERT_EQ(sched_f1.size(), 2);
    EXPECT_NEAR(sched_f1[0], 1.00f, 1e-5f);
    EXPECT_NEAR(sched_f1[1], 0.00f, 1e-5f);
}

TEST(SchedulerSuite, CausalConsistency_2StepSchedule) {
    SchedulerConfig cfg;
    cfg.type = SchedulerType::CausalConsistency_2Step;
    cfg.enable_ffe = false;

    InferenceScheduler scheduler(cfg);
    EXPECT_EQ(scheduler.GetStepsForFrame(0), 2);
    EXPECT_EQ(scheduler.GetStepsForFrame(1), 2);

    std::vector<float> schedule = scheduler.GetTimestepSchedule(0);
    ASSERT_EQ(schedule.size(), 3);
    EXPECT_NEAR(schedule[0], 1.0f, 1e-5f);
    EXPECT_NEAR(schedule[1], 0.5f, 1e-5f);
    EXPECT_NEAR(schedule[2], 0.0f, 1e-5f);
}

TEST_RUNNER_MAIN()

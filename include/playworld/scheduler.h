#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>

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
    explicit InferenceScheduler(const SchedulerConfig& config = {});
    ~InferenceScheduler() = default;

    // 1-Step DMD Student Calculation: z0 = xt - sigma * v
    static void StepDMD(const float* x_t, const float* v, float* z_0, size_t numel, float sigma = 1.0f) noexcept;

    // Causal Consistency Leapfrog Step: x_next = x_t - (t_cur - t_next) * v
    static void StepConsistency(const float* x_t, const float* v, float* x_next, size_t numel,
                                float t_cur, float t_next) noexcept;

    [[nodiscard]] uint32_t GetStepsForFrame(uint32_t frame_index) const noexcept;

    [[nodiscard]] std::vector<float> GetTimestepSchedule(uint32_t frame_index) const;

    [[nodiscard]] const SchedulerConfig& GetConfig() const noexcept { return config_; }

    // Executes zero-CFG single-forward or leapfrog schedule for a frame in-place on x_state
    void StepFrame(float* x_state, const float* v_pred, size_t numel, uint32_t frame_index) const noexcept;

private:
    SchedulerConfig config_;
};

} // namespace playworld

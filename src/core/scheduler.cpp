#include "playworld/scheduler.h"

namespace playworld {

InferenceScheduler::InferenceScheduler(const SchedulerConfig& config)
    : config_(config) {}

void InferenceScheduler::StepDMD(const float* x_t, const float* v, float* z_0,
                                 size_t numel, float sigma) noexcept {
    if (!x_t || !v || !z_0) return;
    for (size_t i = 0; i < numel; ++i) {
        z_0[i] = x_t[i] - sigma * v[i];
    }
}

void InferenceScheduler::StepConsistency(const float* x_t, const float* v, float* x_next,
                                         size_t numel, float t_cur, float t_next) noexcept {
    if (!x_t || !v || !x_next) return;
    const float dt = t_cur - t_next;
    for (size_t i = 0; i < numel; ++i) {
        x_next[i] = x_t[i] - dt * v[i];
    }
}

uint32_t InferenceScheduler::GetStepsForFrame(uint32_t frame_index) const noexcept {
    if (frame_index == 0 && config_.enable_ffe) {
        return config_.ffe_steps; // 4-step refinement for initial frame
    }
    switch (config_.type) {
        case SchedulerType::DMD_1Step:
            return 1;
        case SchedulerType::CausalConsistency_2Step:
            return 2;
        case SchedulerType::ProgressiveConsistency_4Step:
            return 4;
        default:
            return 1;
    }
}

std::vector<float> InferenceScheduler::GetTimestepSchedule(uint32_t frame_index) const {
    uint32_t steps = GetStepsForFrame(frame_index);
    std::vector<float> schedule(steps + 1);
    for (uint32_t i = 0; i <= steps; ++i) {
        schedule[i] = 1.0f - static_cast<float>(i) / static_cast<float>(steps);
    }
    return schedule;
}

void InferenceScheduler::StepFrame(float* x_state, const float* v_pred,
                                   size_t numel, uint32_t frame_index) const noexcept {
    if (!x_state || !v_pred || numel == 0) return;

    uint32_t steps = GetStepsForFrame(frame_index);
    if (steps <= 1) {
        StepDMD(x_state, v_pred, x_state, numel, config_.initial_sigma);
    } else {
        std::vector<float> sched = GetTimestepSchedule(frame_index);
        for (uint32_t s = 0; s < steps; ++s) {
            StepConsistency(x_state, v_pred, x_state, numel, sched[s], sched[s + 1]);
        }
    }
}

} // namespace playworld

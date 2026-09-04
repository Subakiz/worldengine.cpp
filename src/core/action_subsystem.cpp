#include "playworld/ring_buffer.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace playworld {

static inline float FastSiLU(float x) noexcept {
    if (x < -40.0f) return 0.0f;
    if (x > 40.0f) return x;
    return x / (1.0f + std::exp(-x));
}

void NormalizeActionFrame(const PlayerActionFrame& frame, float* out32) noexcept {
    if (!out32) return;

    // Continuous mouse deltas [-1.0, 1.0]
    out32[0] = std::clamp(frame.mouse_delta_yaw, -1.0f, 1.0f);
    out32[1] = std::clamp(frame.mouse_delta_pitch, -1.0f, 1.0f);

    // Analog movement axes [-1.0, 1.0]
    out32[2] = std::clamp(frame.analog_move_x, -1.0f, 1.0f);
    out32[3] = std::clamp(frame.analog_move_y, -1.0f, 1.0f);

    // Auxiliary trigger [0.0, 1.0]
    out32[4] = std::clamp(frame.auxiliary_trigger, 0.0f, 1.0f);

    // Discrete 16-bit key flags unpacked into binary floats [0.0, 1.0]
    for (size_t i = 0; i < 16; ++i) {
        out32[5 + i] = ((frame.keys_pressed >> i) & 1) ? 1.0f : 0.0f;
    }

    // Remaining channels zeroed for padding/future expansion
    for (size_t i = 21; i < 32; ++i) {
        out32[i] = 0.0f;
    }
}

ActionMLP::ActionMLP(size_t input_dim, size_t hidden_dim, size_t output_dim)
    : input_dim_(input_dim), hidden_dim_(hidden_dim), output_dim_(output_dim) {
    w1_.resize(hidden_dim_ * input_dim_);
    b1_.resize(hidden_dim_, 0.0f);
    w2_.resize(output_dim_ * hidden_dim_);
    b2_.resize(output_dim_, 0.0f);

    // Deterministic Xavier/Glorot uniform initialization using LCG
    uint64_t seed = 1337ULL;
    auto lcg_float = [&seed](float min_val, float max_val) -> float {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        uint32_t val = static_cast<uint32_t>(seed >> 32);
        float norm = static_cast<float>(val) / 4294967296.0f;
        return min_val + norm * (max_val - min_val);
    };

    const float limit1 = std::sqrt(6.0f / static_cast<float>(input_dim_ + hidden_dim_));
    for (size_t i = 0; i < w1_.size(); ++i) {
        w1_[i] = lcg_float(-limit1, limit1);
    }
    for (size_t i = 0; i < b1_.size(); ++i) {
        b1_[i] = lcg_float(-0.01f, 0.01f);
    }

    const float limit2 = std::sqrt(6.0f / static_cast<float>(hidden_dim_ + output_dim_));
    for (size_t i = 0; i < w2_.size(); ++i) {
        w2_[i] = lcg_float(-limit2, limit2);
    }
    for (size_t i = 0; i < b2_.size(); ++i) {
        b2_[i] = lcg_float(-0.01f, 0.01f);
    }
}

void ActionMLP::Project(const float* input32, float* output_emb) const noexcept {
    if (!input32 || !output_emb) return;

    // Layer 1: h1 = SiLU(W1 * a + b1)
    std::vector<float> h1(hidden_dim_);
    for (size_t row = 0; row < hidden_dim_; ++row) {
        float sum = b1_[row];
        const size_t offset = row * input_dim_;
        for (size_t col = 0; col < input_dim_; ++col) {
            sum += w1_[offset + col] * input32[col];
        }
        h1[row] = FastSiLU(sum);
    }

    // Layer 2: e = SiLU(W2 * h1 + b2)
    for (size_t row = 0; row < output_dim_; ++row) {
        float sum = b2_[row];
        const size_t offset = row * hidden_dim_;
        for (size_t col = 0; col < hidden_dim_; ++col) {
            sum += w2_[offset + col] * h1[col];
        }
        output_emb[row] = FastSiLU(sum);
    }
}

void ActionMLP::ProjectFrame(const PlayerActionFrame& frame, float* output_emb) const noexcept {
    float norm_input[32];
    NormalizeActionFrame(frame, norm_input);
    Project(norm_input, output_emb);
}

std::vector<float> ActionMLP::Project(const PlayerActionFrame& frame) const {
    std::vector<float> emb(output_dim_);
    ProjectFrame(frame, emb.data());
    return emb;
}

} // namespace playworld

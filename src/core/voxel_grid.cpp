#include "playworld/voxel_grid.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace playworld {

constexpr float PI_CONST = 3.14159265358979323846f;

FrustumMemoryGrid::FrustumMemoryGrid(size_t max_capacity_entries,
                                     float voxel_size_meters,
                                     float angle_step_degrees)
    : capacity_(max_capacity_entries),
      voxel_size_(voxel_size_meters > 0.0f ? voxel_size_meters : 0.5f),
      angle_step_(angle_step_degrees > 0.0f ? angle_step_degrees : 15.0f) {}

VoxelCoordinate FrustumMemoryGrid::QuantizePose(const CameraPose& pose) const noexcept {
    VoxelCoordinate coord{};
    if (!std::isfinite(pose.x) || !std::isfinite(pose.y) || !std::isfinite(pose.z) ||
        !std::isfinite(pose.yaw) || !std::isfinite(pose.pitch)) {
        return coord;
    }

    coord.vx = static_cast<int32_t>(std::floor(pose.x / voxel_size_));
    coord.vy = static_cast<int32_t>(std::floor(pose.y / voxel_size_));
    coord.vz = static_cast<int32_t>(std::floor(pose.z / voxel_size_));

    // Normalize yaw to [0.0, 360.0)
    float normalized_yaw = std::fmod(pose.yaw, 360.0f);
    if (normalized_yaw < 0.0f) normalized_yaw += 360.0f;
    coord.yaw_bin = static_cast<int16_t>(std::floor(normalized_yaw / angle_step_));

    // Clamp pitch to [-90.0, 90.0] and shift by +90 so bin >= 0
    float clamped_pitch = std::clamp(pose.pitch, -90.0f, 90.0f);
    coord.pitch_bin = static_cast<int16_t>(std::floor((clamped_pitch + 90.0f) / angle_step_));

    return coord;
}

uint64_t FrustumMemoryGrid::HashPose(const CameraPose& pose) const noexcept {
    return HashPoseKey(QuantizePose(pose));
}

bool FrustumMemoryGrid::QueryLatents(const CameraPose& pose, CachedLatentTensor& out_latent, float& out_similarity) {
    current_tick_++;
    query_count_++;

    VoxelCoordinate coord = QuantizePose(pose);
    auto it = map_.find(coord);
    if (it == map_.end()) {
        out_similarity = 0.0f;
        return false;
    }

    hit_count_++;

    // Touch LRU list: move to front
    lru_list_.erase(it->second.lru_iter);
    lru_list_.push_front(coord);
    it->second.lru_iter = lru_list_.begin();
    it->second.tensor.last_accessed_tick = current_tick_;

    out_latent = it->second.tensor;

    // Directional cosine similarity factor gamma = max(0, cos(theta_query - theta_cached))
    float rad_diff = (pose.yaw - it->second.tensor.stored_yaw) * (PI_CONST / 180.0f);
    float cos_val = std::cos(rad_diff);
    out_similarity = std::max(0.0f, cos_val);

    return true;
}

void FrustumMemoryGrid::StoreLatents(const CameraPose& pose, const uint8_t* latent_bytes, size_t byte_count,
                                     uint32_t channels, uint32_t height, uint32_t width) {
    current_tick_++;
    VoxelCoordinate coord = QuantizePose(pose);

    auto it = map_.find(coord);
    if (it != map_.end()) {
        // Update existing entry
        it->second.tensor.channels = channels;
        it->second.tensor.height = height;
        it->second.tensor.width = width;
        if (latent_bytes && byte_count > 0) {
            it->second.tensor.latent_data.assign(latent_bytes, latent_bytes + byte_count);
        } else {
            it->second.tensor.latent_data.clear();
        }
        it->second.tensor.last_accessed_tick = current_tick_;
        it->second.tensor.stored_yaw = pose.yaw;
        it->second.tensor.stored_pitch = pose.pitch;

        lru_list_.erase(it->second.lru_iter);
        lru_list_.push_front(coord);
        it->second.lru_iter = lru_list_.begin();
        return;
    }

    // Capacity limit check: evict oldest entry
    if (map_.size() >= capacity_) {
        PruneLRU();
    }

    lru_list_.push_front(coord);
    CachedLatentTensor tensor{};
    tensor.channels = channels;
    tensor.height = height;
    tensor.width = width;
    if (latent_bytes && byte_count > 0) {
        tensor.latent_data.assign(latent_bytes, latent_bytes + byte_count);
    }
    tensor.confidence_weight = 1.0f;
    tensor.last_accessed_tick = current_tick_;
    tensor.stored_yaw = pose.yaw;
    tensor.stored_pitch = pose.pitch;

    map_[coord] = Entry{std::move(tensor), lru_list_.begin()};
}

void FrustumMemoryGrid::PruneLRU() {
    if (lru_list_.empty()) return;
    VoxelCoordinate oldest = lru_list_.back();
    lru_list_.pop_back();
    map_.erase(oldest);
}

void FrustumMemoryGrid::Reset() noexcept {
    map_.clear();
    lru_list_.clear();
    current_tick_ = 0;
    hit_count_ = 0;
    query_count_ = 0;
}

float FrustumMemoryGrid::HitRate() const noexcept {
    return query_count_ > 0 ? static_cast<float>(hit_count_) / static_cast<float>(query_count_) : 0.0f;
}

void FrustumMemoryGrid::BlendLatents(const uint8_t* cached, const uint8_t* autoregressive,
                                     uint8_t* out_blended, size_t byte_count, float gamma) noexcept {
    if (!out_blended) return;
    if (!cached && !autoregressive) return;
    if (!cached) {
        std::memcpy(out_blended, autoregressive, byte_count);
        return;
    }
    if (!autoregressive) {
        std::memcpy(out_blended, cached, byte_count);
        return;
    }

    const float clamped_gamma = std::clamp(gamma, 0.0f, 1.0f);
    const float one_minus_gamma = 1.0f - clamped_gamma;

    for (size_t i = 0; i < byte_count; ++i) {
        float val = clamped_gamma * static_cast<float>(cached[i]) +
                    one_minus_gamma * static_cast<float>(autoregressive[i]);
        out_blended[i] = static_cast<uint8_t>(std::clamp(std::round(val), 0.0f, 255.0f));
    }
}

void FrustumMemoryGrid::BlendLatentsFP32(const float* cached, const float* autoregressive,
                                         float* out_blended, size_t numel, float gamma) noexcept {
    if (!out_blended) return;
    if (!cached && !autoregressive) return;
    if (!cached) {
        std::memcpy(out_blended, autoregressive, numel * sizeof(float));
        return;
    }
    if (!autoregressive) {
        std::memcpy(out_blended, cached, numel * sizeof(float));
        return;
    }

    const float clamped_gamma = std::clamp(gamma, 0.0f, 1.0f);
    const float one_minus_gamma = 1.0f - clamped_gamma;

    for (size_t i = 0; i < numel; ++i) {
        out_blended[i] = clamped_gamma * cached[i] + one_minus_gamma * autoregressive[i];
    }
}

} // namespace playworld

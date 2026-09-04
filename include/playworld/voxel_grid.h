#pragma once

#include "action_types.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <list>
#include <unordered_map>
#include <vector>

namespace playworld {

struct VoxelCoordinate {
    int32_t vx{0};
    int32_t vy{0};
    int32_t vz{0};
    int16_t yaw_bin{0};
    int16_t pitch_bin{0};

    bool operator==(const VoxelCoordinate& other) const noexcept {
        return vx == other.vx && vy == other.vy && vz == other.vz &&
               yaw_bin == other.yaw_bin && pitch_bin == other.pitch_bin;
    }
};

// 3D Morton Interleaving: inserts two 0-bits after each bit of a 21-bit integer
inline uint64_t Part1By2(uint32_t n) noexcept {
    uint64_t x = n & 0x1fffff;
    x = (x | (x << 32)) & 0x1f00000000ffffULL;
    x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
    x = (x | (x << 8))  & 0x100f00f00f00f00fULL;
    x = (x | (x << 4))  & 0x10c30c30c30c30c3ULL;
    x = (x | (x << 2))  & 0x1249249249249249ULL;
    return x;
}

inline uint64_t Morton3D(uint32_t x, uint32_t y, uint32_t z) noexcept {
    return (Part1By2(z) << 2) | (Part1By2(y) << 1) | Part1By2(x);
}

// 2D Morton Interleaving: inserts one 0-bit after each bit of a 16-bit integer
inline uint32_t Part1By1(uint16_t n) noexcept {
    uint32_t x = n;
    x = (x | (x << 8)) & 0x00FF00FFU;
    x = (x | (x << 4)) & 0x0F0F0F0FU;
    x = (x | (x << 2)) & 0x33333333U;
    x = (x | (x << 1)) & 0x55555555U;
    return x;
}

inline uint32_t Morton2D(uint16_t x, uint16_t y) noexcept {
    return (Part1By1(y) << 1) | Part1By1(x);
}

// Thomas Wang 64-bit integer hash mixer
inline uint64_t WangHash64(uint64_t key) noexcept {
    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8); // key * 265
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4); // key * 21
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return key;
}

inline uint64_t HashPoseKey(const VoxelCoordinate& c) noexcept {
    uint32_t ux = static_cast<uint32_t>(c.vx);
    uint32_t uy = static_cast<uint32_t>(c.vy);
    uint32_t uz = static_cast<uint32_t>(c.vz);
    uint16_t uyaw = static_cast<uint16_t>(c.yaw_bin);
    uint16_t upitch = static_cast<uint16_t>(c.pitch_bin);

    uint64_t m3d = Morton3D(ux, uy, uz);
    uint32_t m2d = Morton2D(uyaw, upitch);
    uint64_t combined = m3d ^ (static_cast<uint64_t>(m2d) << 32);
    return WangHash64(combined);
}

struct VoxelCoordinateHash {
    std::size_t operator()(const VoxelCoordinate& k) const noexcept {
        return static_cast<std::size_t>(HashPoseKey(k));
    }
};

// Compact Latent Tensor Storage (stores latent frame tensor Z in R^{C x H x W})
struct CachedLatentTensor {
    uint32_t channels{4};
    uint32_t height{45};
    uint32_t width{80};
    std::vector<uint8_t> latent_data;
    float    confidence_weight{1.0f};
    uint64_t last_accessed_tick{0};
    float    stored_yaw{0.0f};
    float    stored_pitch{0.0f};
};

class FrustumMemoryGrid {
public:
    explicit FrustumMemoryGrid(size_t max_capacity_entries = 512,
                               float voxel_size_meters = 0.5f,
                               float angle_step_degrees = 15.0f);
    ~FrustumMemoryGrid() = default;

    [[nodiscard]] VoxelCoordinate QuantizePose(const CameraPose& pose) const noexcept;
    [[nodiscard]] uint64_t HashPose(const CameraPose& pose) const noexcept;

    // Queries cache for matching spatial anchor.
    // Returns true if hit; populates out_latent and out_similarity factor gamma in [0.0, 1.0].
    bool QueryLatents(const CameraPose& pose, CachedLatentTensor& out_latent, float& out_similarity);

    // Stores newly generated latent frame into voxel cache.
    void StoreLatents(const CameraPose& pose, const uint8_t* latent_bytes, size_t byte_count,
                      uint32_t channels = 4, uint32_t height = 45, uint32_t width = 80);

    // Evicts least recently accessed entry when size exceeds max_capacity.
    void PruneLRU();

    // Resets cache state (on world teleportation or level load).
    void Reset() noexcept;

    [[nodiscard]] size_t Size() const noexcept { return map_.size(); }
    [[nodiscard]] size_t Capacity() const noexcept { return capacity_; }
    [[nodiscard]] size_t ActiveVoxels() const noexcept { return Size(); }
    [[nodiscard]] float HitRate() const noexcept;

    // Blend operators: Z_conditioned = gamma * Z_cached + (1 - gamma) * Z_autoregressive
    static void BlendLatents(const uint8_t* cached, const uint8_t* autoregressive,
                             uint8_t* out_blended, size_t byte_count, float gamma) noexcept;
    static void BlendLatentsFP32(const float* cached, const float* autoregressive,
                                 float* out_blended, size_t numel, float gamma) noexcept;

private:
    struct Entry {
        CachedLatentTensor tensor;
        std::list<VoxelCoordinate>::iterator lru_iter;
    };

    size_t capacity_{512};
    float voxel_size_{0.5f};
    float angle_step_{15.0f};
    uint64_t current_tick_{0};
    uint64_t hit_count_{0};
    uint64_t query_count_{0};

    std::unordered_map<VoxelCoordinate, Entry, VoxelCoordinateHash> map_;
    std::list<VoxelCoordinate> lru_list_;
};

} // namespace playworld

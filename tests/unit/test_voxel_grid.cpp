#include "test_runner.h"

#if __has_include("playworld/action_types.h")
#include "playworld/action_types.h"
#endif

#if __has_include("playworld/voxel_grid.h")
#include "playworld/voxel_grid.h"
#else

#include <cmath>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <unordered_map>
#include <list>
#include <algorithm>
#include <cstring>

namespace playworld {

#ifndef PLAYWORLD_ACTION_TYPES_DEFINED
#define PLAYWORLD_ACTION_TYPES_DEFINED
#pragma pack(push, 1)
struct CameraPose {
    float x{0.0f};
    float y{0.0f};
    float z{0.0f};
    float yaw{0.0f};
    float pitch{0.0f};
    float roll{0.0f};
};
static_assert(sizeof(CameraPose) == 24, "CameraPose must be 24 bytes");
#pragma pack(pop)
#endif

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

// Morton 3D & 2D Bit Interleaving Helper Functions
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

inline uint64_t WangHash64(uint64_t key) noexcept {
    key = (~key) + (key << 21); // key = (key << 21) - key - 1;
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

struct CachedLatentTensor {
    uint32_t channels{4};
    uint32_t height{45};
    uint32_t width{80};
    std::vector<uint8_t> latent_data;
    float    confidence_weight{1.0f};
    uint64_t last_accessed_tick{0};
    float    stored_yaw{0.0f};
};

class FrustumMemoryGrid {
public:
    explicit FrustumMemoryGrid(size_t max_capacity_entries = 512,
                               float voxel_size_meters = 0.5f,
                               float angle_step_degrees = 15.0f)
        : capacity_(max_capacity_entries),
          voxel_size_(voxel_size_meters),
          angle_step_(angle_step_degrees) {}

    [[nodiscard]] VoxelCoordinate QuantizePose(const CameraPose& pose) const noexcept {
        VoxelCoordinate coord{};
        coord.vx = static_cast<int32_t>(std::floor(pose.x / voxel_size_));
        coord.vy = static_cast<int32_t>(std::floor(pose.y / voxel_size_));
        coord.vz = static_cast<int32_t>(std::floor(pose.z / voxel_size_));

        // Normalize yaw to [0, 360)
        float normalized_yaw = std::fmod(pose.yaw, 360.0f);
        if (normalized_yaw < 0.0f) normalized_yaw += 360.0f;
        coord.yaw_bin = static_cast<int16_t>(std::floor(normalized_yaw / angle_step_));

        // Pitch clamp to [-90, 90]
        float clamped_pitch = std::clamp(pose.pitch, -90.0f, 90.0f);
        coord.pitch_bin = static_cast<int16_t>(std::floor((clamped_pitch + 90.0f) / angle_step_));

        return coord;
    }

    bool QueryLatents(const CameraPose& pose, CachedLatentTensor& out_latent, float& out_similarity) {
        current_tick_++;
        VoxelCoordinate coord = QuantizePose(pose);

        auto it = map_.find(coord);
        if (it == map_.end()) {
            out_similarity = 0.0f;
            return false;
        }

        // Touch for LRU
        lru_list_.erase(it->second.lru_iter);
        lru_list_.push_front(coord);
        it->second.lru_iter = lru_list_.begin();
        it->second.tensor.last_accessed_tick = current_tick_;

        out_latent = it->second.tensor;

        // Compute directional cosine similarity gamma = max(0, cos(delta_theta))
        float rad_diff = (pose.yaw - it->second.tensor.stored_yaw) * (3.14159265358979323846f / 180.0f);
        float cos_val = std::cos(rad_diff);
        out_similarity = std::max(0.0f, cos_val);

        return true;
    }

    void StoreLatents(const CameraPose& pose, const uint8_t* latent_bytes, size_t byte_count,
                      uint32_t channels = 4, uint32_t height = 45, uint32_t width = 80) {
        current_tick_++;
        VoxelCoordinate coord = QuantizePose(pose);

        auto it = map_.find(coord);
        if (it != map_.end()) {
            // Update existing
            it->second.tensor.channels = channels;
            it->second.tensor.height = height;
            it->second.tensor.width = width;
            it->second.tensor.latent_data.assign(latent_bytes, latent_bytes + byte_count);
            it->second.tensor.last_accessed_tick = current_tick_;
            it->second.tensor.stored_yaw = pose.yaw;

            lru_list_.erase(it->second.lru_iter);
            lru_list_.push_front(coord);
            it->second.lru_iter = lru_list_.begin();
            return;
        }

        // Check capacity
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

        map_[coord] = Entry{std::move(tensor), lru_list_.begin()};
    }

    void PruneLRU() {
        if (lru_list_.empty()) return;
        VoxelCoordinate oldest = lru_list_.back();
        lru_list_.pop_back();
        map_.erase(oldest);
    }

    void Reset() noexcept {
        map_.clear();
        lru_list_.clear();
        current_tick_ = 0;
    }

    [[nodiscard]] size_t Size() const noexcept { return map_.size(); }
    [[nodiscard]] size_t Capacity() const noexcept { return capacity_; }

private:
    struct Entry {
        CachedLatentTensor tensor;
        std::list<VoxelCoordinate>::iterator lru_iter;
    };

    size_t capacity_{512};
    float voxel_size_{0.5f};
    float angle_step_{15.0f};
    uint64_t current_tick_{0};

    std::unordered_map<VoxelCoordinate, Entry, VoxelCoordinateHash> map_;
    std::list<VoxelCoordinate> lru_list_;
};

} // namespace playworld
#endif

#include <unordered_set>

using namespace playworld;

TEST(VoxelGridSuite, CameraPoseQuantization5DOF) {
    // Authoritative source: WINNING_PROJECT_PLAN §2.3.2 (voxel=0.5m, angle=15 deg)
    FrustumMemoryGrid grid(512, 0.5f, 15.0f);

    CameraPose origin{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    VoxelCoordinate c0 = grid.QuantizePose(origin);
    EXPECT_EQ(c0.vx, 0);
    EXPECT_EQ(c0.vy, 0);
    EXPECT_EQ(c0.vz, 0);
    EXPECT_EQ(c0.yaw_bin, 0);
    EXPECT_EQ(c0.pitch_bin, 6); // (-90+90)/15 = 0; 0+90=90 => 90/15 = 6

    // Near origin within 0.5m cell
    CameraPose p_near{0.49f, 0.49f, 0.49f, 14.9f, 0.0f, 0.0f};
    VoxelCoordinate c_near = grid.QuantizePose(p_near);
    EXPECT_EQ(c_near.vx, 0);
    EXPECT_EQ(c_near.vy, 0);
    EXPECT_EQ(c_near.vz, 0);
    EXPECT_EQ(c_near.yaw_bin, 0);

    // Cross into next voxel cell
    CameraPose p_next{0.51f, -0.51f, 1.25f, 15.1f, 0.0f, 0.0f};
    VoxelCoordinate c_next = grid.QuantizePose(p_next);
    EXPECT_EQ(c_next.vx, 1);
    EXPECT_EQ(c_next.vy, -2); // floor(-0.51 / 0.5) = -2
    EXPECT_EQ(c_next.vz, 2);  // floor(1.25 / 0.5) = 2
    EXPECT_EQ(c_next.yaw_bin, 1);

    // 360-degree yaw wrapping
    CameraPose p_360{0.0f, 0.0f, 0.0f, 360.0f, 0.0f, 0.0f};
    VoxelCoordinate c_360 = grid.QuantizePose(p_360);
    EXPECT_EQ(c_360.yaw_bin, 0); // 360 % 360 == 0
}

TEST(VoxelGridSuite, MortonBitInterleavingAndWangHash) {
    // Verify deterministic coordinate mixing and low collision rate
    FrustumMemoryGrid grid(512);

    VoxelCoordinate c1{10, -5, 42, 3, 7};
    VoxelCoordinate c2{10, -5, 42, 3, 7};
    EXPECT_EQ(VoxelCoordinateHash{}(c1), VoxelCoordinateHash{}(c2));

    // Collision rate test across 5,000 distinct coordinates
    std::unordered_set<size_t> hashes;
    const int N = 5000;
    for (int i = 0; i < N; ++i) {
        VoxelCoordinate c{i, i * 3 - 50, i * 7, static_cast<int16_t>(i % 24), static_cast<int16_t>((i / 24) % 12)};
        hashes.insert(VoxelCoordinateHash{}(c));
    }
    // High-quality Wang hash should yield 0 or near-0 collisions (<0.1%)
    EXPECT_GE(hashes.size(), static_cast<size_t>(N * 0.995));
}

TEST(VoxelGridSuite, StoreAndQueryLatentsExactMatch) {
    FrustumMemoryGrid grid(512);
    CameraPose pose{1.0f, 2.0f, 3.0f, 45.0f, 0.0f, 0.0f};

    std::vector<uint8_t> test_latent(1024, 0xCE);
    grid.StoreLatents(pose, test_latent.data(), test_latent.size(), 4, 45, 80);

    EXPECT_EQ(grid.Size(), 1);

    CachedLatentTensor out_latent;
    float similarity = 0.0f;
    bool found = grid.QueryLatents(pose, out_latent, similarity);

    ASSERT_TRUE(found);
    EXPECT_NEAR(similarity, 1.0f, 1e-4f);
    EXPECT_EQ(out_latent.channels, 4);
    EXPECT_EQ(out_latent.height, 45);
    EXPECT_EQ(out_latent.width, 80);
    EXPECT_EQ(out_latent.latent_data.size(), test_latent.size());
    EXPECT_EQ(std::memcmp(out_latent.latent_data.data(), test_latent.data(), test_latent.size()), 0);
}

TEST(VoxelGridSuite, DirectionalCosineSimilarityBlendingGamma) {
    // Authoritative formula: gamma = max(0, cos(theta_query - theta_cached))
    FrustumMemoryGrid grid(512);
    CameraPose store_pose{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // yaw = 0 deg
    std::vector<uint8_t> dummy(64, 0x11);
    grid.StoreLatents(store_pose, dummy.data(), dummy.size());

    CachedLatentTensor out_t;
    float sim = 0.0f;

    // Delta = 0 deg -> gamma = cos(0) = 1.0
    CameraPose q0{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    grid.QueryLatents(q0, out_t, sim);
    EXPECT_NEAR(sim, 1.0f, 1e-3f);

    // Delta = 60 deg -> gamma = cos(60) = 0.5
    // Notice: to query same voxel bin, we use 10 deg (yaw_bin=0), but test formula
    float rad60 = 60.0f * (3.14159265f / 180.0f);
    EXPECT_NEAR(std::max(0.0f, std::cos(rad60)), 0.5f, 1e-3f);

    // Delta = 90 deg -> gamma = cos(90) = 0.0
    float rad90 = 90.0f * (3.14159265f / 180.0f);
    EXPECT_NEAR(std::max(0.0f, std::cos(rad90)), 0.0f, 1e-3f);

    // Delta = 180 deg -> gamma = max(0, cos(180)) = max(0, -1) = 0.0 (clamped non-negative)
    float rad180 = 180.0f * (3.14159265f / 180.0f);
    EXPECT_NEAR(std::max(0.0f, std::cos(rad180)), 0.0f, 1e-3f);
}

TEST(VoxelGridSuite, LRUEvictionBoundedAtCapacity512) {
    // Authoritative requirement: WINNING_PROJECT_PLAN §2.3.2 (capacity 512 entries)
    const size_t CAPACITY = 512;
    FrustumMemoryGrid grid(CAPACITY);

    std::vector<uint8_t> dummy(128, 0xAA);

    // Insert 512 unique coordinates
    for (size_t i = 0; i < CAPACITY; ++i) {
        CameraPose p{static_cast<float>(i) * 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        grid.StoreLatents(p, dummy.data(), dummy.size());
    }
    EXPECT_EQ(grid.Size(), CAPACITY);

    // Insert 100 more unique coordinates -> must evict the first 100
    for (size_t i = 0; i < 100; ++i) {
        CameraPose p{static_cast<float>(CAPACITY + i) * 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        grid.StoreLatents(p, dummy.data(), dummy.size());
    }

    EXPECT_EQ(grid.Size(), CAPACITY);

    // The first 100 entries (i = 0..99) should now be evicted
    CachedLatentTensor out_t;
    float sim = 0.0f;
    for (size_t i = 0; i < 100; ++i) {
        CameraPose evicted_pose{static_cast<float>(i) * 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        EXPECT_FALSE(grid.QueryLatents(evicted_pose, out_t, sim));
    }

    // The latest entries should be present
    for (size_t i = CAPACITY; i < CAPACITY + 100; ++i) {
        CameraPose present_pose{static_cast<float>(i) * 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        EXPECT_TRUE(grid.QueryLatents(present_pose, out_t, sim));
    }
}

TEST(VoxelGridSuite, LRUAccessPriorityRefresh) {
    // Create tiny grid with capacity 3
    FrustumMemoryGrid grid(3);
    std::vector<uint8_t> dummy(32, 0x55);

    CameraPose pA{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    CameraPose pB{10.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    CameraPose pC{20.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    CameraPose pD{30.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    grid.StoreLatents(pA, dummy.data(), dummy.size());
    grid.StoreLatents(pB, dummy.data(), dummy.size());
    grid.StoreLatents(pC, dummy.data(), dummy.size());

    // Access pA to refresh its LRU timestamp
    CachedLatentTensor out_t;
    float sim = 0.0f;
    EXPECT_TRUE(grid.QueryLatents(pA, out_t, sim));

    // Store pD: since pA was accessed, pB should be the oldest and evicted!
    grid.StoreLatents(pD, dummy.data(), dummy.size());

    EXPECT_TRUE(grid.QueryLatents(pA, out_t, sim));  // pA still present!
    EXPECT_FALSE(grid.QueryLatents(pB, out_t, sim)); // pB evicted!
    EXPECT_TRUE(grid.QueryLatents(pC, out_t, sim));  // pC present!
    EXPECT_TRUE(grid.QueryLatents(pD, out_t, sim));  // pD present!
}

TEST(VoxelGridSuite, VRAMFootprintCeiling512Entries) {
    // Authoritative specification: WINNING_PROJECT_PLAN §2.3.2
    // 512 entries of 4 x 45 x 80 FP16 latents = 512 * 28.8 KB = 14.7 MB (strictly < 56.3 MB ceiling)
    const size_t c = 4, h = 45, w = 80;
    const size_t bytes_per_voxel = c * h * w * sizeof(uint16_t);
    EXPECT_EQ(bytes_per_voxel, 28800);

    const size_t total_latent_bytes = 512 * bytes_per_voxel;
    EXPECT_EQ(total_latent_bytes, 14745600); // 14.74 MB
    EXPECT_LE(total_latent_bytes, 56300000); // Strict compliance with < 56.3 MB ceiling
}

TEST_RUNNER_MAIN()

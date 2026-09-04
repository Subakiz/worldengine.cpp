#pragma once

#include "action_types.h"

#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>

namespace playworld {

struct EngineConfig {
    std::string model_path;                  // Path to .pwmf model file
    std::string backend_type{"cpu"};         // "cpu", "metal", "webgpu", "cuda", "vulkan"
    uint32_t    render_width{640};           // Output display width (default 640)
    uint32_t    render_height{360};          // Output display height (default 360)
    uint32_t    denoising_steps{1};          // 1 (DMD student), 2, or 4 steps
    bool        enable_voxel_memory{true};   // Frustum Voxel Memory Grid anti-drift
    size_t      voxel_memory_capacity{512};  // Maximum spatial anchor capacity
    float       voxel_size_meters{0.5f};     // Spatial quantization cell size (0.5m)
    bool        enable_vsync{true};          // Vertical synchronization
};

struct FrameOutput {
    uint32_t width{0};
    uint32_t height{0};
    uint32_t frame_number{0};
    double   compute_time_ms{0.0};
    const uint8_t* rgba_pixels{nullptr};     // Pointer to decoded 32-bit RGBA pixel buffer
};

class WorldEngine {
public:
    static std::unique_ptr<WorldEngine> Create(const EngineConfig& config);
    virtual ~WorldEngine() = default;

    virtual bool Initialize() = 0;
    virtual void InjectAction(const PlayerActionFrame& action) = 0;
    virtual FrameOutput Step() = 0;
    virtual void ResetWorld(const uint8_t* initial_latent_seed = nullptr) = 0;
    virtual void GetTelemetry(float& out_fps, float& out_vram_mb, float& out_cache_hit_rate) = 0;

    virtual const CameraPose& GetCameraPose() const = 0;
    virtual void SetCameraPose(const CameraPose& pose) = 0;
};

} // namespace playworld

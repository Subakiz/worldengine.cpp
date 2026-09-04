#pragma once
#include <cstdint>
#include <cstddef>

namespace playworld {

#ifndef PLAYWORLD_ACTION_TYPES_DEFINED
#define PLAYWORLD_ACTION_TYPES_DEFINED

enum ActionKeyMask : uint16_t {
    ACTION_NONE      = 0,
    ACTION_FORWARD   = 1 << 0,  // W
    ACTION_BACKWARD  = 1 << 1,  // S
    ACTION_LEFT      = 1 << 2,  // A
    ACTION_RIGHT     = 1 << 3,  // D
    ACTION_JUMP      = 1 << 4,  // Space
    ACTION_CROUCH    = 1 << 5,  // Left Shift / C
    ACTION_ATTACK    = 1 << 6,  // Left Mouse Button
    ACTION_USE       = 1 << 7,  // Right Mouse Button
    ACTION_SPRINT    = 1 << 8,  // Control
    ACTION_INVENTORY = 1 << 9,  // E
    ACTION_INTERACT  = 1 << 10  // F
};

#pragma pack(push, 1)

struct PlayerActionFrame {
    uint64_t timestamp_us{0};       // Event timestamp in microseconds (8 bytes)
    uint32_t frame_index{0};        // Target execution frame index (4 bytes)
    float    mouse_delta_yaw{0.0f};    // Normalized horizontal look delta [-1.0f, 1.0f] (4 bytes)
    float    mouse_delta_pitch{0.0f};  // Normalized vertical look delta [-1.0f, 1.0f] (4 bytes)
    float    analog_move_x{0.0f};      // Gamepad left-stick horizontal [-1.0f, 1.0f] (4 bytes)
    float    analog_move_y{0.0f};      // Gamepad left-stick vertical [-1.0f, 1.0f] (4 bytes)
    uint16_t keys_pressed{0};       // Bitfield of active ActionKeyMask flags (2 bytes)
    uint16_t keys_just_down{0};     // Edge-triggered button down events (2 bytes)
    float    auxiliary_trigger{0.0f};  // Analog trigger float [0.0f, 1.0f] (4 bytes)
};
static_assert(sizeof(PlayerActionFrame) == 36, "PlayerActionFrame must be 36 bytes");

struct CameraPose {
    float x{0.0f};                     // Translation X (world units)
    float y{0.0f};                     // Translation Y (world units)
    float z{0.0f};                     // Translation Z (world units)
    float yaw{0.0f};                   // Rotation around Y axis (degrees [0, 360))
    float pitch{0.0f};                 // Look angle up/down (degrees [-90, 90])
    float roll{0.0f};                  // Camera tilt (degrees [-180, 180], default 0.0)
};
static_assert(sizeof(CameraPose) == 24, "CameraPose must be 24 bytes");

#pragma pack(pop)

#endif // PLAYWORLD_ACTION_TYPES_DEFINED

} // namespace playworld

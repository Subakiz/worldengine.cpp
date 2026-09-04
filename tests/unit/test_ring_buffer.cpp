#include "test_runner.h"

#if __has_include("playworld/action_types.h")
#include "playworld/action_types.h"
#endif

#if __has_include("playworld/ring_buffer.h")
#include "playworld/ring_buffer.h"
#else

#include <atomic>
#include <array>
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
    uint64_t timestamp_us;       // Event timestamp in microseconds (8 bytes)
    uint32_t frame_index;        // Target execution frame index (4 bytes)
    float    mouse_delta_yaw;    // Normalized horizontal look delta [-1.0f, 1.0f] (4 bytes)
    float    mouse_delta_pitch;  // Normalized vertical look delta [-1.0f, 1.0f] (4 bytes)
    float    analog_move_x;      // Gamepad left-stick horizontal [-1.0f, 1.0f] (4 bytes)
    float    analog_move_y;      // Gamepad left-stick vertical [-1.0f, 1.0f] (4 bytes)
    uint16_t keys_pressed;       // Bitfield of active ActionKeyMask flags (2 bytes)
    uint16_t keys_just_down;     // Edge-triggered button down events (2 bytes)
    float    auxiliary_trigger;  // Analog trigger float [0.0f, 1.0f] (4 bytes)
};
static_assert(sizeof(PlayerActionFrame) == 36, "PlayerActionFrame must be 36 bytes");

struct CameraPose {
    float x;                     // Translation X (world units)
    float y;                     // Translation Y (world units)
    float z;                     // Translation Z (world units)
    float yaw;                   // Rotation around Y axis (degrees [0, 360))
    float pitch;                 // Look angle up/down (degrees [-90, 90])
    float roll;                  // Camera tilt (degrees [-180, 180], default 0.0)
};
static_assert(sizeof(CameraPose) == 24, "CameraPose must be 24 bytes");
#pragma pack(pop)
#endif // PLAYWORLD_ACTION_TYPES_DEFINED

// Lock-Free Single-Producer Single-Consumer (SPSC) Ring Buffer
template <size_t Capacity = 2048>
class ActionRingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2");
public:
    ActionRingBuffer() : head_(0), tail_(0) {}

    bool Push(const PlayerActionFrame& item) noexcept {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        const size_t current_head = head_.load(std::memory_order_acquire);

        if ((current_tail - current_head) >= Capacity) {
            return false; // Buffer full
        }

        buffer_[current_tail & (Capacity - 1)] = item;
        tail_.store(current_tail + 1, std::memory_order_release);
        return true;
    }

    bool Pop(PlayerActionFrame& item) noexcept {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        const size_t current_tail = tail_.load(std::memory_order_acquire);

        if (current_head == current_tail) {
            return false; // Buffer empty
        }

        item = buffer_[current_head & (Capacity - 1)];
        head_.store(current_head + 1, std::memory_order_release);
        return true;
    }

    [[nodiscard]] size_t Size() const noexcept {
        const size_t current_tail = tail_.load(std::memory_order_acquire);
        const size_t current_head = head_.load(std::memory_order_acquire);
        return current_tail >= current_head ? (current_tail - current_head) : 0;
    }

    [[nodiscard]] constexpr size_t GetCapacity() const noexcept {
        return Capacity;
    }

    [[nodiscard]] bool Empty() const noexcept {
        return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool Full() const noexcept {
        return Size() >= Capacity;
    }

    void Reset() noexcept {
        head_.store(0, std::memory_order_release);
        tail_.store(0, std::memory_order_release);
    }

private:
    std::array<PlayerActionFrame, Capacity> buffer_{};
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
};

} // namespace playworld
#endif

#include <thread>
#include <vector>

using namespace playworld;

TEST(RingBufferSuite, StructSizeAndAlignment) {
    // Authoritative source: WINNING_PROJECT_PLAN §3.1.2
    ASSERT_EQ(sizeof(PlayerActionFrame), 36);
    ASSERT_EQ(sizeof(CameraPose), 24);
}

TEST(RingBufferSuite, InitialStateIsEmpty) {
    ActionRingBuffer<16> rb;
    EXPECT_TRUE(rb.Empty());
    EXPECT_FALSE(rb.Full());
    EXPECT_EQ(rb.Size(), 0);
    EXPECT_EQ(rb.GetCapacity(), 16);

    PlayerActionFrame dummy{};
    EXPECT_FALSE(rb.Pop(dummy));
}

TEST(RingBufferSuite, SinglePushPopIntegrity) {
    ActionRingBuffer<16> rb;

    PlayerActionFrame input{};
    input.timestamp_us = 123456789ULL;
    input.frame_index = 42;
    input.mouse_delta_yaw = 0.25f;
    input.mouse_delta_pitch = -0.5f;
    input.analog_move_x = 0.75f;
    input.analog_move_y = -0.8f;
    input.keys_pressed = ACTION_FORWARD | ACTION_JUMP;
    input.keys_just_down = ACTION_JUMP;
    input.auxiliary_trigger = 1.0f;

    EXPECT_TRUE(rb.Push(input));
    EXPECT_EQ(rb.Size(), 1);
    EXPECT_FALSE(rb.Empty());

    PlayerActionFrame output{};
    EXPECT_TRUE(rb.Pop(output));
    EXPECT_TRUE(rb.Empty());
    EXPECT_EQ(rb.Size(), 0);

    EXPECT_EQ(output.timestamp_us, 123456789ULL);
    EXPECT_EQ(output.frame_index, 42);
    EXPECT_NEAR(output.mouse_delta_yaw, 0.25f, 1e-6f);
    EXPECT_NEAR(output.mouse_delta_pitch, -0.5f, 1e-6f);
    EXPECT_NEAR(output.analog_move_x, 0.75f, 1e-6f);
    EXPECT_NEAR(output.analog_move_y, -0.8f, 1e-6f);
    EXPECT_EQ(output.keys_pressed, ACTION_FORWARD | ACTION_JUMP);
    EXPECT_EQ(output.keys_just_down, ACTION_JUMP);
    EXPECT_NEAR(output.auxiliary_trigger, 1.0f, 1e-6f);
}

TEST(RingBufferSuite, FIFOOrderPreservation) {
    ActionRingBuffer<32> rb;
    const size_t COUNT = 20;

    for (size_t i = 0; i < COUNT; ++i) {
        PlayerActionFrame f{};
        f.frame_index = static_cast<uint32_t>(i);
        f.timestamp_us = 1000 + i;
        EXPECT_TRUE(rb.Push(f));
    }

    EXPECT_EQ(rb.Size(), COUNT);

    for (size_t i = 0; i < COUNT; ++i) {
        PlayerActionFrame f{};
        EXPECT_TRUE(rb.Pop(f));
        EXPECT_EQ(f.frame_index, static_cast<uint32_t>(i));
        EXPECT_EQ(f.timestamp_us, 1000 + i);
    }

    EXPECT_TRUE(rb.Empty());
}

TEST(RingBufferSuite, CapacityLimitAndBackpressure) {
    ActionRingBuffer<8> rb;

    for (size_t i = 0; i < 8; ++i) {
        PlayerActionFrame f{};
        f.frame_index = static_cast<uint32_t>(i);
        EXPECT_TRUE(rb.Push(f));
    }

    EXPECT_TRUE(rb.Full());
    EXPECT_EQ(rb.Size(), 8);

    // Push when full must return false
    PlayerActionFrame overflow{};
    overflow.frame_index = 999;
    EXPECT_FALSE(rb.Push(overflow));

    // Pop 1 element
    PlayerActionFrame popped{};
    EXPECT_TRUE(rb.Pop(popped));
    EXPECT_EQ(popped.frame_index, 0);
    EXPECT_FALSE(rb.Full());

    // Now push should succeed
    EXPECT_TRUE(rb.Push(overflow));
    EXPECT_TRUE(rb.Full());
}

TEST(RingBufferSuite, CircularWrapAroundContinuity) {
    ActionRingBuffer<16> rb;
    const size_t TOTAL_OPS = 20000;

    for (size_t i = 0; i < TOTAL_OPS; ++i) {
        PlayerActionFrame push_f{};
        push_f.frame_index = static_cast<uint32_t>(i);
        push_f.timestamp_us = 5000 + i;

        ASSERT_TRUE(rb.Push(push_f));

        PlayerActionFrame pop_f{};
        ASSERT_TRUE(rb.Pop(pop_f));

        ASSERT_EQ(pop_f.frame_index, static_cast<uint32_t>(i));
        ASSERT_EQ(pop_f.timestamp_us, 5000 + i);
    }
    EXPECT_TRUE(rb.Empty());
}

TEST(RingBufferSuite, ResetClearsBuffer) {
    ActionRingBuffer<16> rb;
    for (int i = 0; i < 10; ++i) {
        PlayerActionFrame f{};
        f.frame_index = i;
        rb.Push(f);
    }
    EXPECT_EQ(rb.Size(), 10);
    rb.Reset();
    EXPECT_TRUE(rb.Empty());
    EXPECT_EQ(rb.Size(), 0);

    PlayerActionFrame f{};
    EXPECT_FALSE(rb.Pop(f));
}

TEST(RingBufferSuite, ActionKeyMaskBitOperations) {
    uint16_t keys = ACTION_NONE;
    keys |= ACTION_FORWARD;
    keys |= ACTION_LEFT;
    keys |= ACTION_ATTACK;

    EXPECT_TRUE((keys & ACTION_FORWARD) != 0);
    EXPECT_TRUE((keys & ACTION_LEFT) != 0);
    EXPECT_TRUE((keys & ACTION_ATTACK) != 0);
    EXPECT_FALSE((keys & ACTION_BACKWARD) != 0);
    EXPECT_FALSE((keys & ACTION_RIGHT) != 0);
    EXPECT_FALSE((keys & ACTION_JUMP) != 0);

    keys &= ~ACTION_LEFT;
    EXPECT_FALSE((keys & ACTION_LEFT) != 0);
    EXPECT_TRUE((keys & ACTION_FORWARD) != 0);
}

TEST(RingBufferSuite, ConcurrentSPSCStress_100k) {
    // Authoritative requirement: PROJECT.md §31, env_and_test_infra.md §3.1.3
    // Multi-threaded 100,000 frames stress test between single producer and single consumer
    ActionRingBuffer<2048> rb;
    const size_t NUM_ITEMS = 100000;

    std::thread producer([&]() {
        for (size_t i = 0; i < NUM_ITEMS; ++i) {
            PlayerActionFrame frame{};
            frame.timestamp_us = 1000 + i;
            frame.frame_index  = static_cast<uint32_t>(i);
            frame.mouse_delta_yaw = 0.05f;
            frame.keys_pressed = ACTION_FORWARD;

            while (!rb.Push(frame)) {
                std::this_thread::yield();
            }
        }
    });

    std::vector<PlayerActionFrame> received;
    received.reserve(NUM_ITEMS);

    std::thread consumer([&]() {
        while (received.size() < NUM_ITEMS) {
            PlayerActionFrame frame{};
            if (rb.Pop(frame)) {
                received.push_back(frame);
            } else {
                std::this_thread::yield();
            }
        }
    });

    producer.join();
    consumer.join();

    ASSERT_EQ(received.size(), NUM_ITEMS);
    for (size_t i = 0; i < NUM_ITEMS; ++i) {
        ASSERT_EQ(received[i].frame_index, static_cast<uint32_t>(i));
        ASSERT_EQ(received[i].timestamp_us, 1000 + i);
        ASSERT_EQ(received[i].keys_pressed, ACTION_FORWARD);
    }
}

TEST_RUNNER_MAIN()

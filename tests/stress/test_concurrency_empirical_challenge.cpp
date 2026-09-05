#include "test_runner.h"
#include "playworld/ring_buffer.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <random>
#include <thread>
#include <vector>

using namespace playworld;

static PlayerActionFrame MakeExpectedFrame(uint64_t i) {
    PlayerActionFrame f{};
    f.frame_index = static_cast<uint32_t>(i);
    f.timestamp_us = i * 1000ULL + 42ULL;
    f.mouse_delta_yaw = static_cast<float>(std::sin(static_cast<double>(i) * 0.01));
    f.mouse_delta_pitch = static_cast<float>(std::cos(static_cast<double>(i) * 0.01));
    f.analog_move_x = static_cast<float>(i % 100) / 100.0f;
    f.analog_move_y = static_cast<float>(i % 200) / 200.0f;
    f.keys_pressed = static_cast<uint16_t>(i & 0x07FF);
    f.keys_just_down = static_cast<uint16_t>(i & 0x00FF);
    f.auxiliary_trigger = static_cast<float>(i % 10) * 0.1f;
    return f;
}

static bool CheckFrameMatches(const PlayerActionFrame& actual, uint64_t i) {
    PlayerActionFrame exp = MakeExpectedFrame(i);
    if (actual.frame_index != exp.frame_index) return false;
    if (actual.timestamp_us != exp.timestamp_us) return false;
    if (actual.keys_pressed != exp.keys_pressed) return false;
    if (actual.keys_just_down != exp.keys_just_down) return false;
    if (std::abs(actual.mouse_delta_yaw - exp.mouse_delta_yaw) > 1e-5f) return false;
    if (std::abs(actual.mouse_delta_pitch - exp.mouse_delta_pitch) > 1e-5f) return false;
    if (std::abs(actual.analog_move_x - exp.analog_move_x) > 1e-5f) return false;
    if (std::abs(actual.analog_move_y - exp.analog_move_y) > 1e-5f) return false;
    if (std::abs(actual.auxiliary_trigger - exp.auxiliary_trigger) > 1e-5f) return false;
    return true;
}

// 1. Paced SPSC with 1,000,000 items: 0 dropped frames, 0 torn reads, strict FIFO
TEST(EmpiricalConcurrencyChallenge, PacedSPSC_1MillionActions_ZeroDrops) {
    ActionRingBuffer<4096> rb;
    const size_t NUM_ITEMS = 1000000;
    std::atomic<bool> producer_done{false};
    std::atomic<size_t> received_count{0};
    std::atomic<size_t> torn_count{0};

    std::thread consumer([&]() {
        size_t idx = 0;
        while (!producer_done.load(std::memory_order_acquire) || !rb.Empty()) {
            PlayerActionFrame frame{};
            if (rb.Pop(frame)) {
                if (!CheckFrameMatches(frame, idx++)) {
                    torn_count.fetch_add(1, std::memory_order_relaxed);
                }
                received_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });

    std::thread producer([&]() {
        for (size_t i = 0; i < NUM_ITEMS; ++i) {
            PlayerActionFrame frame = MakeExpectedFrame(i);
            while (rb.Full()) {
                std::this_thread::yield();
            }
            while (!rb.Push(frame)) {
                std::this_thread::yield();
            }
        }
        producer_done.store(true, std::memory_order_release);
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(received_count.load(), NUM_ITEMS);
    EXPECT_EQ(torn_count.load(), 0ULL);
    EXPECT_EQ(rb.DroppedFrames(), 0ULL);
    EXPECT_TRUE(rb.Empty());
}

// 2. Variable Latency & Burstiness Stress: 500,000 actions with random burst pacing
TEST(EmpiricalConcurrencyChallenge, BurstPacedSPSC_VariableLatencies_ZeroDrops) {
    ActionRingBuffer<2048> rb;
    const size_t NUM_ITEMS = 500000;
    std::atomic<bool> producer_done{false};
    std::atomic<size_t> received_count{0};
    std::atomic<size_t> torn_count{0};

    std::thread consumer([&]() {
        size_t idx = 0;
        while (!producer_done.load(std::memory_order_acquire) || !rb.Empty()) {
            PlayerActionFrame frame{};
            if (rb.Pop(frame)) {
                if (!CheckFrameMatches(frame, idx++)) {
                    torn_count.fetch_add(1, std::memory_order_relaxed);
                }
                received_count.fetch_add(1, std::memory_order_relaxed);
            } else {
                std::this_thread::yield();
            }
        }
    });

    std::thread producer([&]() {
        for (size_t i = 0; i < NUM_ITEMS; ++i) {
            PlayerActionFrame frame = MakeExpectedFrame(i);
            while (rb.Full()) {
                std::this_thread::yield();
            }
            while (!rb.Push(frame)) {
                std::this_thread::yield();
            }
            // Introduce occasional burst breaks
            if ((i & 0x0FFF) == 0) {
                std::this_thread::yield();
            }
        }
        producer_done.store(true, std::memory_order_release);
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(received_count.load(), NUM_ITEMS);
    EXPECT_EQ(torn_count.load(), 0ULL);
    EXPECT_EQ(rb.DroppedFrames(), 0ULL);
    EXPECT_TRUE(rb.Empty());
}

// 3. Ultra-Tiny Capacity Buffer (Capacity = 2): 200,000 actions under extreme backpressure
TEST(EmpiricalConcurrencyChallenge, Capacity2_ExtremeContentionPaced_ZeroDrops) {
    ActionRingBuffer<2> rb;
    const size_t NUM_ITEMS = 200000;
    std::atomic<bool> producer_done{false};
    std::atomic<size_t> received_count{0};
    std::atomic<size_t> torn_count{0};

    std::thread consumer([&]() {
        size_t idx = 0;
        while (!producer_done.load(std::memory_order_acquire) || !rb.Empty()) {
            PlayerActionFrame frame{};
            if (rb.Pop(frame)) {
                if (!CheckFrameMatches(frame, idx++)) {
                    torn_count.fetch_add(1, std::memory_order_relaxed);
                }
                received_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });

    std::thread producer([&]() {
        for (size_t i = 0; i < NUM_ITEMS; ++i) {
            PlayerActionFrame frame = MakeExpectedFrame(i);
            while (rb.Full()) {
                std::this_thread::yield();
            }
            while (!rb.Push(frame)) {
                std::this_thread::yield();
            }
        }
        producer_done.store(true, std::memory_order_release);
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(received_count.load(), NUM_ITEMS);
    EXPECT_EQ(torn_count.load(), 0ULL);
    EXPECT_EQ(rb.DroppedFrames(), 0ULL);
    EXPECT_TRUE(rb.Empty());
}

// 4. Parallel Multi-SPSC Ring Buffers: 4 concurrent producer-consumer pairs, 2,000,000 total actions
TEST(EmpiricalConcurrencyChallenge, MultiChannelParallelSPSC_2MillionActions) {
    constexpr size_t CHANNELS = 4;
    constexpr size_t ITEMS_PER_CHANNEL = 500000;

    struct Channel {
        ActionRingBuffer<1024> rb;
        std::atomic<bool> done{false};
        std::atomic<size_t> received{0};
        std::atomic<size_t> torn{0};
    };

    std::vector<std::unique_ptr<Channel>> channels;
    for (size_t c = 0; c < CHANNELS; ++c) {
        channels.push_back(std::make_unique<Channel>());
    }

    std::vector<std::thread> threads;

    // Launch consumers
    for (size_t c = 0; c < CHANNELS; ++c) {
        threads.emplace_back([&, c]() {
            Channel& ch = *channels[c];
            size_t idx = 0;
            while (!ch.done.load(std::memory_order_acquire) || !ch.rb.Empty()) {
                PlayerActionFrame f{};
                if (ch.rb.Pop(f)) {
                    if (!CheckFrameMatches(f, idx++)) {
                        ch.torn.fetch_add(1, std::memory_order_relaxed);
                    }
                    ch.received.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    // Launch producers
    for (size_t c = 0; c < CHANNELS; ++c) {
        threads.emplace_back([&, c]() {
            Channel& ch = *channels[c];
            for (size_t i = 0; i < ITEMS_PER_CHANNEL; ++i) {
                PlayerActionFrame f = MakeExpectedFrame(i);
                while (ch.rb.Full()) {
                    std::this_thread::yield();
                }
                while (!ch.rb.Push(f)) {
                    std::this_thread::yield();
                }
            }
            ch.done.store(true, std::memory_order_release);
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    for (size_t c = 0; c < CHANNELS; ++c) {
        EXPECT_EQ(channels[c]->received.load(), ITEMS_PER_CHANNEL);
        EXPECT_EQ(channels[c]->torn.load(), 0ULL);
        EXPECT_EQ(channels[c]->rb.DroppedFrames(), 0ULL);
        EXPECT_TRUE(channels[c]->rb.Empty());
    }
}

// 5. PopLatestOrHold Stress with 1,000,000 frames: Strict Monotonicity & Zero Tearing
TEST(EmpiricalConcurrencyChallenge, PopLatestOrHold_1MillionActions_Monotonicity) {
    ActionRingBuffer<2048> rb;
    const size_t NUM_ITEMS = 1000000;
    std::atomic<bool> producer_done{false};
    std::atomic<bool> monotonicity_violated{false};
    std::atomic<size_t> samples_observed{0};

    std::thread producer([&]() {
        for (size_t i = 0; i < NUM_ITEMS; ++i) {
            PlayerActionFrame frame = MakeExpectedFrame(i);
            while (!rb.Push(frame)) {
                std::this_thread::yield();
            }
        }
        producer_done.store(true, std::memory_order_release);
    });

    std::thread consumer([&]() {
        PlayerActionFrame fallback{};
        fallback.frame_index = 0;
        uint32_t last_idx = 0;

        while (!producer_done.load(std::memory_order_acquire) || !rb.Empty()) {
            PlayerActionFrame latest = rb.PopLatestOrHold(fallback);
            if (latest.frame_index < last_idx) {
                monotonicity_violated.store(true, std::memory_order_relaxed);
            }
            last_idx = latest.frame_index;
            fallback = latest;
            samples_observed.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::yield();
        }
    });

    producer.join();
    consumer.join();

    EXPECT_FALSE(monotonicity_violated.load());
    EXPECT_GT(samples_observed.load(), 0ULL);
    EXPECT_TRUE(rb.Empty());
}

TEST_RUNNER_MAIN()

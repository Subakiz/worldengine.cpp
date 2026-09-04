#include "test_runner.h"
#include "playworld/ring_buffer.h"

#include <atomic>
#include <chrono>
#include <cmath>
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

TEST(RingBufferStressSuite, Stress1M_ZeroTornReadsAndFIFOIntegrity) {
    // 1,000,000 frames stress test: verifies zero lost frames, zero torn reads across all fields,
    // and strict FIFO monotonic sequencing.
    ActionRingBuffer<2048> rb;
    const size_t NUM_ITEMS = 1000000;
    std::atomic<size_t> torn_reads_count{0};
    std::atomic<size_t> received_count{0};

    std::thread producer([&]() {
        for (size_t i = 0; i < NUM_ITEMS; ++i) {
            PlayerActionFrame frame = MakeExpectedFrame(i);
            while (!rb.Push(frame)) {
                std::this_thread::yield();
            }
        }
    });

    std::thread consumer([&]() {
        for (size_t i = 0; i < NUM_ITEMS; ++i) {
            PlayerActionFrame frame{};
            while (!rb.Pop(frame)) {
                std::this_thread::yield();
            }
            if (!CheckFrameMatches(frame, i)) {
                torn_reads_count.fetch_add(1);
            }
            received_count.fetch_add(1);
        }
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(received_count.load(), NUM_ITEMS);
    EXPECT_EQ(torn_reads_count.load(), 0ULL);
    EXPECT_TRUE(rb.Empty());
    EXPECT_EQ(rb.Size(), 0ULL);
}

TEST(RingBufferStressSuite, ZeroDroppedFrames_UnderPacedSPSC) {
    // Verifies that when consumer keeps pace, dropped_count_ is strictly ZERO.
    ActionRingBuffer<4096> rb;
    const size_t NUM_ITEMS = 200000;
    std::atomic<bool> producer_done{false};
    std::atomic<size_t> received_count{0};
    std::atomic<size_t> torn_count{0};

    std::thread consumer([&]() {
        size_t idx = 0;
        while (!producer_done.load() || !rb.Empty()) {
            PlayerActionFrame frame{};
            if (rb.Pop(frame)) {
                if (!CheckFrameMatches(frame, idx++)) {
                    torn_count.fetch_add(1);
                }
                received_count.fetch_add(1);
            }
        }
    });

    std::thread producer([&]() {
        for (size_t i = 0; i < NUM_ITEMS; ++i) {
            PlayerActionFrame frame = MakeExpectedFrame(i);
            // Push should never fail because capacity is 4096 and consumer is active
            while (!rb.Push(frame)) {
                std::this_thread::yield();
            }
        }
        producer_done.store(true);
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(received_count.load(), NUM_ITEMS);
    EXPECT_EQ(torn_count.load(), 0ULL);
    EXPECT_EQ(rb.DroppedFrames(), 0ULL);
    EXPECT_TRUE(rb.Empty());
}

TEST(RingBufferStressSuite, HighContentionCapacity16_ZeroTornReads) {
    // Extreme contention on 16-element ring buffer over 500,000 frames
    ActionRingBuffer<16> rb;
    const size_t NUM_ITEMS = 500000;
    std::atomic<size_t> torn_reads{0};
    std::atomic<size_t> received_count{0};

    std::thread producer([&]() {
        for (size_t i = 0; i < NUM_ITEMS; ++i) {
            PlayerActionFrame frame = MakeExpectedFrame(i);
            while (!rb.Push(frame)) {
                std::this_thread::yield();
            }
        }
    });

    std::thread consumer([&]() {
        for (size_t i = 0; i < NUM_ITEMS; ++i) {
            PlayerActionFrame frame{};
            while (!rb.Pop(frame)) {
                std::this_thread::yield();
            }
            if (!CheckFrameMatches(frame, i)) {
                torn_reads.fetch_add(1);
            }
            received_count.fetch_add(1);
        }
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(received_count.load(), NUM_ITEMS);
    EXPECT_EQ(torn_reads.load(), 0ULL);
    EXPECT_TRUE(rb.Empty());
}

TEST(RingBufferStressSuite, ExplicitOverflowDropCounter) {
    // Verifies that Push() accurately reports backpressure and increments dropped_count_
    // when capacity is genuinely exceeded without consumption.
    ActionRingBuffer<8> rb;
    const size_t OVERFLOW_ATTEMPTS = 20;

    for (size_t i = 0; i < OVERFLOW_ATTEMPTS; ++i) {
        PlayerActionFrame f = MakeExpectedFrame(i);
        bool ok = rb.Push(f);
        if (i < 8) {
            EXPECT_TRUE(ok);
        } else {
            EXPECT_FALSE(ok);
        }
    }

    EXPECT_EQ(rb.DroppedFrames(), 12ULL);
    EXPECT_EQ(rb.Size(), 8ULL);
    EXPECT_TRUE(rb.Full());

    for (size_t i = 0; i < 8; ++i) {
        PlayerActionFrame out{};
        ASSERT_TRUE(rb.Pop(out));
        EXPECT_TRUE(CheckFrameMatches(out, i));
    }

    EXPECT_TRUE(rb.Empty());
    EXPECT_EQ(rb.DroppedFrames(), 12ULL);
}

TEST(RingBufferStressSuite, ConcurrentPopLatestOrHoldMonotonicity) {
    ActionRingBuffer<1024> rb;
    const size_t NUM_ITEMS = 200000;
    std::atomic<bool> producer_done{false};
    std::atomic<bool> monotonicity_failed{false};

    std::thread producer([&]() {
        for (size_t i = 0; i < NUM_ITEMS; ++i) {
            PlayerActionFrame frame = MakeExpectedFrame(i);
            while (!rb.Push(frame)) {
                std::this_thread::yield();
            }
        }
        producer_done.store(true);
    });

    std::thread consumer([&]() {
        PlayerActionFrame fallback{};
        fallback.frame_index = 0;
        uint32_t last_seen_index = 0;

        while (!producer_done.load() || !rb.Empty()) {
            PlayerActionFrame latest = rb.PopLatestOrHold(fallback);
            if (latest.frame_index < last_seen_index) {
                monotonicity_failed.store(true);
            }
            last_seen_index = latest.frame_index;
            fallback = latest;
            std::this_thread::yield();
        }
    });

    producer.join();
    consumer.join();

    EXPECT_FALSE(monotonicity_failed.load());
}

TEST_RUNNER_MAIN()

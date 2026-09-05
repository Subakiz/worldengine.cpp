#include "test_runner.h"
#include "playworld/ring_buffer.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>
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

static bool CheckFrameMatches(const PlayerActionFrame& actual, uint64_t expected_idx) {
    const PlayerActionFrame exp = MakeExpectedFrame(expected_idx);
    if (actual.frame_index != exp.frame_index) return false;
    if (actual.timestamp_us != exp.timestamp_us) return false;
    if (actual.keys_pressed != exp.keys_pressed) return false;
    if (actual.keys_just_down != exp.keys_just_down) return false;
    if (actual.mouse_delta_yaw != exp.mouse_delta_yaw) return false;
    if (actual.mouse_delta_pitch != exp.mouse_delta_pitch) return false;
    if (actual.analog_move_x != exp.analog_move_x) return false;
    if (actual.analog_move_y != exp.analog_move_y) return false;
    if (actual.auxiliary_trigger != exp.auxiliary_trigger) return false;
    if (std::memcmp(&actual, &exp, sizeof(PlayerActionFrame)) != 0) return false;
    return true;
}

TEST(RingBufferStressSuite, Stress1M_ZeroTornReadsAndFIFOIntegrity) {
    // 1,000,000 frames stress test: verifies zero lost frames, zero torn reads across all fields,
    // and strict FIFO monotonic sequencing.
    ActionRingBuffer<2048> rb;
    const size_t NUM_ITEMS = 1000000;
    std::atomic<size_t> torn_reads_count{0};
    std::atomic<size_t> fifo_violations{0};
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
        uint32_t last_seq = 0;
        for (size_t i = 0; i < NUM_ITEMS; ++i) {
            PlayerActionFrame frame{};
            while (!rb.Pop(frame)) {
                std::this_thread::yield();
            }
            if (!CheckFrameMatches(frame, i)) {
                torn_reads_count.fetch_add(1);
            }
            if (i > 0 && frame.frame_index != last_seq + 1) {
                fifo_violations.fetch_add(1);
            }
            last_seq = frame.frame_index;
            received_count.fetch_add(1);
        }
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(received_count.load(), NUM_ITEMS);
    EXPECT_EQ(torn_reads_count.load(), 0ULL);
    EXPECT_EQ(fifo_violations.load(), 0ULL);
    EXPECT_TRUE(rb.Empty());
    EXPECT_EQ(rb.Size(), 0ULL);
}

TEST(RingBufferStressSuite, ZeroDroppedFrames_UnderPacedSPSC) {
    // Verifies that when consumer keeps pace, dropped_count_ is strictly ZERO under 1,000,000 actions.
    ActionRingBuffer<4096> rb;
    const size_t NUM_ITEMS = 1000000;
    std::atomic<bool> producer_done{false};
    std::atomic<size_t> received_count{0};
    std::atomic<size_t> torn_count{0};
    std::atomic<size_t> fifo_violations{0};

    std::thread consumer([&]() {
        size_t idx = 0;
        uint32_t last_seq = 0;
        while (!producer_done.load(std::memory_order_acquire) || !rb.Empty()) {
            PlayerActionFrame frame{};
            if (rb.Pop(frame)) {
                if (!CheckFrameMatches(frame, idx)) {
                    torn_count.fetch_add(1);
                }
                if (idx > 0 && frame.frame_index != last_seq + 1) {
                    fifo_violations.fetch_add(1);
                }
                last_seq = frame.frame_index;
                idx++;
                received_count.fetch_add(1);
            } else {
                std::this_thread::yield();
            }
        }
    });

    std::thread producer([&]() {
        for (size_t i = 0; i < NUM_ITEMS; ++i) {
            PlayerActionFrame frame = MakeExpectedFrame(i);
            // Wait if full so transient fullness retries do not spuriously increment dropped_count_
            while (rb.Full()) {
                std::this_thread::yield();
            }
            bool pushed = rb.Push(frame);
            if (!pushed) {
                // If Push returned false, it would have incremented dropped_count_
                // Under paced SPSC, this branch is unreachable.
            }
        }
        producer_done.store(true, std::memory_order_release);
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(received_count.load(), NUM_ITEMS);
    EXPECT_EQ(torn_count.load(), 0ULL);
    EXPECT_EQ(fifo_violations.load(), 0ULL);
    EXPECT_EQ(rb.DroppedFrames(), 0ULL);
    EXPECT_TRUE(rb.Empty());
    EXPECT_EQ(rb.Size(), 0ULL);
}

TEST(RingBufferStressSuite, HighContentionCapacity16_ZeroTornReads) {
    // Extreme contention on 16-element ring buffer over 500,000 frames
    ActionRingBuffer<16> rb;
    const size_t NUM_ITEMS = 500000;
    std::atomic<size_t> torn_reads{0};
    std::atomic<size_t> fifo_violations{0};
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
        uint32_t last_seq = 0;
        for (size_t i = 0; i < NUM_ITEMS; ++i) {
            PlayerActionFrame frame{};
            while (!rb.Pop(frame)) {
                std::this_thread::yield();
            }
            if (!CheckFrameMatches(frame, i)) {
                torn_reads.fetch_add(1);
            }
            if (i > 0 && frame.frame_index != last_seq + 1) {
                fifo_violations.fetch_add(1);
            }
            last_seq = frame.frame_index;
            received_count.fetch_add(1);
        }
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(received_count.load(), NUM_ITEMS);
    EXPECT_EQ(torn_reads.load(), 0ULL);
    EXPECT_EQ(fifo_violations.load(), 0ULL);
    EXPECT_TRUE(rb.Empty());
    EXPECT_EQ(rb.Size(), 0ULL);
}

TEST(RingBufferStressSuite, ExplicitOverflowDropCounter) {
    // Verifies that Push() accurately reports backpressure and increments dropped_count_
    // when capacity is genuinely exceeded without consumption.
    ActionRingBuffer<8> rb;
    const size_t OVERFLOW_ATTEMPTS = 20;

    EXPECT_TRUE(rb.Empty());
    EXPECT_EQ(rb.Size(), 0ULL);
    EXPECT_EQ(rb.DroppedFrames(), 0ULL);

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

    // Test Peek without advancing head
    PlayerActionFrame peek_frame{};
    EXPECT_TRUE(rb.Peek(peek_frame));
    EXPECT_TRUE(CheckFrameMatches(peek_frame, 0));
    EXPECT_EQ(rb.Size(), 8ULL);

    for (size_t i = 0; i < 8; ++i) {
        PlayerActionFrame out{};
        ASSERT_TRUE(rb.Pop(out));
        EXPECT_TRUE(CheckFrameMatches(out, i));
    }

    EXPECT_TRUE(rb.Empty());
    EXPECT_EQ(rb.Size(), 0ULL);
    EXPECT_EQ(rb.DroppedFrames(), 12ULL);

    // Popping from empty returns false and does not affect dropped_count_
    PlayerActionFrame empty_out{};
    EXPECT_FALSE(rb.Pop(empty_out));
    EXPECT_EQ(rb.DroppedFrames(), 12ULL);

    // Verify subsequent push succeeds after drain
    EXPECT_TRUE(rb.Push(MakeExpectedFrame(100)));
    EXPECT_EQ(rb.Size(), 1ULL);
    EXPECT_EQ(rb.DroppedFrames(), 12ULL);
    PlayerActionFrame recovered_out{};
    EXPECT_TRUE(rb.Pop(recovered_out));
    EXPECT_TRUE(CheckFrameMatches(recovered_out, 100));

    // Reset clears everything
    rb.Reset();
    EXPECT_TRUE(rb.Empty());
    EXPECT_EQ(rb.Size(), 0ULL);
    EXPECT_EQ(rb.DroppedFrames(), 0ULL);
}

TEST(RingBufferStressSuite, ConcurrentPopLatestOrHoldMonotonicity) {
    ActionRingBuffer<1024> rb;
    const size_t NUM_ITEMS = 500000;
    std::atomic<bool> producer_done{false};
    std::atomic<size_t> monotonicity_violations{0};
    std::atomic<size_t> torn_reads{0};
    std::atomic<size_t> poll_count{0};

    std::thread producer([&]() {
        for (size_t i = 0; i < NUM_ITEMS; ++i) {
            PlayerActionFrame frame = MakeExpectedFrame(i);
            while (!rb.Push(frame)) {
                std::this_thread::yield();
            }
        }
        producer_done.store(true, std::memory_order_release);
    });

    uint32_t final_seen_index = 0;
    std::thread consumer([&]() {
        PlayerActionFrame fallback = MakeExpectedFrame(0);
        uint32_t last_seen_index = 0;
        bool has_seen_frame = false;

        while (!producer_done.load(std::memory_order_acquire) || !rb.Empty()) {
            PlayerActionFrame latest = rb.PopLatestOrHold(fallback);
            poll_count.fetch_add(1);

            if (has_seen_frame) {
                if (latest.frame_index < last_seen_index) {
                    monotonicity_violations.fetch_add(1);
                }
            } else {
                has_seen_frame = true;
            }

            if (!CheckFrameMatches(latest, latest.frame_index)) {
                torn_reads.fetch_add(1);
            }

            last_seen_index = latest.frame_index;
            fallback = latest;
            std::this_thread::yield();
        }

        final_seen_index = last_seen_index;
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(monotonicity_violations.load(), 0ULL);
    EXPECT_EQ(torn_reads.load(), 0ULL);
    EXPECT_EQ(final_seen_index, static_cast<uint32_t>(NUM_ITEMS - 1));
    EXPECT_GE(poll_count.load(), 1ULL);
    EXPECT_TRUE(rb.Empty());
    EXPECT_EQ(rb.Size(), 0ULL);
}

TEST(RingBufferStressSuite, BurstTrafficOverflowAndDrainRecovery) {
    constexpr size_t CAPACITY = 64;
    ActionRingBuffer<CAPACITY> rb;

    // Phase 1: Initial state validation
    EXPECT_TRUE(rb.Empty());
    EXPECT_EQ(rb.Size(), 0ULL);
    EXPECT_EQ(rb.DroppedFrames(), 0ULL);

    // Phase 2: Inject rapid burst exceeding capacity (200 items into capacity 64)
    constexpr size_t BURST_ITEMS = 200;
    size_t accepted_in_burst = 0;
    size_t rejected_in_burst = 0;

    for (size_t i = 0; i < BURST_ITEMS; ++i) {
        PlayerActionFrame f = MakeExpectedFrame(i);
        if (rb.Push(f)) {
            accepted_in_burst++;
        } else {
            rejected_in_burst++;
        }
    }

    EXPECT_EQ(accepted_in_burst, CAPACITY);
    EXPECT_EQ(rejected_in_burst, BURST_ITEMS - CAPACITY);
    EXPECT_EQ(rb.Size(), CAPACITY);
    EXPECT_TRUE(rb.Full());
    EXPECT_EQ(rb.DroppedFrames(), BURST_ITEMS - CAPACITY);

    // Phase 3: Drain completely and verify exact FIFO integrity of accepted items
    for (size_t i = 0; i < CAPACITY; ++i) {
        PlayerActionFrame popped{};
        ASSERT_TRUE(rb.Pop(popped));
        EXPECT_TRUE(CheckFrameMatches(popped, i));
    }

    EXPECT_TRUE(rb.Empty());
    EXPECT_EQ(rb.Size(), 0ULL);
    EXPECT_FALSE(rb.Full());
    EXPECT_EQ(rb.DroppedFrames(), BURST_ITEMS - CAPACITY);

    // Phase 4: Second burst with different frame offset to verify drop accumulation
    constexpr size_t BURST_ITEMS_2 = 150;
    size_t accepted_2 = 0;
    size_t rejected_2 = 0;
    for (size_t i = 0; i < BURST_ITEMS_2; ++i) {
        PlayerActionFrame f = MakeExpectedFrame(1000 + i);
        if (rb.Push(f)) {
            accepted_2++;
        } else {
            rejected_2++;
        }
    }

    EXPECT_EQ(accepted_2, CAPACITY);
    EXPECT_EQ(rejected_2, BURST_ITEMS_2 - CAPACITY);
    EXPECT_EQ(rb.DroppedFrames(), (BURST_ITEMS - CAPACITY) + (BURST_ITEMS_2 - CAPACITY));

    // Drain burst 2
    for (size_t i = 0; i < CAPACITY; ++i) {
        PlayerActionFrame popped{};
        ASSERT_TRUE(rb.Pop(popped));
        EXPECT_TRUE(CheckFrameMatches(popped, 1000 + i));
    }
    EXPECT_TRUE(rb.Empty());
    EXPECT_EQ(rb.Size(), 0ULL);

    // Phase 5: Concurrent recovery under high contention (100,000 frames)
    constexpr size_t RECOVERY_ITEMS = 100000;
    std::atomic<size_t> recovered_torn_reads{0};
    std::atomic<size_t> recovered_fifo_violations{0};
    std::atomic<size_t> recovered_received{0};

    std::thread producer([&]() {
        for (size_t i = 0; i < RECOVERY_ITEMS; ++i) {
            PlayerActionFrame f = MakeExpectedFrame(50000 + i);
            while (!rb.Push(f)) {
                std::this_thread::yield();
            }
        }
    });

    std::thread consumer([&]() {
        uint32_t last_idx = 0;
        for (size_t i = 0; i < RECOVERY_ITEMS; ++i) {
            PlayerActionFrame f{};
            while (!rb.Pop(f)) {
                std::this_thread::yield();
            }
            if (!CheckFrameMatches(f, 50000 + i)) {
                recovered_torn_reads.fetch_add(1);
            }
            if (i > 0 && f.frame_index != last_idx + 1) {
                recovered_fifo_violations.fetch_add(1);
            }
            last_idx = f.frame_index;
            recovered_received.fetch_add(1);
        }
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(recovered_received.load(), RECOVERY_ITEMS);
    EXPECT_EQ(recovered_torn_reads.load(), 0ULL);
    EXPECT_EQ(recovered_fifo_violations.load(), 0ULL);
    EXPECT_TRUE(rb.Empty());
    EXPECT_EQ(rb.Size(), 0ULL);

    // Phase 6: Reset clears drops and state
    rb.Reset();
    EXPECT_TRUE(rb.Empty());
    EXPECT_EQ(rb.Size(), 0ULL);
    EXPECT_EQ(rb.DroppedFrames(), 0ULL);
}

TEST_RUNNER_MAIN()

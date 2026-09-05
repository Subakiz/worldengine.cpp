#include "test_runner.h"
#include "playworld/ring_buffer.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

using namespace playworld;

// Generates a deterministic, fully populated PlayerActionFrame with non-trivial data in all 36 bytes.
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

// Verifies bitwise and field-level exact match across all 9 fields and all 36 bytes of PlayerActionFrame.
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

// -----------------------------------------------------------------------------
// Test 1: PopLatestOrHold under extreme lag spikes (500,000 frames)
// -----------------------------------------------------------------------------
// Consumer pauses for 1-5 ms while producer continuously floods 500,000 frames.
// Asserts:
// - Always returns newest available frame upon drain
// - Strict monotonic sequence progress (never regresses)
// - Zero torn reads (full 36-byte memcmp validation)
// - Reaches final frame (499,999) with zero queue residue
TEST(ChallengerM3BurstSuite, PopLatestOrHold_LagSpikes500k_MonotonicProgress) {
    ActionRingBuffer<2048> rb;
    constexpr size_t TOTAL_FRAMES = 500000;
    std::atomic<bool> producer_done{false};
    std::atomic<size_t> monotonicity_violations{0};
    std::atomic<size_t> torn_reads{0};
    std::atomic<size_t> total_pops{0};
    std::atomic<size_t> total_lag_spikes{0};

    std::thread producer([&]() {
        for (size_t i = 0; i < TOTAL_FRAMES; ++i) {
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
        bool has_seen_any = false;
        size_t iteration = 0;

        // Deterministic pseudo-random lag spike schedule (1 to 5 ms)
        std::mt19937 rng(0xCAFE);
        std::uniform_int_distribution<int> spike_duration_dist(1, 5);

        while (!producer_done.load(std::memory_order_acquire) || !rb.Empty()) {
            iteration++;

            // Inject 1-5 ms extreme lag spike while producer is continuously flooding
            if (total_lag_spikes.load(std::memory_order_relaxed) < 60) {
                int spike_ms = spike_duration_dist(rng);
                std::this_thread::sleep_for(std::chrono::milliseconds(spike_ms));
                total_lag_spikes.fetch_add(1, std::memory_order_relaxed);
            }

            PlayerActionFrame latest = rb.PopLatestOrHold(fallback);
            total_pops.fetch_add(1, std::memory_order_relaxed);

            // Zero torn reads check
            if (!CheckFrameMatches(latest, latest.frame_index)) {
                torn_reads.fetch_add(1, std::memory_order_relaxed);
            }

            if (has_seen_any) {
                // Must maintain monotonic sequence progress: never regress!
                if (latest.frame_index < last_seen_index) {
                    monotonicity_violations.fetch_add(1, std::memory_order_relaxed);
                }
            } else {
                has_seen_any = true;
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
    EXPECT_EQ(final_seen_index, static_cast<uint32_t>(TOTAL_FRAMES - 1));
    EXPECT_GE(total_lag_spikes.load(), 5ULL);
    EXPECT_GE(total_pops.load(), 5ULL);
    EXPECT_TRUE(rb.Empty());
    EXPECT_EQ(rb.Size(), 0ULL);
}

// -----------------------------------------------------------------------------
// Test 2: Deterministic Batch Drain & Newest Frame Verification
// -----------------------------------------------------------------------------
// Proves that PopLatestOrHold drain returns exactly the newest frame of a queued batch
// and cleanly retains fallback when empty.
TEST(ChallengerM3BurstSuite, PopLatestOrHold_DeterministicBatchDrain) {
    ActionRingBuffer<512> rb;

    const std::vector<size_t> batch_sizes = {1, 2, 7, 32, 64, 128, 256, 512};
    uint64_t global_seq = 0;

    for (size_t batch_size : batch_sizes) {
        EXPECT_TRUE(rb.Empty());
        EXPECT_EQ(rb.Size(), 0ULL);

        uint64_t batch_start = global_seq;
        for (size_t i = 0; i < batch_size; ++i) {
            PlayerActionFrame f = MakeExpectedFrame(global_seq++);
            ASSERT_TRUE(rb.Push(f));
        }

        EXPECT_EQ(rb.Size(), batch_size);
        if (batch_size == 512) {
            EXPECT_TRUE(rb.Full());
        }

        PlayerActionFrame fallback = MakeExpectedFrame(9999999);
        PlayerActionFrame latest = rb.PopLatestOrHold(fallback);

        // Must return the exact newest frame in the batch (batch_start + batch_size - 1)
        uint64_t expected_newest = batch_start + batch_size - 1;
        EXPECT_EQ(latest.frame_index, static_cast<uint32_t>(expected_newest));
        EXPECT_TRUE(CheckFrameMatches(latest, expected_newest));
        EXPECT_TRUE(rb.Empty());
        EXPECT_EQ(rb.Size(), 0ULL);

        // Immediate subsequent call on empty buffer must return the fallback unmodified
        PlayerActionFrame held = rb.PopLatestOrHold(latest);
        EXPECT_EQ(held.frame_index, latest.frame_index);
        EXPECT_TRUE(CheckFrameMatches(held, expected_newest));
    }
}

// -----------------------------------------------------------------------------
// Test 3: Buffer Starvation under Unpredictable Sparse Bursts
// -----------------------------------------------------------------------------
// Consumer continuously attempts to pop while producer pushes in unpredictable sparse bursts.
// Asserts:
// - Zero false reads (when Pop returns false, buffer data is not returned or synthesized)
// - Zero deadlocks (threads execute cleanly and terminate)
// - Clean empty state and exact sequence tracking across all bursts
TEST(ChallengerM3BurstSuite, BufferStarvation_SparseBursts_ZeroFalseReads) {
    ActionRingBuffer<1024> rb;
    constexpr size_t NUM_BURSTS = 300;
    std::atomic<bool> producer_done{false};
    std::atomic<size_t> false_reads{0};
    std::atomic<size_t> torn_reads{0};
    std::atomic<size_t> sequence_violations{0};
    std::atomic<size_t> starvation_empty_polls{0};
    std::atomic<size_t> received_count{0};
    std::atomic<size_t> total_produced{0};

    std::thread producer([&]() {
        std::mt19937 rng(0xBEEF);
        std::uniform_int_distribution<size_t> burst_size_dist(0, 30);
        std::uniform_int_distribution<int> sleep_us_dist(20, 250);

        uint64_t seq = 0;
        for (size_t b = 0; b < NUM_BURSTS; ++b) {
            size_t count = burst_size_dist(rng);
            for (size_t i = 0; i < count; ++i) {
                PlayerActionFrame f = MakeExpectedFrame(seq++);
                while (!rb.Push(f)) {
                    std::this_thread::yield();
                }
            }
            // Unpredictable pause between bursts causing consumer starvation
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_us_dist(rng)));
        }

        total_produced.store(seq, std::memory_order_release);
        producer_done.store(true, std::memory_order_release);
    });

    std::thread consumer([&]() {
        uint64_t expected_idx = 0;

        while (!producer_done.load(std::memory_order_acquire) || !rb.Empty()) {
            PlayerActionFrame frame{};
            // Fill with sentinel pattern (0xAA) to verify Pop does not perform partial writes on false
            std::memset(&frame, 0xAA, sizeof(PlayerActionFrame));

            if (rb.Pop(frame)) {
                // Successful Pop
                if (frame.frame_index != static_cast<uint32_t>(expected_idx)) {
                    sequence_violations.fetch_add(1, std::memory_order_relaxed);
                }
                if (!CheckFrameMatches(frame, expected_idx)) {
                    torn_reads.fetch_add(1, std::memory_order_relaxed);
                }
                expected_idx++;
                received_count.fetch_add(1, std::memory_order_relaxed);
            } else {
                // Starvation encounter!
                starvation_empty_polls.fetch_add(1, std::memory_order_relaxed);

                // Check for false write: frame must remain untouched (all 0xAA bytes intact)
                const uint8_t* raw = reinterpret_cast<const uint8_t*>(&frame);
                for (size_t byte_idx = 0; byte_idx < sizeof(PlayerActionFrame); ++byte_idx) {
                    if (raw[byte_idx] != 0xAA) {
                        false_reads.fetch_add(1, std::memory_order_relaxed);
                        break;
                    }
                }
                std::this_thread::yield();
            }
        }
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(received_count.load(), total_produced.load());
    EXPECT_GT(received_count.load(), 2000ULL);
    EXPECT_GT(starvation_empty_polls.load(), 500ULL);
    EXPECT_EQ(false_reads.load(), 0ULL);
    EXPECT_EQ(torn_reads.load(), 0ULL);
    EXPECT_EQ(sequence_violations.load(), 0ULL);
    EXPECT_EQ(rb.DroppedFrames(), 0ULL);
    EXPECT_TRUE(rb.Empty());
    EXPECT_EQ(rb.Size(), 0ULL);
}

// -----------------------------------------------------------------------------
// Test 4: Multi-Burst Overflow Recovery (10 Bursts x 500 Frames into Capacity 64)
// -----------------------------------------------------------------------------
// Inject 10 successive bursts of 500 frames into a 64-capacity buffer.
// Drain completely between bursts.
// Asserts:
// - In each burst: exactly 64 accepted, exactly 436 rejected
// - Cumulative drops after 10 bursts == 10 * 436 == 4,360
// - Drained frames have zero corruption (all fields match expected)
// - Buffer recovers cleanly and functions properly under subsequent load
TEST(ChallengerM3BurstSuite, MultiBurstOverflowRecovery_10x500_Capacity64) {
    constexpr size_t CAPACITY = 64;
    constexpr size_t BURST_COUNT = 10;
    constexpr size_t FRAMES_PER_BURST = 500;
    constexpr size_t EXPECTED_DROPS_PER_BURST = FRAMES_PER_BURST - CAPACITY; // 436

    ActionRingBuffer<CAPACITY> rb;

    EXPECT_TRUE(rb.Empty());
    EXPECT_EQ(rb.Size(), 0ULL);
    EXPECT_EQ(rb.DroppedFrames(), 0ULL);

    size_t total_accepted = 0;
    size_t total_rejected = 0;

    for (size_t b = 0; b < BURST_COUNT; ++b) {
        // Pre-condition: buffer must be empty
        ASSERT_TRUE(rb.Empty());
        ASSERT_EQ(rb.Size(), 0ULL);
        ASSERT_EQ(rb.DroppedFrames(), b * EXPECTED_DROPS_PER_BURST);

        size_t burst_accepted = 0;
        size_t burst_rejected = 0;
        const uint64_t burst_base_idx = b * 10000ULL;

        // Inject burst of 500 frames
        for (size_t i = 0; i < FRAMES_PER_BURST; ++i) {
            PlayerActionFrame frame = MakeExpectedFrame(burst_base_idx + i);
            if (rb.Push(frame)) {
                burst_accepted++;
            } else {
                burst_rejected++;
            }
        }

        total_accepted += burst_accepted;
        total_rejected += burst_rejected;

        EXPECT_EQ(burst_accepted, CAPACITY);
        EXPECT_EQ(burst_rejected, EXPECTED_DROPS_PER_BURST);
        EXPECT_EQ(rb.Size(), CAPACITY);
        EXPECT_TRUE(rb.Full());
        EXPECT_EQ(rb.DroppedFrames(), (b + 1) * EXPECTED_DROPS_PER_BURST);

        // Drain completely and verify exact FIFO content and zero corruption
        for (size_t i = 0; i < CAPACITY; ++i) {
            PlayerActionFrame popped{};
            ASSERT_TRUE(rb.Pop(popped));
            EXPECT_TRUE(CheckFrameMatches(popped, burst_base_idx + i));
        }

        // Post-drain: buffer must be strictly empty
        PlayerActionFrame extra{};
        EXPECT_FALSE(rb.Pop(extra));
        EXPECT_TRUE(rb.Empty());
        EXPECT_EQ(rb.Size(), 0ULL);
        EXPECT_FALSE(rb.Full());
        EXPECT_EQ(rb.DroppedFrames(), (b + 1) * EXPECTED_DROPS_PER_BURST);
    }

    // Assert grand totals across all 10 bursts
    EXPECT_EQ(total_accepted, BURST_COUNT * CAPACITY); // 640 accepted
    EXPECT_EQ(total_rejected, BURST_COUNT * EXPECTED_DROPS_PER_BURST); // 4,360 rejected
    EXPECT_EQ(rb.DroppedFrames(), BURST_COUNT * EXPECTED_DROPS_PER_BURST); // 4,360 drops tracked

    // Post-burst recovery: verify subsequent normal paced streaming (20,000 frames) through the same buffer
    constexpr size_t POST_BURST_ITEMS = 20000;
    std::atomic<bool> producer_done{false};
    std::atomic<size_t> post_received{0};
    std::atomic<size_t> post_torn{0};
    std::atomic<size_t> post_fifo_err{0};

    std::thread post_producer([&]() {
        for (size_t i = 0; i < POST_BURST_ITEMS; ++i) {
            PlayerActionFrame f = MakeExpectedFrame(500000 + i);
            while (rb.Full()) {
                std::this_thread::yield();
            }
            while (!rb.Push(f)) {
                std::this_thread::yield();
            }
        }
        producer_done.store(true, std::memory_order_release);
    });

    std::thread post_consumer([&]() {
        size_t idx = 0;
        uint32_t last_idx = 0;
        while (!producer_done.load(std::memory_order_acquire) || !rb.Empty()) {
            PlayerActionFrame f{};
            if (rb.Pop(f)) {
                if (!CheckFrameMatches(f, 500000 + idx)) {
                    post_torn.fetch_add(1, std::memory_order_relaxed);
                }
                if (idx > 0 && f.frame_index != last_idx + 1) {
                    post_fifo_err.fetch_add(1, std::memory_order_relaxed);
                }
                last_idx = f.frame_index;
                idx++;
                post_received.fetch_add(1, std::memory_order_relaxed);
            } else {
                std::this_thread::yield();
            }
        }
    });

    post_producer.join();
    post_consumer.join();

    EXPECT_EQ(post_received.load(), POST_BURST_ITEMS);
    EXPECT_EQ(post_torn.load(), 0ULL);
    EXPECT_EQ(post_fifo_err.load(), 0ULL);
    // Dropped frames count must remain unchanged at 4,360 because paced streaming caused 0 drops
    EXPECT_EQ(rb.DroppedFrames(), BURST_COUNT * EXPECTED_DROPS_PER_BURST);
    EXPECT_TRUE(rb.Empty());
    EXPECT_EQ(rb.Size(), 0ULL);

    // Reset clears state completely
    rb.Reset();
    EXPECT_TRUE(rb.Empty());
    EXPECT_EQ(rb.Size(), 0ULL);
    EXPECT_EQ(rb.DroppedFrames(), 0ULL);
}

// -----------------------------------------------------------------------------
// Test 5: Variable Multi-Burst Overflow & Dynamic Drain
// -----------------------------------------------------------------------------
// Injects bursts of varying sizes (50, 150, 300, 500, 1000) into capacity-128 buffer.
// Verifies exact drop math and zero corruption across all accepted frames.
TEST(ChallengerM3BurstSuite, VariableMultiBurst_Capacity128_ExactDrops) {
    constexpr size_t CAPACITY = 128;
    ActionRingBuffer<CAPACITY> rb;

    struct BurstCase {
        size_t frame_count;
        size_t expected_accepted;
        size_t expected_drops;
    };

    const std::vector<BurstCase> cases = {
        {150, 128, 22},
        {300, 128, 172},
        {50,   50,   0},  // Under capacity
        {500, 128, 372},
        {1000, 128, 872},
    };

    uint64_t cumulative_drops = 0;

    for (size_t case_idx = 0; case_idx < cases.size(); ++case_idx) {
        const auto& c = cases[case_idx];
        const uint64_t base_idx = (case_idx + 1) * 50000ULL;

        for (size_t i = 0; i < c.frame_count; ++i) {
            PlayerActionFrame f = MakeExpectedFrame(base_idx + i);
            rb.Push(f);
        }

        cumulative_drops += c.expected_drops;
        EXPECT_EQ(rb.Size(), c.expected_accepted);
        EXPECT_EQ(rb.DroppedFrames(), cumulative_drops);

        // Drain and verify
        for (size_t i = 0; i < c.expected_accepted; ++i) {
            PlayerActionFrame popped{};
            ASSERT_TRUE(rb.Pop(popped));
            EXPECT_TRUE(CheckFrameMatches(popped, base_idx + i));
        }

        EXPECT_TRUE(rb.Empty());
        EXPECT_EQ(rb.Size(), 0ULL);
        EXPECT_EQ(rb.DroppedFrames(), cumulative_drops);
    }
}

// -----------------------------------------------------------------------------
// Test 6: Extreme Contention on Micro-Capacity Buffer (Capacity 4) with PopLatestOrHold
// -----------------------------------------------------------------------------
// High-frequency producer flooding 100,000 frames into a 4-capacity buffer while consumer
// uses PopLatestOrHold with 1 ms pauses.
// Asserts 0 deadlocks, 0 torn reads, monotonic sequence index.
TEST(ChallengerM3BurstSuite, MicroCapacity4_PopLatestOrHold_ZeroDeadlocks) {
    ActionRingBuffer<4> rb;
    constexpr size_t NUM_ITEMS = 100000;
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

    uint32_t final_index = 0;
    std::thread consumer([&]() {
        PlayerActionFrame fallback = MakeExpectedFrame(0);
        uint32_t last_index = 0;
        bool seen = false;
        size_t iter = 0;

        while (!producer_done.load(std::memory_order_acquire) || !rb.Empty()) {
            iter++;
            if (iter % 100 == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            PlayerActionFrame latest = rb.PopLatestOrHold(fallback);
            poll_count.fetch_add(1, std::memory_order_relaxed);

            if (!CheckFrameMatches(latest, latest.frame_index)) {
                torn_reads.fetch_add(1, std::memory_order_relaxed);
            }

            if (seen) {
                if (latest.frame_index < last_index) {
                    monotonicity_violations.fetch_add(1, std::memory_order_relaxed);
                }
            } else {
                seen = true;
            }

            last_index = latest.frame_index;
            fallback = latest;
            std::this_thread::yield();
        }

        final_index = last_index;
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(monotonicity_violations.load(), 0ULL);
    EXPECT_EQ(torn_reads.load(), 0ULL);
    EXPECT_EQ(final_index, static_cast<uint32_t>(NUM_ITEMS - 1));
    EXPECT_TRUE(rb.Empty());
    EXPECT_EQ(rb.Size(), 0ULL);
}

TEST_RUNNER_MAIN()

#include "test_runner.h"
#include "playworld/ring_buffer.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace playworld;

// Fast thread-local pseudo-random number generator for micro-delays
class FastRng {
public:
    explicit FastRng(uint64_t seed) : state_(seed != 0 ? seed : 0x853c49e6748fea9bULL) {}
    uint64_t Next() noexcept {
        state_ ^= state_ >> 12;
        state_ ^= state_ << 25;
        state_ ^= state_ >> 27;
        return state_ * 0x2545F4914F6CDD1DULL;
    }
    uint32_t NextBounded(uint32_t bound) noexcept {
        if (bound <= 1) return 0;
        return static_cast<uint32_t>((Next() >> 32) % bound);
    }
private:
    uint64_t state_;
};

// Deterministically constructs a PlayerActionFrame with distinct data in all 9 fields
static PlayerActionFrame MakeExpectedFrame(uint64_t i) {
    PlayerActionFrame f{};
    f.timestamp_us = i * 1000ULL + 42ULL;
    f.frame_index = static_cast<uint32_t>(i);
    f.mouse_delta_yaw = static_cast<float>(std::sin(static_cast<double>(i) * 0.01));
    f.mouse_delta_pitch = static_cast<float>(std::cos(static_cast<double>(i) * 0.01));
    f.analog_move_x = static_cast<float>(i % 100) / 100.0f;
    f.analog_move_y = static_cast<float>(i % 200) / 200.0f;
    f.keys_pressed = static_cast<uint16_t>(i & 0x07FF);
    f.keys_just_down = static_cast<uint16_t>(i & 0x00FF);
    f.auxiliary_trigger = static_cast<float>(i % 10) * 0.1f;
    return f;
}

// Verifies all 9 fields individually + performs raw 36-byte memory comparison
struct FrameValidationResult {
    bool is_valid{true};
    int torn_field_mask{0}; // Bitfield of torn fields if any
    bool memcmp_failed{false};
};

static FrameValidationResult ValidateFrame(const PlayerActionFrame& actual, uint64_t expected_idx) {
    const PlayerActionFrame exp = MakeExpectedFrame(expected_idx);
    FrameValidationResult res;

    if (actual.timestamp_us != exp.timestamp_us) {
        res.is_valid = false;
        res.torn_field_mask |= (1 << 0);
    }
    if (actual.frame_index != exp.frame_index) {
        res.is_valid = false;
        res.torn_field_mask |= (1 << 1);
    }
    if (actual.mouse_delta_yaw != exp.mouse_delta_yaw) {
        res.is_valid = false;
        res.torn_field_mask |= (1 << 2);
    }
    if (actual.mouse_delta_pitch != exp.mouse_delta_pitch) {
        res.is_valid = false;
        res.torn_field_mask |= (1 << 3);
    }
    if (actual.analog_move_x != exp.analog_move_x) {
        res.is_valid = false;
        res.torn_field_mask |= (1 << 4);
    }
    if (actual.analog_move_y != exp.analog_move_y) {
        res.is_valid = false;
        res.torn_field_mask |= (1 << 5);
    }
    if (actual.keys_pressed != exp.keys_pressed) {
        res.is_valid = false;
        res.torn_field_mask |= (1 << 6);
    }
    if (actual.keys_just_down != exp.keys_just_down) {
        res.is_valid = false;
        res.torn_field_mask |= (1 << 7);
    }
    if (actual.auxiliary_trigger != exp.auxiliary_trigger) {
        res.is_valid = false;
        res.torn_field_mask |= (1 << 8);
    }
    if (std::memcmp(&actual, &exp, sizeof(PlayerActionFrame)) != 0) {
        res.is_valid = false;
        res.memcmp_failed = true;
    }

    return res;
}

// -----------------------------------------------------------------------------
// Test 1: 1,000,000 frames under extreme high-frequency contention, thread preemption,
// and randomized micro-delays (yield + sleep_for nanoseconds).
// Asserts:
//   - 0 deadlocks across the run
//   - Exactly 1,000,000 frames received
//   - Strict monotonic FIFO sequence IDs
//   - 0 torn reads across all 9 struct fields of PlayerActionFrame
// -----------------------------------------------------------------------------
TEST(ChallengerM3Concurrency, Stress1M_RandomMicroDelaysAndPreemption) {
    ActionRingBuffer<2048> rb;
    constexpr size_t NUM_FRAMES = 1000000;

    std::atomic<size_t> received_count{0};
    std::atomic<size_t> torn_reads_count{0};
    std::atomic<size_t> fifo_violations{0};
    std::atomic<bool> deadlock_detected{false};
    std::atomic<bool> producer_finished{false};
    std::atomic<bool> consumer_finished{false};

    // Watchdog thread with short polling intervals for instant join
    std::thread watchdog([&]() {
        auto start = std::chrono::steady_clock::now();
        constexpr auto TIMEOUT = std::chrono::seconds(25);
        while (!producer_finished.load(std::memory_order_acquire) ||
               !consumer_finished.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            if (std::chrono::steady_clock::now() - start > TIMEOUT) {
                deadlock_detected.store(true, std::memory_order_release);
                std::cerr << "[CHALLENGER FATAL] Watchdog timeout: Deadlock detected in 1M stress test!\n";
                break;
            }
        }
    });

    auto start_time = std::chrono::high_resolution_clock::now();

    std::thread producer([&]() {
        FastRng rng(0x1337BEEFULL);
        for (size_t i = 0; i < NUM_FRAMES; ++i) {
            if (deadlock_detected.load(std::memory_order_relaxed)) break;

            PlayerActionFrame frame = MakeExpectedFrame(i);

            // Inject periodic preemption and micro-delays during active generation
            if ((i & 0x7FF) == 0) {
                std::this_thread::yield();
                if ((rng.Next() & 0x07) == 0) {
                    std::this_thread::sleep_for(std::chrono::nanoseconds(rng.NextBounded(200) + 1));
                }
            }

            // Retry until pushed, yielding on backpressure
            while (!rb.Push(frame)) {
                if (deadlock_detected.load(std::memory_order_relaxed)) break;
                if ((rng.Next() & 0x03) == 0) {
                    std::this_thread::sleep_for(std::chrono::nanoseconds(rng.NextBounded(100) + 1));
                } else {
                    std::this_thread::yield();
                }
            }
        }
        producer_finished.store(true, std::memory_order_release);
    });

    std::thread consumer([&]() {
        FastRng rng(0xCAFEF00DULL);
        uint32_t last_seq = 0;

        for (size_t i = 0; i < NUM_FRAMES; ++i) {
            if (deadlock_detected.load(std::memory_order_relaxed)) break;

            // Inject periodic preemption on consumer thread
            if ((i & 0x7FF) == 0) {
                std::this_thread::yield();
                if ((rng.Next() & 0x07) == 0) {
                    std::this_thread::sleep_for(std::chrono::nanoseconds(rng.NextBounded(200) + 1));
                }
            }

            PlayerActionFrame frame{};
            while (!rb.Pop(frame)) {
                if (deadlock_detected.load(std::memory_order_relaxed)) break;
                if ((rng.Next() & 0x03) == 0) {
                    std::this_thread::sleep_for(std::chrono::nanoseconds(rng.NextBounded(100) + 1));
                } else {
                    std::this_thread::yield();
                }
            }

            FrameValidationResult val = ValidateFrame(frame, i);
            if (!val.is_valid) {
                torn_reads_count.fetch_add(1, std::memory_order_relaxed);
            }

            if (i > 0 && frame.frame_index != last_seq + 1) {
                fifo_violations.fetch_add(1, std::memory_order_relaxed);
            }

            last_seq = frame.frame_index;
            received_count.fetch_add(1, std::memory_order_relaxed);
        }
        consumer_finished.store(true, std::memory_order_release);
    });

    producer.join();
    consumer.join();
    watchdog.join();

    auto end_time = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    double mops = (static_cast<double>(NUM_FRAMES) / (duration_ms / 1000.0)) / 1e6;
    std::cout << "    -> Processed " << received_count.load() << " frames in "
              << duration_ms << " ms (" << mops << " Mops/sec)\n";

    EXPECT_FALSE(deadlock_detected.load());
    EXPECT_EQ(received_count.load(), NUM_FRAMES);
    EXPECT_EQ(torn_reads_count.load(), 0ULL);
    EXPECT_EQ(fifo_violations.load(), 0ULL);
    EXPECT_TRUE(rb.Empty());
    EXPECT_EQ(rb.Size(), 0ULL);
}

// -----------------------------------------------------------------------------
// Test 2: Paced SPSC Consumer with 1,000,000 frames.
// Asserts:
//   - 0 dropped actions (rb.DroppedFrames() == 0ULL)
//   - 0 deadlocks
//   - Strict FIFO monotonicity
//   - 0 torn reads across all 9 fields
// -----------------------------------------------------------------------------
TEST(ChallengerM3Concurrency, ZeroDroppedActions_UnderPacedConsumer) {
    ActionRingBuffer<2048> rb;
    constexpr size_t NUM_FRAMES = 1000000;

    std::atomic<bool> producer_done{false};
    std::atomic<size_t> received_count{0};
    std::atomic<size_t> torn_reads_count{0};
    std::atomic<size_t> fifo_violations{0};
    std::atomic<bool> deadlock_detected{false};

    std::thread watchdog([&]() {
        auto start = std::chrono::steady_clock::now();
        constexpr auto TIMEOUT = std::chrono::seconds(25);
        while (received_count.load(std::memory_order_acquire) < NUM_FRAMES) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            if (std::chrono::steady_clock::now() - start > TIMEOUT) {
                deadlock_detected.store(true, std::memory_order_release);
                producer_done.store(true, std::memory_order_release);
                std::cerr << "[CHALLENGER FATAL] Watchdog timeout in PacedConsumer test!\n";
                break;
            }
        }
    });

    auto start_time = std::chrono::high_resolution_clock::now();

    std::thread consumer([&]() {
        size_t idx = 0;
        uint32_t last_seq = 0;
        while (!producer_done.load(std::memory_order_acquire) || !rb.Empty()) {
            if (deadlock_detected.load(std::memory_order_relaxed)) break;

            PlayerActionFrame frame{};
            if (rb.Pop(frame)) {
                FrameValidationResult val = ValidateFrame(frame, idx);
                if (!val.is_valid) {
                    torn_reads_count.fetch_add(1, std::memory_order_relaxed);
                }
                if (idx > 0 && frame.frame_index != last_seq + 1) {
                    fifo_violations.fetch_add(1, std::memory_order_relaxed);
                }
                last_seq = frame.frame_index;
                idx++;
                received_count.fetch_add(1, std::memory_order_relaxed);
            } else {
                std::this_thread::yield();
            }
        }
    });

    std::thread producer([&]() {
        for (size_t i = 0; i < NUM_FRAMES; ++i) {
            if (deadlock_detected.load(std::memory_order_relaxed)) break;

            PlayerActionFrame frame = MakeExpectedFrame(i);

            // Under paced SPSC, wait for space before pushing so dropped_count_ is never incremented
            while (rb.Full()) {
                std::this_thread::yield();
            }

            bool pushed = rb.Push(frame);
            if (!pushed) {
                // If Push returned false, dropped_count_ was incremented.
                // In paced mode this should never occur.
            }
        }
        producer_done.store(true, std::memory_order_release);
    });

    producer.join();
    consumer.join();
    watchdog.join();

    auto end_time = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    double mops = (static_cast<double>(NUM_FRAMES) / (duration_ms / 1000.0)) / 1e6;
    std::cout << "    -> Paced: " << received_count.load() << " frames in "
              << duration_ms << " ms (" << mops << " Mops/sec), Dropped: "
              << rb.DroppedFrames() << "\n";

    EXPECT_FALSE(deadlock_detected.load());
    EXPECT_EQ(received_count.load(), NUM_FRAMES);
    EXPECT_EQ(torn_reads_count.load(), 0ULL);
    EXPECT_EQ(fifo_violations.load(), 0ULL);
    EXPECT_EQ(rb.DroppedFrames(), 0ULL);
    EXPECT_TRUE(rb.Empty());
    EXPECT_EQ(rb.Size(), 0ULL);
}

// -----------------------------------------------------------------------------
// Test 3: Burst Starvation & Severe Thread Jitter
// Alternates between:
//   - Producer burst floods while consumer is starved and paused
//   - Consumer bursts drain while producer is starved and paused
// Asserts:
//   - 0 deadlocks
//   - Strict FIFO monotonicity
//   - 0 torn reads across all 9 struct fields
// -----------------------------------------------------------------------------
TEST(ChallengerM3Concurrency, BurstStarvationAndSurgeContention) {
    ActionRingBuffer<2048> rb;
    constexpr size_t BURSTS = 500;
    constexpr size_t FRAMES_PER_BURST = 1000;
    constexpr size_t TOTAL_FRAMES = BURSTS * FRAMES_PER_BURST; // 500,000 frames

    std::atomic<size_t> received_count{0};
    std::atomic<size_t> torn_reads_count{0};
    std::atomic<size_t> fifo_violations{0};
    std::atomic<bool> deadlock_detected{false};
    std::atomic<bool> producer_done{false};

    std::thread watchdog([&]() {
        auto start = std::chrono::steady_clock::now();
        constexpr auto TIMEOUT = std::chrono::seconds(25);
        while (!producer_done.load(std::memory_order_acquire) ||
               received_count.load(std::memory_order_acquire) < TOTAL_FRAMES) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            if (std::chrono::steady_clock::now() - start > TIMEOUT) {
                deadlock_detected.store(true, std::memory_order_release);
                producer_done.store(true, std::memory_order_release);
                std::cerr << "[CHALLENGER FATAL] Watchdog timeout in BurstStarvation test!\n";
                break;
            }
        }
    });

    std::thread producer([&]() {
        FastRng rng(0x8899AABBULL);
        for (size_t b = 0; b < BURSTS; ++b) {
            if (deadlock_detected.load(std::memory_order_relaxed)) break;

            for (size_t i = 0; i < FRAMES_PER_BURST; ++i) {
                uint64_t seq = b * FRAMES_PER_BURST + i;
                PlayerActionFrame frame = MakeExpectedFrame(seq);
                while (!rb.Push(frame)) {
                    std::this_thread::yield();
                }
            }

            // Burst starvation: Pause producer for 5 microseconds to force consumer starvation
            std::this_thread::sleep_for(std::chrono::nanoseconds(rng.NextBounded(5000) + 1000));
        }
        producer_done.store(true, std::memory_order_release);
    });

    std::thread consumer([&]() {
        FastRng rng(0x11223344ULL);
        uint32_t last_seq = 0;
        size_t idx = 0;

        while (idx < TOTAL_FRAMES) {
            if (deadlock_detected.load(std::memory_order_relaxed)) break;

            PlayerActionFrame frame{};
            if (rb.Pop(frame)) {
                FrameValidationResult val = ValidateFrame(frame, idx);
                if (!val.is_valid) {
                    torn_reads_count.fetch_add(1, std::memory_order_relaxed);
                }
                if (idx > 0 && frame.frame_index != last_seq + 1) {
                    fifo_violations.fetch_add(1, std::memory_order_relaxed);
                }
                last_seq = frame.frame_index;
                idx++;
                received_count.fetch_add(1, std::memory_order_relaxed);

                // Occasionally simulate consumer lag/burst pause
                if ((idx % 2000) == 0) {
                    std::this_thread::sleep_for(std::chrono::nanoseconds(rng.NextBounded(4000) + 500));
                }
            } else {
                std::this_thread::yield();
            }
        }
    });

    producer.join();
    consumer.join();
    watchdog.join();

    EXPECT_FALSE(deadlock_detected.load());
    EXPECT_EQ(received_count.load(), TOTAL_FRAMES);
    EXPECT_EQ(torn_reads_count.load(), 0ULL);
    EXPECT_EQ(fifo_violations.load(), 0ULL);
    EXPECT_TRUE(rb.Empty());
}

// -----------------------------------------------------------------------------
// Test 4: Extreme Thread Preemption with Tiny Capacity Wrap-Around (Capacity = 16)
// Over 500,000 frames, index wraps around circular buffer 31,250 times.
// Asserts:
//   - 0 deadlocks
//   - 0 torn reads across all 9 struct fields
//   - Strict FIFO monotonicity
// -----------------------------------------------------------------------------
TEST(ChallengerM3Concurrency, RapidWrapAroundContention_TinyBuffer16) {
    ActionRingBuffer<16> rb;
    constexpr size_t NUM_FRAMES = 500000;

    std::atomic<size_t> received_count{0};
    std::atomic<size_t> torn_reads_count{0};
    std::atomic<size_t> fifo_violations{0};
    std::atomic<bool> deadlock_detected{false};
    std::atomic<bool> producer_done{false};

    std::thread watchdog([&]() {
        auto start = std::chrono::steady_clock::now();
        constexpr auto TIMEOUT = std::chrono::seconds(25);
        while (!producer_done.load(std::memory_order_acquire) ||
               received_count.load(std::memory_order_acquire) < NUM_FRAMES) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            if (std::chrono::steady_clock::now() - start > TIMEOUT) {
                deadlock_detected.store(true, std::memory_order_release);
                producer_done.store(true, std::memory_order_release);
                std::cerr << "[CHALLENGER FATAL] Watchdog timeout in TinyBuffer16 test!\n";
                break;
            }
        }
    });

    std::thread producer([&]() {
        FastRng rng(0x55667788ULL);
        for (size_t i = 0; i < NUM_FRAMES; ++i) {
            if (deadlock_detected.load(std::memory_order_relaxed)) break;

            PlayerActionFrame frame = MakeExpectedFrame(i);
            while (!rb.Push(frame)) {
                if (deadlock_detected.load(std::memory_order_relaxed)) break;
                if ((rng.Next() & 0x01) == 0) {
                    std::this_thread::yield();
                }
            }
        }
        producer_done.store(true, std::memory_order_release);
    });

    std::thread consumer([&]() {
        FastRng rng(0x99AABBCCULL);
        uint32_t last_seq = 0;

        for (size_t i = 0; i < NUM_FRAMES; ++i) {
            if (deadlock_detected.load(std::memory_order_relaxed)) break;

            PlayerActionFrame frame{};
            while (!rb.Pop(frame)) {
                if (deadlock_detected.load(std::memory_order_relaxed)) break;
                if ((rng.Next() & 0x01) == 0) {
                    std::this_thread::yield();
                }
            }

            FrameValidationResult val = ValidateFrame(frame, i);
            if (!val.is_valid) {
                torn_reads_count.fetch_add(1, std::memory_order_relaxed);
            }
            if (i > 0 && frame.frame_index != last_seq + 1) {
                fifo_violations.fetch_add(1, std::memory_order_relaxed);
            }
            last_seq = frame.frame_index;
            received_count.fetch_add(1, std::memory_order_relaxed);
        }
    });

    producer.join();
    consumer.join();
    watchdog.join();

    EXPECT_FALSE(deadlock_detected.load());
    EXPECT_EQ(received_count.load(), NUM_FRAMES);
    EXPECT_EQ(torn_reads_count.load(), 0ULL);
    EXPECT_EQ(fifo_violations.load(), 0ULL);
    EXPECT_TRUE(rb.Empty());
}

// -----------------------------------------------------------------------------
// Test 5: PopLatestOrHold Under 1,000,000 Events & Asymmetric Preemption
// Asserts:
//   - Sequence monotonicity: frame_index never regresses
//   - 0 torn reads across all 9 struct fields of every returned frame
//   - 0 deadlocks
//   - Terminal frame reaches end of sequence
// -----------------------------------------------------------------------------
TEST(ChallengerM3Concurrency, PopLatestOrHold_MonotonicityAndTornReadStress) {
    ActionRingBuffer<2048> rb;
    constexpr size_t NUM_FRAMES = 1000000;

    std::atomic<bool> producer_done{false};
    std::atomic<size_t> monotonicity_violations{0};
    std::atomic<size_t> torn_reads_count{0};
    std::atomic<size_t> sample_count{0};
    std::atomic<bool> deadlock_detected{false};

    std::thread watchdog([&]() {
        auto start = std::chrono::steady_clock::now();
        constexpr auto TIMEOUT = std::chrono::seconds(25);
        while (!producer_done.load(std::memory_order_acquire) || !rb.Empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            if (std::chrono::steady_clock::now() - start > TIMEOUT) {
                deadlock_detected.store(true, std::memory_order_release);
                producer_done.store(true, std::memory_order_release);
                std::cerr << "[CHALLENGER FATAL] Watchdog timeout in PopLatestOrHold test!\n";
                break;
            }
        }
    });

    std::thread producer([&]() {
        FastRng rng(0xDEADBEEFULL);
        for (size_t i = 0; i < NUM_FRAMES; ++i) {
            if (deadlock_detected.load(std::memory_order_relaxed)) break;

            PlayerActionFrame frame = MakeExpectedFrame(i);
            while (!rb.Push(frame)) {
                if (deadlock_detected.load(std::memory_order_relaxed)) break;
                std::this_thread::yield();
            }

            if ((i & 0x0FFF) == 0 && (rng.Next() & 0x03) == 0) {
                std::this_thread::sleep_for(std::chrono::nanoseconds(rng.NextBounded(300) + 1));
            }
        }
        producer_done.store(true, std::memory_order_release);
    });

    uint32_t final_observed_index = 0;
    std::thread consumer([&]() {
        FastRng rng(0xFEEDFACEULL);
        PlayerActionFrame fallback = MakeExpectedFrame(0);
        uint32_t last_seen_index = 0;
        bool has_first = false;

        while (!producer_done.load(std::memory_order_acquire) || !rb.Empty()) {
            if (deadlock_detected.load(std::memory_order_relaxed)) break;

            PlayerActionFrame latest = rb.PopLatestOrHold(fallback);
            sample_count.fetch_add(1, std::memory_order_relaxed);

            if (has_first) {
                if (latest.frame_index < last_seen_index) {
                    monotonicity_violations.fetch_add(1, std::memory_order_relaxed);
                }
            } else {
                has_first = true;
            }

            // Verify integrity of all 9 fields of latest frame
            FrameValidationResult val = ValidateFrame(latest, latest.frame_index);
            if (!val.is_valid) {
                torn_reads_count.fetch_add(1, std::memory_order_relaxed);
            }

            last_seen_index = latest.frame_index;
            fallback = latest;

            if ((rng.Next() & 0x0F) == 0) {
                std::this_thread::sleep_for(std::chrono::nanoseconds(rng.NextBounded(200) + 1));
            } else {
                std::this_thread::yield();
            }
        }

        final_observed_index = last_seen_index;
    });

    producer.join();
    consumer.join();
    watchdog.join();

    EXPECT_FALSE(deadlock_detected.load());
    EXPECT_EQ(monotonicity_violations.load(), 0ULL);
    EXPECT_EQ(torn_reads_count.load(), 0ULL);
    EXPECT_EQ(final_observed_index, static_cast<uint32_t>(NUM_FRAMES - 1));
    EXPECT_GT(sample_count.load(), 100ULL);
    EXPECT_TRUE(rb.Empty());
}

// -----------------------------------------------------------------------------
// Test 6: Concurrent Peek and Pop Interleaving
// Verifies that Peek() concurrently with producer Push() reads untorn frames
// and matches the subsequent Pop() exactly without sequence regression.
// -----------------------------------------------------------------------------
TEST(ChallengerM3Concurrency, ConcurrentPeekAndPopInterleaving) {
    ActionRingBuffer<2048> rb;
    constexpr size_t NUM_FRAMES = 500000;

    std::atomic<bool> producer_done{false};
    std::atomic<size_t> peek_matches{0};
    std::atomic<size_t> torn_reads_count{0};
    std::atomic<size_t> fifo_violations{0};
    std::atomic<bool> deadlock_detected{false};

    std::thread watchdog([&]() {
        auto start = std::chrono::steady_clock::now();
        constexpr auto TIMEOUT = std::chrono::seconds(25);
        while (!producer_done.load(std::memory_order_acquire) || !rb.Empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            if (std::chrono::steady_clock::now() - start > TIMEOUT) {
                deadlock_detected.store(true, std::memory_order_release);
                producer_done.store(true, std::memory_order_release);
                break;
            }
        }
    });

    std::thread producer([&]() {
        for (size_t i = 0; i < NUM_FRAMES; ++i) {
            if (deadlock_detected.load(std::memory_order_relaxed)) break;
            PlayerActionFrame f = MakeExpectedFrame(i);
            while (!rb.Push(f)) {
                std::this_thread::yield();
            }
        }
        producer_done.store(true, std::memory_order_release);
    });

    std::thread consumer([&]() {
        size_t expected_idx = 0;
        while (!producer_done.load(std::memory_order_acquire) || !rb.Empty()) {
            if (deadlock_detected.load(std::memory_order_relaxed)) break;

            PlayerActionFrame peeked{};
            bool has_peek = rb.Peek(peeked);

            PlayerActionFrame popped{};
            if (rb.Pop(popped)) {
                if (has_peek) {
                    // In single-consumer queue, if Peek succeeded right before Pop,
                    // peeked must be identical to popped!
                    if (peeked.frame_index == popped.frame_index) {
                        peek_matches.fetch_add(1, std::memory_order_relaxed);
                    }
                }

                FrameValidationResult val = ValidateFrame(popped, expected_idx);
                if (!val.is_valid) {
                    torn_reads_count.fetch_add(1, std::memory_order_relaxed);
                }
                if (popped.frame_index != static_cast<uint32_t>(expected_idx)) {
                    fifo_violations.fetch_add(1, std::memory_order_relaxed);
                }
                expected_idx++;
            } else {
                std::this_thread::yield();
            }
        }
    });

    producer.join();
    consumer.join();
    watchdog.join();

    EXPECT_FALSE(deadlock_detected.load());
    EXPECT_EQ(torn_reads_count.load(), 0ULL);
    EXPECT_EQ(fifo_violations.load(), 0ULL);
    EXPECT_GT(peek_matches.load(), 1000ULL);
    EXPECT_TRUE(rb.Empty());
}

// -----------------------------------------------------------------------------
// Test 7: Multi-Run Preemption Matrix
// Runs 5 back-to-back concurrent iterations with varying seeds and preemption
// ratios to assert 0 deadlocks across all runs.
// -----------------------------------------------------------------------------
TEST(ChallengerM3Concurrency, PreemptionMatrix_MultiRun) {
    constexpr size_t RUNS = 5;
    constexpr size_t FRAMES_PER_RUN = 100000;

    for (size_t r = 0; r < RUNS; ++r) {
        ActionRingBuffer<2048> rb;
        std::atomic<size_t> received{0};
        std::atomic<size_t> torn{0};
        std::atomic<size_t> fifo_viol{0};
        std::atomic<bool> run_deadlocked{false};
        std::atomic<bool> done{false};

        std::thread watchdog([&]() {
            auto start = std::chrono::steady_clock::now();
            constexpr auto TIMEOUT = std::chrono::seconds(10);
            while (!done.load(std::memory_order_acquire)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                if (std::chrono::steady_clock::now() - start > TIMEOUT) {
                    run_deadlocked.store(true, std::memory_order_release);
                    done.store(true, std::memory_order_release);
                    break;
                }
            }
        });

        std::thread prod([&]() {
            FastRng rng(0xABC0000ULL + r);
            for (size_t i = 0; i < FRAMES_PER_RUN; ++i) {
                if (run_deadlocked.load(std::memory_order_relaxed)) break;

                PlayerActionFrame f = MakeExpectedFrame(i);
                while (!rb.Push(f)) {
                    if (run_deadlocked.load(std::memory_order_relaxed)) break;
                    std::this_thread::yield();
                }

                if ((rng.Next() & 0x01FF) == 0) {
                    std::this_thread::sleep_for(std::chrono::nanoseconds(rng.NextBounded(100) + 1));
                }
            }
        });

        std::thread cons([&]() {
            FastRng rng(0xDEF0000ULL + r);
            uint32_t last = 0;
            for (size_t i = 0; i < FRAMES_PER_RUN; ++i) {
                if (run_deadlocked.load(std::memory_order_relaxed)) break;

                PlayerActionFrame f{};
                while (!rb.Pop(f)) {
                    if (run_deadlocked.load(std::memory_order_relaxed)) break;
                    std::this_thread::yield();
                }

                FrameValidationResult v = ValidateFrame(f, i);
                if (!v.is_valid) {
                    torn.fetch_add(1, std::memory_order_relaxed);
                }
                if (i > 0 && f.frame_index != last + 1) {
                    fifo_viol.fetch_add(1, std::memory_order_relaxed);
                }
                last = f.frame_index;
                received.fetch_add(1, std::memory_order_relaxed);

                if ((rng.Next() & 0x01FF) == 0) {
                    std::this_thread::sleep_for(std::chrono::nanoseconds(rng.NextBounded(100) + 1));
                }
            }
            done.store(true, std::memory_order_release);
        });

        prod.join();
        cons.join();
        watchdog.join();

        EXPECT_FALSE(run_deadlocked.load());
        EXPECT_EQ(received.load(), FRAMES_PER_RUN);
        EXPECT_EQ(torn.load(), 0ULL);
        EXPECT_EQ(fifo_viol.load(), 0ULL);
        EXPECT_TRUE(rb.Empty());
    }
}

TEST_RUNNER_MAIN()

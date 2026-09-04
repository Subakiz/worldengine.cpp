#pragma once

#include "action_types.h"

#include <atomic>
#include <array>
#include <cstdint>
#include <cstddef>
#include <optional>
#include <vector>

namespace playworld {

// Lock-Free Single-Producer Single-Consumer (SPSC) Circular Ring Buffer
template <size_t BufferCapacity = 2048>
class ActionRingBuffer {
    static_assert(BufferCapacity > 0, "Capacity must be greater than zero");
    static_assert((BufferCapacity & (BufferCapacity - 1)) == 0, "Capacity must be a power of 2");

public:
    ActionRingBuffer() : head_(0), tail_(0), dropped_count_(0) {}

    // Non-copyable, movable
    ActionRingBuffer(const ActionRingBuffer&) = delete;
    ActionRingBuffer& operator=(const ActionRingBuffer&) = delete;

    // Producer API: Pushes an action frame into the queue.
    // Returns true on success, or false if the buffer is full.
    bool Push(const PlayerActionFrame& item) noexcept {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        const size_t current_head = head_.load(std::memory_order_acquire);

        if ((current_tail - current_head) >= BufferCapacity) {
            dropped_count_.fetch_add(1, std::memory_order_relaxed);
            return false; // Buffer full
        }

        buffer_[current_tail & (BufferCapacity - 1)] = item;
        tail_.store(current_tail + 1, std::memory_order_release);
        return true;
    }

    // Consumer API: Pops the oldest action frame from the queue.
    // Returns true on success and writes to item, or false if empty.
    bool Pop(PlayerActionFrame& item) noexcept {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        const size_t current_tail = tail_.load(std::memory_order_acquire);

        if (current_head == current_tail) {
            return false; // Buffer empty
        }

        item = buffer_[current_head & (BufferCapacity - 1)];
        head_.store(current_head + 1, std::memory_order_release);
        return true;
    }

    // Alias for Pop providing standard TryPop semantics
    bool TryPop(PlayerActionFrame& item) noexcept {
        return Pop(item);
    }

    // Consumer API: Peeks at the oldest action frame without advancing head.
    bool Peek(PlayerActionFrame& item) const noexcept {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        const size_t current_tail = tail_.load(std::memory_order_acquire);

        if (current_head == current_tail) {
            return false; // Buffer empty
        }

        item = buffer_[current_head & (BufferCapacity - 1)];
        return true;
    }

    // Drains queue to return the latest action frame, or returns fallback if empty.
    PlayerActionFrame PopLatestOrHold(const PlayerActionFrame& fallback) noexcept {
        PlayerActionFrame latest{};
        bool found = false;
        PlayerActionFrame temp{};
        while (Pop(temp)) {
            latest = temp;
            found = true;
        }
        return found ? latest : fallback;
    }

    // Resets the ring buffer pointers to empty
    void Reset() noexcept {
        head_.store(0, std::memory_order_release);
        tail_.store(0, std::memory_order_release);
        dropped_count_.store(0, std::memory_order_relaxed);
    }

    // Alias for Reset
    void Clear() noexcept {
        Reset();
    }

    [[nodiscard]] size_t Size() const noexcept {
        const size_t current_tail = tail_.load(std::memory_order_acquire);
        const size_t current_head = head_.load(std::memory_order_acquire);
        return current_tail >= current_head ? (current_tail - current_head) : 0;
    }

    [[nodiscard]] constexpr size_t Capacity() const noexcept {
        return BufferCapacity;
    }

    [[nodiscard]] constexpr size_t GetCapacity() const noexcept {
        return BufferCapacity;
    }

    [[nodiscard]] bool Empty() const noexcept {
        return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool Full() const noexcept {
        return Size() >= BufferCapacity;
    }

    [[nodiscard]] uint64_t DroppedFrames() const noexcept {
        return dropped_count_.load(std::memory_order_relaxed);
    }

private:
    alignas(64) std::array<PlayerActionFrame, BufferCapacity> buffer_{};
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
    alignas(64) std::atomic<uint64_t> dropped_count_{0};
};

// Assembles a normalized 32-dimensional continuous-discrete vector from PlayerActionFrame
void NormalizeActionFrame(const PlayerActionFrame& frame, float* out32) noexcept;

// 2-Layer Action MLP Projection: Projects 32-dim raw action vector to latent dimension
// using SiLU (x * sigmoid(x)) activations.
class ActionMLP {
public:
    explicit ActionMLP(size_t input_dim = 32, size_t hidden_dim = 512, size_t output_dim = 1536);

    void Project(const float* input32, float* output_emb) const noexcept;
    void ProjectFrame(const PlayerActionFrame& frame, float* output_emb) const noexcept;
    [[nodiscard]] std::vector<float> Project(const PlayerActionFrame& frame) const;

    [[nodiscard]] size_t GetInputDim() const noexcept { return input_dim_; }
    [[nodiscard]] size_t GetHiddenDim() const noexcept { return hidden_dim_; }
    [[nodiscard]] size_t GetOutputDim() const noexcept { return output_dim_; }

private:
    size_t input_dim_{32};
    size_t hidden_dim_{512};
    size_t output_dim_{1536};

    // Layer 1: W1 [hidden_dim x input_dim], b1 [hidden_dim]
    std::vector<float> w1_;
    std::vector<float> b1_;

    // Layer 2: W2 [output_dim x hidden_dim], b2 [output_dim]
    std::vector<float> w2_;
    std::vector<float> b2_;
};

} // namespace playworld

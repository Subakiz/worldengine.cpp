#include "test_runner.h"
#include "playworld/voxel_grid.h"
#include "playworld/pwmf_format.h"
#include "playworld/ring_buffer.h"
#include "playworld/action_types.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace playworld;

// ============================================================================
// Intentional Fault Triggers (for ASan and UBSan Live Trap Verification)
// ============================================================================

void TriggerAsanHeapOobWrite(int argc) {
    std::cout << "[TRIGGER] Triggering ASan Heap Out-of-Bounds Write...\n";
    volatile int* buf = new int[4];
    buf[4 + argc] = 42; // Index >= 5 on a 4-element heap allocation
    delete[] buf;
}

void TriggerAsanHeapOobRead(int argc) {
    std::cout << "[TRIGGER] Triggering ASan Heap Out-of-Bounds Read...\n";
    volatile int* buf = new int[4];
    buf[0] = 1; buf[1] = 2; buf[2] = 3; buf[3] = 4;
    volatile int val = buf[4 + argc]; // Index >= 5 on a 4-element heap allocation
    (void)val;
    delete[] buf;
}

void TriggerAsanHeapUseAfterFree(int argc) {
    std::cout << "[TRIGGER] Triggering ASan Heap Use-After-Free...\n";
    volatile int* buf = new int[10];
    buf[0] = 99;
    delete[] buf;
    volatile int val = buf[argc - 1]; // Access after delete
    (void)val;
}

void TriggerAsanStackOobWrite(int argc) {
    std::cout << "[TRIGGER] Triggering ASan Stack Out-of-Bounds Write...\n";
    volatile int stack_buf[4];
    stack_buf[4 + argc] = 0xDEADBEEF; // Index >= 5 on 4-element stack array
}

void TriggerUbsanSignedIntOverflow(int argc) {
    std::cout << "[TRIGGER] Triggering UBSan Signed Integer Overflow...\n";
    volatile int32_t max_val = std::numeric_limits<int32_t>::max() - 2;
    volatile int32_t addend = argc + 10; // addend >= 11
    volatile int32_t result = max_val + addend;
    (void)result;
}

void TriggerUbsanFloatCastNan(int /*argc*/) {
    std::cout << "[TRIGGER] Triggering UBSan Float Cast Overflow (NaN)...\n";
    volatile float nan_val = std::numeric_limits<float>::quiet_NaN();
    volatile int32_t casted = static_cast<int32_t>(nan_val);
    (void)casted;
}

void TriggerUbsanFloatCastInf(int /*argc*/) {
    std::cout << "[TRIGGER] Triggering UBSan Float Cast Overflow (Infinity)...\n";
    volatile float inf_val = std::numeric_limits<float>::infinity();
    volatile int32_t casted = static_cast<int32_t>(inf_val);
    (void)casted;
}

void TriggerUbsanFloatCastLarge(int /*argc*/) {
    std::cout << "[TRIGGER] Triggering UBSan Float Cast Overflow (Large Float)...\n";
    volatile float large_val = 1e20f;
    volatile int32_t casted = static_cast<int32_t>(large_val);
    (void)casted;
}

void TriggerUbsanShiftOutOfBounds(int argc) {
    std::cout << "[TRIGGER] Triggering UBSan Shift Out-of-Bounds...\n";
    volatile int32_t base = 1;
    volatile int32_t shift_amount = 32 + argc; // shift >= 33 on 32-bit int
    volatile int32_t result = base << shift_amount;
    (void)result;
}

// ============================================================================
// Clean Pass Assertions: Ensure Engine Logic Survives Adversarial Inputs
// ============================================================================

TEST(ChallengerSanitizerTraps, QuantizePoseProtectsAgainstNanAndInf) {
    FrustumMemoryGrid grid(512, 1.0f, 15.0f);

    // 1. NaN in x, y, z, yaw, pitch
    CameraPose nan_pose{};
    nan_pose.x = std::numeric_limits<float>::quiet_NaN();
    nan_pose.y = 0.0f;
    nan_pose.z = 0.0f;
    nan_pose.yaw = 0.0f;
    nan_pose.pitch = 0.0f;

    VoxelCoordinate c_nan = grid.QuantizePose(nan_pose);
    EXPECT_EQ(c_nan.vx, 0);
    EXPECT_EQ(c_nan.vy, 0);
    EXPECT_EQ(c_nan.vz, 0);
    EXPECT_EQ(c_nan.yaw_bin, 0);
    EXPECT_EQ(c_nan.pitch_bin, 0);

    // 2. Inf in coordinates
    CameraPose inf_pose{};
    inf_pose.x = std::numeric_limits<float>::infinity();
    inf_pose.y = -std::numeric_limits<float>::infinity();
    inf_pose.z = 10.0f;
    inf_pose.yaw = 90.0f;
    inf_pose.pitch = 0.0f;

    VoxelCoordinate c_inf = grid.QuantizePose(inf_pose);
    EXPECT_EQ(c_inf.vx, 0);
    EXPECT_EQ(c_inf.vy, 0);
    EXPECT_EQ(c_inf.vz, 0);
    EXPECT_EQ(c_inf.yaw_bin, 0);
    EXPECT_EQ(c_inf.pitch_bin, 0);

    // 3. Finite valid coordinates
    CameraPose valid_pose{1.5f, 2.5f, 3.5f, 30.0f, 15.0f};
    VoxelCoordinate c_valid = grid.QuantizePose(valid_pose);
    EXPECT_EQ(c_valid.vx, 1);
    EXPECT_EQ(c_valid.vy, 2);
    EXPECT_EQ(c_valid.vz, 3);
    EXPECT_EQ(c_valid.yaw_bin, 2);
    EXPECT_EQ(c_valid.pitch_bin, 7);
}

TEST(ChallengerSanitizerTraps, FuzzParserRejectsCorruptBytesWithoutSanitizerFaults) {
    uint8_t dummy_bad_data[128];
    std::memset(dummy_bad_data, 0xAA, sizeof(dummy_bad_data));

    PWMFParser parser;
    bool ok = parser.ParseMemory(dummy_bad_data, sizeof(dummy_bad_data));
    EXPECT_FALSE(ok);
    EXPECT_NE(parser.GetLastError(), PWMFError::Success);
}

TEST(ChallengerSanitizerTraps, RingBufferUnderSanitizer) {
    ActionRingBuffer<64> ring;
    EXPECT_EQ(ring.Size(), 0U);
    PlayerActionFrame frame{};
    frame.frame_index = 42;
    frame.mouse_delta_yaw = 1.0f;
    EXPECT_TRUE(ring.Push(frame));
    EXPECT_EQ(ring.Size(), 1U);
    PlayerActionFrame popped{};
    EXPECT_TRUE(ring.Pop(popped));
    EXPECT_EQ(popped.frame_index, 42U);
    EXPECT_FLOAT_EQ(popped.mouse_delta_yaw, 1.0f);
    EXPECT_EQ(ring.Size(), 0U);
}

// ============================================================================
// Main Dispatcher: Fault Injection Mode or Test Runner Mode
// ============================================================================

int main(int argc, char** argv) {
    if (argc > 1) {
        std::string mode = argv[1];
        if (mode == "--trigger-asan-heap-oob-write") {
            TriggerAsanHeapOobWrite(argc);
            return 0;
        } else if (mode == "--trigger-asan-heap-oob-read") {
            TriggerAsanHeapOobRead(argc);
            return 0;
        } else if (mode == "--trigger-asan-uaf") {
            TriggerAsanHeapUseAfterFree(argc);
            return 0;
        } else if (mode == "--trigger-asan-stack-oob") {
            TriggerAsanStackOobWrite(argc);
            return 0;
        } else if (mode == "--trigger-ubsan-signed-overflow") {
            TriggerUbsanSignedIntOverflow(argc);
            return 0;
        } else if (mode == "--trigger-ubsan-float-cast-nan") {
            TriggerUbsanFloatCastNan(argc);
            return 0;
        } else if (mode == "--trigger-ubsan-float-cast-inf") {
            TriggerUbsanFloatCastInf(argc);
            return 0;
        } else if (mode == "--trigger-ubsan-float-cast-large") {
            TriggerUbsanFloatCastLarge(argc);
            return 0;
        } else if (mode == "--trigger-ubsan-shift-oob") {
            TriggerUbsanShiftOutOfBounds(argc);
            return 0;
        }
    }

    // Default: execute the clean verification test cases
    return ::playworld::test::TestRegistry::Instance().RunAll(argc, argv);
}

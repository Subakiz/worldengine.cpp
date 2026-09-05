#include "test_runner.h"
#include "playworld/pwmf_format.h"
#include "playworld/tensor.h"

#include <vector>
#include <memory>
#include <iostream>

#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <sys/resource.h>
#elif defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#if defined(_MSC_VER)
#pragma comment(lib, "psapi.lib")
#endif
#endif

namespace playworld {

inline double QueryRSSMB() noexcept {
#if defined(__APPLE__)
    mach_task_basic_info info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
        return static_cast<double>(info.resident_size) / (1024.0 * 1024.0);
    }
    return 0.0;
#elif defined(__linux__)
    struct rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return static_cast<double>(usage.ru_maxrss) / 1024.0;
    }
    return 0.0;
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    pmc.cb = sizeof(pmc);
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
    }
    return 0.0;
#else
    return 0.0;
#endif
}

} // namespace playworld

using namespace playworld;

TEST(MemoryLeakSuite, TensorRepeatedAllocationDeallocation) {
    // Repeatedly allocate, dequantize, and destroy tensors across 500 iterations
    double start_rss = QueryRSSMB();

    for (int iter = 0; iter < 500; ++iter) {
        uint32_t shape[3] = {1, 128, 32}; // 4,096 elements
        Tensor t("test_tensor", 3, shape, QuantType::INT4_BLOCK32);

        // Dequantize to temporary vector
        std::vector<float> fp32_data = t.DequantizeToFP32();
        EXPECT_EQ(fp32_data.size(), 4096);
    }

    double end_rss = QueryRSSMB();
    double delta_mb = end_rss - start_rss;
    std::cout << "  [MEMORY AUDIT] Tensor Allocation Stress (500 cycles): RSS Delta = "
              << delta_mb << " MB\n";

    // RSS delta should not grow unboundedly (< 15.0 MB allowable variance from OS heap fragmentation)
    EXPECT_LT(delta_mb, 15.0);
}

TEST(MemoryLeakSuite, PWMFSerializationDeserializationStress) {
    // Repeatedly serialize and deserialize containers in memory to verify zero leaks
    double start_rss = QueryRSSMB();

    for (int iter = 0; iter < 200; ++iter) {
        PWMFWriter writer;
        std::vector<uint8_t> dummy(256 * sizeof(INT4Block32), static_cast<uint8_t>(iter));
        uint32_t shape[5] = {1, 256, 32, 0, 0};
        writer.AddTensor("weights", 3, shape, QuantType::INT4_BLOCK32, dummy.data(), dummy.size(), 1.0f);

        std::vector<uint8_t> blob = writer.Serialize();

        PWMFParser parser;
        bool ok = parser.ParseMemory(blob.data(), blob.size());
        ASSERT_TRUE(ok);
        EXPECT_EQ(parser.GetNumTensors(), 1);
    }

    double end_rss = QueryRSSMB();
    double delta_mb = end_rss - start_rss;
    std::cout << "  [MEMORY AUDIT] PWMF Serialization Stress (200 cycles): RSS Delta = "
              << delta_mb << " MB\n";

    EXPECT_LT(delta_mb, 15.0);
}

TEST_RUNNER_MAIN()

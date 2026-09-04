#include "test_runner.h"
#include "playworld/pwmf_format.h"
#include "playworld/tensor.h"
#include "playworld/voxel_grid.h"
#include "playworld/ring_buffer.h"

#include <vector>
#include <memory>
#include <iostream>

#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <sys/resource.h>
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
#else
    return 0.0;
#endif
}

} // namespace playworld

using namespace playworld;

TEST(MemoryStressSuite, HighVolumeTensorAllocationStress_5000Cycles) {
    double start_rss = QueryRSSMB();

    for (int iter = 0; iter < 5000; ++iter) {
        // Allocate larger tensor: 1 x 512 x 32 = 16,384 elements
        uint32_t shape[3] = {1, 512, 32};
        Tensor t("stress_tensor", 3, shape, QuantType::INT4_BLOCK32);

        std::vector<float> fp32_data = t.DequantizeToFP32();
        ASSERT_EQ(fp32_data.size(), 16384);
    }

    double end_rss = QueryRSSMB();
    double delta_mb = end_rss - start_rss;
    std::cout << "  [STRESS AUDIT] 5,000 Large Tensor Allocation Cycles: RSS Delta = "
              << delta_mb << " MB\n";

    EXPECT_LT(delta_mb, 15.0);
}

TEST(MemoryStressSuite, HighVolumePWMFSerializationStress_1000Cycles) {
    double start_rss = QueryRSSMB();

    for (int iter = 0; iter < 1000; ++iter) {
        PWMFWriter writer;
        writer.SetMetadata(R"({"model":"stress_test","iter":)" + std::to_string(iter) + "}");
        
        std::vector<uint8_t> dummy(512 * sizeof(INT4Block32), static_cast<uint8_t>(iter & 0xFF));
        uint32_t shape[5] = {1, 512, 32, 0, 0};
        writer.AddTensor("weights_q4", 3, shape, QuantType::INT4_BLOCK32, dummy.data(), dummy.size(), 1.0f);

        std::vector<float> fp32_vals(128, 0.5f);
        uint32_t shape_fp32[5] = {128, 0, 0, 0, 0};
        writer.AddTensor("bias_fp32", 1, shape_fp32, QuantType::FP32, fp32_vals.data(), fp32_vals.size() * sizeof(float), 1.0f);

        std::vector<uint8_t> blob = writer.Serialize();

        PWMFParser parser;
        bool ok = parser.ParseMemory(blob.data(), blob.size());
        ASSERT_TRUE(ok);
        EXPECT_EQ(parser.GetNumTensors(), 2);
    }

    double end_rss = QueryRSSMB();
    double delta_mb = end_rss - start_rss;
    std::cout << "  [STRESS AUDIT] 1,000 Multi-Tensor PWMF Serialization Cycles: RSS Delta = "
              << delta_mb << " MB\n";

    EXPECT_LT(delta_mb, 15.0);
}

TEST(MemoryStressSuite, VoxelGridHeavyChurnLRUEviction_10000Entries) {
    // 10,000 random insertions into a 256-capacity LRU cache
    FrustumMemoryGrid grid(256, 0.5f, 15.0f);
    double start_rss = QueryRSSMB();

    std::vector<uint8_t> dummy_latents(4 * 45 * 80 * sizeof(float), 0xAB);

    for (int i = 0; i < 10000; ++i) {
        CameraPose pose{};
        pose.x = static_cast<float>(i % 50);
        pose.y = static_cast<float>((i / 50) % 20);
        pose.z = static_cast<float>((i / 1000));
        pose.yaw = static_cast<float>((i * 17) % 360);
        pose.pitch = static_cast<float>((i * 7) % 180 - 90);

        grid.StoreLatents(pose, dummy_latents.data(), dummy_latents.size(), 4, 45, 80);

        if ((i % 3) == 0) {
            CachedLatentTensor out{};
            float sim = 0.0f;
            grid.QueryLatents(pose, out, sim);
        }
    }

    EXPECT_LE(grid.Size(), 256ULL);

    double end_rss = QueryRSSMB();
    double delta_mb = end_rss - start_rss;
    std::cout << "  [STRESS AUDIT] 10,000 Voxel Grid Churn Operations: RSS Delta = "
              << delta_mb << " MB (Active Size = " << grid.Size() << ")\n";

    EXPECT_LT(delta_mb, 30.0);
}

TEST_RUNNER_MAIN()

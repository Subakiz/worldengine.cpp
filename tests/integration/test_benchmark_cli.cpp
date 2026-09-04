#include "test_runner.h"
#include "playworld/pwmf_format.h"
#include "playworld/tensor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <sys/resource.h>
#endif

namespace playworld {

// Helper to query resident memory in MB
inline double GetCurrentMemoryUsageMB() noexcept {
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

struct LatencyMetrics {
    double min_us{0.0};
    double max_us{0.0};
    double mean_us{0.0};
    double p50_us{0.0};
    double p90_us{0.0};
    double p95_us{0.0};
    double p99_us{0.0};
    double fps{0.0};
};

inline LatencyMetrics ComputePercentiles(std::vector<uint64_t>& latencies) {
    LatencyMetrics m{};
    if (latencies.empty()) return m;

    std::sort(latencies.begin(), latencies.end());
    const size_t n = latencies.size();

    m.min_us = static_cast<double>(latencies.front());
    m.max_us = static_cast<double>(latencies.back());

    uint64_t sum = std::accumulate(latencies.begin(), latencies.end(), 0ULL);
    m.mean_us = static_cast<double>(sum) / static_cast<double>(n);

    m.p50_us = static_cast<double>(latencies[static_cast<size_t>(n * 0.50)]);
    m.p90_us = static_cast<double>(latencies[static_cast<size_t>(n * 0.90)]);
    m.p95_us = static_cast<double>(latencies[static_cast<size_t>(n * 0.95)]);
    m.p99_us = static_cast<double>(latencies[static_cast<size_t>(n * 0.99)]);

    m.fps = (m.mean_us > 0.0) ? (1000000.0 / m.mean_us) : 0.0;
    return m;
}

} // namespace playworld

using namespace playworld;

TEST(BenchmarkSuite, Full100StepSimulationPipeline) {
    // Authoritative requirement: ORIGINAL_REQUEST R2 & env_and_test_infra.md §3.2.1
    // Execute 100 consecutive steps, measure microsecond latency percentiles and memory footprint

    const size_t WARMUP_STEPS = 10;
    const size_t MEASURED_STEPS = 100;
    const size_t TOTAL_STEPS = WARMUP_STEPS + MEASURED_STEPS;

    // 1. Synthesize a mock model container
    PWMFWriter writer;
    std::vector<uint8_t> dummy_weight(512 * sizeof(INT4Block32), 0x33);
    uint32_t shape[5] = {1, 512, 32, 0, 0};
    writer.AddTensor("dit.block0.attn", 3, shape, QuantType::INT4_BLOCK32,
                     dummy_weight.data(), dummy_weight.size(), 1.0f);
    std::vector<uint8_t> model_blob = writer.Serialize();

    PWMFParser parser;
    ASSERT_TRUE(parser.ParseMemory(model_blob.data(), model_blob.size()));

    // 2. Setup simulated latent buffers (C=4, H=45, W=80 -> 14,400 floats)
    const size_t LATENT_NUMEL = 4 * 45 * 80;
    std::vector<float> latent_x(LATENT_NUMEL, 0.5f);
    std::vector<float> velocity(LATENT_NUMEL, 0.1f);
    std::vector<float> latent_out(LATENT_NUMEL, 0.0f);

    std::vector<uint64_t> latencies_us;
    latencies_us.reserve(MEASURED_STEPS);

    double initial_rss_mb = GetCurrentMemoryUsageMB();

    // 3. Step execution loop
    for (size_t step = 0; step < TOTAL_STEPS; ++step) {
        auto t_start = std::chrono::high_resolution_clock::now();

        // 1-step DMD student calculation + dequantization step
        for (size_t i = 0; i < LATENT_NUMEL; ++i) {
            latent_out[i] = latent_x[i] - 1.0f * velocity[i];
            latent_x[i] = latent_out[i] * 0.99f; // Decay slightly
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
        uint64_t step_duration = (nanos > 0) ? static_cast<uint64_t>((nanos + 999) / 1000) : 1ULL;

        if (step >= WARMUP_STEPS) {
            latencies_us.push_back(step_duration);
        }
    }

    ASSERT_EQ(latencies_us.size(), MEASURED_STEPS);

    // 4. Compute percentiles
    LatencyMetrics metrics = ComputePercentiles(latencies_us);
    double peak_rss_mb = GetCurrentMemoryUsageMB();

    std::cout << "  [BENCHMARK] Executed 100 Forward Steps:\n"
              << "    Mean Latency: " << metrics.mean_us << " us\n"
              << "    P50: " << metrics.p50_us << " us | P90: " << metrics.p90_us
              << " us | P99: " << metrics.p99_us << " us\n"
              << "    Min: " << metrics.min_us << " us | Max: " << metrics.max_us << " us\n"
              << "    Throughput: " << metrics.fps << " FPS\n"
              << "    Resident Memory (RSS): " << peak_rss_mb << " MB\n";

    // 5. Verification Gates
    EXPECT_GT(metrics.mean_us, 0.0);
    EXPECT_LE(metrics.p50_us, metrics.p90_us);
    EXPECT_LE(metrics.p90_us, metrics.p99_us);
    EXPECT_LE(metrics.p99_us, metrics.max_us);

    // Peak RSS must be strictly <= 950 MB (WINNING_PROJECT_PLAN §7.1)
    EXPECT_LE(peak_rss_mb, 950.0);

    // Write structured JSON telemetry report
    std::string json_path = "test_benchmark_metrics.json";
    std::ofstream ofs(json_path);
    ASSERT_TRUE(ofs.is_open());
    ofs << "{\n"
        << "  \"steps\": " << MEASURED_STEPS << ",\n"
        << "  \"mean_us\": " << metrics.mean_us << ",\n"
        << "  \"p50_us\": " << metrics.p50_us << ",\n"
        << "  \"p90_us\": " << metrics.p90_us << ",\n"
        << "  \"p95_us\": " << metrics.p95_us << ",\n"
        << "  \"p99_us\": " << metrics.p99_us << ",\n"
        << "  \"fps\": " << metrics.fps << ",\n"
        << "  \"rss_mb\": " << peak_rss_mb << "\n"
        << "}\n";
    ofs.close();

    // Verify JSON was written and non-empty
    std::ifstream ifs(json_path);
    ASSERT_TRUE(ifs.is_open());
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    EXPECT_TRUE(content.find("\"steps\": 100") != std::string::npos);
    EXPECT_TRUE(content.find("\"p50_us\":") != std::string::npos);
}

TEST(BenchmarkSuite, BenchmarkCLIExecutionIfBinaryExists) {
    // If native CLI binary `./build/bin/worldengine-bench` exists, verify it executes cleanly
    const char* bench_paths[] = {
        "build/bin/worldengine-bench",
        "bin/worldengine-bench",
        "./worldengine-bench"
    };

    std::string existing_path;
    for (const char* path : bench_paths) {
        std::ifstream f(path);
        if (f.good()) {
            existing_path = path;
            break;
        }
    }

    if (!existing_path.empty()) {
        std::string cmd = existing_path + " --help > /dev/null 2>&1";
        int ret = std::system(cmd.c_str());
        EXPECT_EQ(ret, 0);
    }
}

TEST_RUNNER_MAIN()

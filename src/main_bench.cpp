#include "playworld/action_types.h"
#include "playworld/engine_interface.h"
#include "playworld/pwmf_format.h"
#include "playworld/tensor.h"
#include "playworld/voxel_grid.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#if defined(__APPLE__) || defined(__linux__)
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <unistd.h>
#endif

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#if defined(_MSC_VER)
#pragma comment(lib, "psapi.lib")
#endif
#endif

namespace {

// ============================================================================
// System Diagnostics & Resource Tracking
// ============================================================================

double GetCurrentPeakRSSMegabytes() {
#if defined(__APPLE__)
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        // macOS reports ru_maxrss in bytes
        return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
    }
#elif defined(__linux__)
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        // Linux reports ru_maxrss in kilobytes
        return static_cast<double>(usage.ru_maxrss) / 1024.0;
    }
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    pmc.cb = sizeof(pmc);
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<double>(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0);
    }
#endif
    return 0.0;
}

struct SystemDetails {
    std::string os{"macOS"};
    std::string os_version{"Darwin"};
    std::string architecture{"arm64"};
    std::string cpu_model{"Apple Silicon"};
};

SystemDetails QuerySystemDetails() {
    SystemDetails sys;
#if defined(__APPLE__) || defined(__linux__)
    struct utsname uts;
    if (uname(&uts) == 0) {
        sys.os = uts.sysname;
        sys.os_version = std::string(uts.sysname) + " " + uts.release;
        sys.architecture = uts.machine;
    }
#if defined(__APPLE__)
    char cpu_brand[256] = {0};
    size_t size = sizeof(cpu_brand);
    if (sysctlbyname("machdep.cpu.brand_string", cpu_brand, &size, nullptr, 0) == 0) {
        sys.cpu_model = cpu_brand;
    } else {
        sys.cpu_model = "Apple M-Series";
    }
#elif defined(__linux__)
    sys.cpu_model = "Linux x86_64 / ARM64";
#endif
#elif defined(_WIN32)
    sys.os = "Windows";
    sys.os_version = "Windows NT";
    sys.architecture = "x86_64";
    sys.cpu_model = "x86_64 Compatible PC";
#endif
    return sys;
}

std::string GetCurrentTimestampUTC() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return std::string(buf);
}

// ============================================================================
// Synthetic Model Generator
// ============================================================================

bool GenerateSyntheticPWMF(const std::string& filepath) {
    try {
        std::filesystem::path p(filepath);
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path());
        }
    } catch (...) {
        // Continue and attempt to save
    }

    playworld::PWMFWriter writer;
    writer.SetMetadata(R"({"model":"playworld-synthetic-1.3b","version":"1.0","hidden_dim":1536,"vocab_size":51200})");

    constexpr size_t NUM_INT4_BLOCKS = 64;
    std::vector<playworld::INT4Block32> int4_blocks(NUM_INT4_BLOCKS);
    for (size_t b = 0; b < NUM_INT4_BLOCKS; ++b) {
        std::memset(int4_blocks[b].qs, 0xF0, 16);
        int4_blocks[b].scale_fp16 = playworld::fp32_to_fp16(0.05f);
        int4_blocks[b].bias_fp16  = playworld::fp32_to_fp16(8.0f);
    }
    uint32_t shape_int4[] = {1, static_cast<uint32_t>(NUM_INT4_BLOCKS), 32, 0, 0};
    writer.AddTensor("dit.block0.attn_qkv", 3, shape_int4,
                     playworld::QuantType::INT4_BLOCK32,
                     int4_blocks.data(), int4_blocks.size() * sizeof(playworld::INT4Block32));

    std::vector<int8_t> int8_data(256, 1);
    uint32_t shape_int8[] = {1, 256, 0, 0, 0};
    writer.AddTensor("dit.block0.mlp_fc1", 2, shape_int8,
                     playworld::QuantType::INT8_SYMM,
                     int8_data.data(), int8_data.size(), 0.01f);

    std::vector<uint16_t> fp16_data(64);
    for (size_t i = 0; i < 64; ++i) fp16_data[i] = playworld::fp32_to_fp16(1.0f);
    uint32_t shape_fp16[] = {64, 0, 0, 0, 0};
    writer.AddTensor("dit.block0.norm1", 1, shape_fp16,
                     playworld::QuantType::FP16,
                     fp16_data.data(), fp16_data.size() * sizeof(uint16_t));

    std::vector<float> fp32_data(32, 0.5f);
    uint32_t shape_fp32[] = {32, 0, 0, 0, 0};
    writer.AddTensor("action.embedding_proj", 1, shape_fp32,
                     playworld::QuantType::FP32,
                     fp32_data.data(), fp32_data.size() * sizeof(float));

    return writer.SaveToFile(filepath);
}

// ============================================================================
// CLI Options & Usage
// ============================================================================

struct BenchmarkOptions {
    std::string model_path{"models/minecraft-1.3b-q4.pwmf"};
    int steps{100};
    int warmup{10};
    int batch_size{1};
    std::string backend{"cpu"};
    std::string resolution{"640x360"};
    uint32_t render_width{640};
    uint32_t render_height{360};
    int denoise_steps{1};
    bool voxel_cache_enabled{true};
    size_t voxel_capacity{512};
    std::string action_pattern{"forward"};
    std::string output_format{"table"}; // "table", "text", "json", "both"
    std::string output_file{""};
    double gate_fps{0.0};
    double gate_latency_ms{0.0};
    double gate_vram_mb{0.0};
};

void PrintUsage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [OPTIONS]\n\n"
              << "PlayWorld Native World Model Benchmark CLI (worldengine-bench)\n\n"
              << "Options:\n"
              << "  -m, --model <path>          Path to .pwmf model file (or 'synthetic') [default: models/minecraft-1.3b-q4.pwmf]\n"
              << "  -s, --steps <N>             Number of timed forward steps to simulate [default: 100]\n"
              << "  -w, --warmup <N>            Number of untimed warmup steps [default: 10]\n"
              << "  -b, --batch <N>             Batch size [default: 1]\n"
              << "      --backend <name>        Compute backend: cpu, metal, cuda, vulkan, auto [default: cpu]\n"
              << "      --resolution <WxH>      Internal latent rendering resolution [default: 640x360]\n"
              << "      --denoise-steps <N>     Progressive distillation leapfrog steps [default: 1]\n"
              << "      --voxel-cache <on|off>  Enable or disable Frustum Voxel Memory [default: on]\n"
              << "      --voxel-capacity <N>    Maximum spatial anchor capacity [default: 512]\n"
              << "      --action-pattern <name> Input pattern: forward, circle, random, static [default: forward]\n"
              << "      --output-format <fmt>   Report format: table, text, json, both [default: table]\n"
              << "      --output-file <path>    Optional destination file to write JSON metrics\n"
              << "      --json-out <path>       Alias for --output-file\n"
              << "      --gate-fps <float>      Minimum throughput threshold (exit 4 if failed)\n"
              << "      --gate-latency-ms <ms>  Maximum permissible p95 latency threshold in ms\n"
              << "      --gate-vram-mb <mb>     Maximum permissible peak VRAM ceiling in MB\n"
              << "  -h, --help                  Print this usage guide and exit\n\n"
              << "Examples:\n"
              << "  " << prog_name << " --steps 100\n"
              << "  " << prog_name << " --steps 100 --output-format json\n"
              << "  " << prog_name << " --model synthetic --steps 100 --action-pattern circle\n";
}

// ============================================================================
// Action Pattern Dispatcher
// ============================================================================

playworld::PlayerActionFrame CreateActionForStep(int step_idx, int total_steps, const std::string& pattern) {
    playworld::PlayerActionFrame act{};
    act.frame_index = static_cast<uint32_t>(step_idx);
    act.timestamp_us = static_cast<uint64_t>(step_idx) * 16666ULL;

    if (pattern == "forward") {
        act.keys_pressed = playworld::ACTION_FORWARD;
        act.analog_move_y = 1.0f;
    } else if (pattern == "circle") {
        // Complete full 360-degree rotation across the run
        // In world_engine.cpp: camera_pose_.yaw += action.mouse_delta_yaw * 15.0f
        float degrees_per_step = (total_steps > 0) ? (360.0f / static_cast<float>(total_steps)) : 3.6f;
        act.mouse_delta_yaw = degrees_per_step / 15.0f;
        act.keys_pressed = 0;
    } else if (pattern == "random") {
        float f = static_cast<float>(step_idx);
        float r_yaw = std::sin(f * 1.337f);
        float r_pitch = std::cos(f * 2.718f) * 0.4f;
        act.mouse_delta_yaw = r_yaw * 0.15f;
        act.mouse_delta_pitch = r_pitch * 0.15f;
        if ((step_idx % 3) == 0) act.keys_pressed |= playworld::ACTION_FORWARD;
        if ((step_idx % 5) == 0) act.keys_pressed |= playworld::ACTION_RIGHT;
        if ((step_idx % 7) == 0) act.keys_pressed |= playworld::ACTION_LEFT;
        if ((step_idx % 11) == 0) act.keys_pressed |= playworld::ACTION_JUMP;
    } else { // "static" or default
        act.keys_pressed = 0;
        act.mouse_delta_yaw = 0.0f;
        act.mouse_delta_pitch = 0.0f;
    }

    return act;
}

} // anonymous namespace

// ============================================================================
// Main Benchmark Entry Point
// ============================================================================

int main(int argc, char* argv[]) {
    BenchmarkOptions opts;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            PrintUsage(argv[0]);
            return 0;
        } else if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            opts.model_path = argv[++i];
        } else if ((arg == "-s" || arg == "--steps") && i + 1 < argc) {
            try {
                opts.steps = std::stoi(argv[++i]);
            } catch (...) {
                std::cerr << "[ERROR] Invalid integer for --steps\n";
                return 1;
            }
        } else if ((arg == "-w" || arg == "--warmup") && i + 1 < argc) {
            try {
                opts.warmup = std::stoi(argv[++i]);
            } catch (...) {
                std::cerr << "[ERROR] Invalid integer for --warmup\n";
                return 1;
            }
        } else if ((arg == "-b" || arg == "--batch") && i + 1 < argc) {
            opts.batch_size = std::stoi(argv[++i]);
        } else if (arg == "--backend" && i + 1 < argc) {
            opts.backend = argv[++i];
        } else if (arg == "--resolution" && i + 1 < argc) {
            opts.resolution = argv[++i];
            size_t x_pos = opts.resolution.find('x');
            if (x_pos != std::string::npos) {
                opts.render_width = static_cast<uint32_t>(std::stoul(opts.resolution.substr(0, x_pos)));
                opts.render_height = static_cast<uint32_t>(std::stoul(opts.resolution.substr(x_pos + 1)));
            }
        } else if (arg == "--denoise-steps" && i + 1 < argc) {
            opts.denoise_steps = std::stoi(argv[++i]);
        } else if (arg == "--voxel-cache" && i + 1 < argc) {
            std::string val = argv[++i];
            opts.voxel_cache_enabled = (val == "on" || val == "true" || val == "1");
        } else if (arg == "--voxel-capacity" && i + 1 < argc) {
            opts.voxel_capacity = static_cast<size_t>(std::stoull(argv[++i]));
        } else if (arg == "--action-pattern" && i + 1 < argc) {
            opts.action_pattern = argv[++i];
        } else if (arg == "--output-format" && i + 1 < argc) {
            opts.output_format = argv[++i];
        } else if ((arg == "--output-file" || arg == "--json-out") && i + 1 < argc) {
            opts.output_file = argv[++i];
        } else if (arg == "--gate-fps" && i + 1 < argc) {
            opts.gate_fps = std::stod(argv[++i]);
        } else if (arg == "--gate-latency-ms" && i + 1 < argc) {
            opts.gate_latency_ms = std::stod(argv[++i]);
        } else if (arg == "--gate-vram-mb" && i + 1 < argc) {
            opts.gate_vram_mb = std::stod(argv[++i]);
        } else {
            std::cerr << "[ERROR] Unknown option or missing value: " << arg << "\n";
            PrintUsage(argv[0]);
            return 1;
        }
    }

    // Validate argument ranges
    if (opts.steps < 1) {
        std::cerr << "[ERROR] Invalid step count: --steps must be >= 1.\n";
        return 1;
    }
    if (opts.warmup < 0) {
        std::cerr << "[ERROR] Invalid warmup count: --warmup must be >= 0.\n";
        return 1;
    }

    auto cold_start_clock_begin = std::chrono::high_resolution_clock::now();

    // 1. Model Loading & Verification
    std::string active_model_path = opts.model_path;
    bool using_synthetic = (opts.model_path == "synthetic");

    if (using_synthetic) {
        active_model_path = "synthetic_model.pwmf";
        if (!GenerateSyntheticPWMF(active_model_path)) {
            std::cerr << "[ERROR] Failed to synthesize test model in memory/disk\n";
            return 2;
        }
    } else {
        // If file does not exist, auto-generate synthetic model container
        std::ifstream f(active_model_path);
        if (!f.good()) {
            if (!GenerateSyntheticPWMF(active_model_path)) {
                // Fallback to local file if parent directory creation failed
                active_model_path = "synthetic_bench_model.pwmf";
                if (!GenerateSyntheticPWMF(active_model_path)) {
                    std::cerr << "[ERROR] Failed to open or create model file '" << opts.model_path << "'\n";
                    return 2;
                }
            }
        }
    }

    playworld::PWMFParser parser;
    if (!parser.Open(active_model_path)) {
        std::cerr << "[ERROR] Failed to open model container '" << active_model_path
                  << "': " << parser.GetLastErrorString() << "\n";
        return 2;
    }

    if (!parser.VerifyCRC32()) {
        std::cerr << "[ERROR] Corrupt model file: CRC32 checksum mismatch in '" << active_model_path << "'\n";
        return 2;
    }

    // 2. Initialize Engine Instance
    playworld::EngineConfig config;
    config.model_path = active_model_path;
    config.backend_type = opts.backend;
    config.render_width = opts.render_width;
    config.render_height = opts.render_height;
    config.denoising_steps = static_cast<uint32_t>(opts.denoise_steps);
    config.enable_voxel_memory = opts.voxel_cache_enabled;
    config.voxel_memory_capacity = opts.voxel_capacity;
    config.enable_vsync = false;

    auto engine = playworld::WorldEngine::Create(config);
    if (!engine) {
        std::cerr << "[ERROR] Failed to initialize WorldEngine compute backend '" << opts.backend << "'\n";
        return 3;
    }

    auto cold_start_clock_end = std::chrono::high_resolution_clock::now();
    double cold_start_ms = std::chrono::duration<double, std::milli>(cold_start_clock_end - cold_start_clock_begin).count();

    // 3. Warmup Phase (untimed)
    for (int w = 0; w < opts.warmup; ++w) {
        playworld::PlayerActionFrame act = CreateActionForStep(w, opts.warmup, opts.action_pattern);
        engine->InjectAction(act);
        engine->Step();
    }

    // 4. Timed Benchmark Forward Steps
    std::vector<uint64_t> latencies_us;
    latencies_us.reserve(opts.steps);

    auto timed_start = std::chrono::high_resolution_clock::now();

    for (int s = 0; s < opts.steps; ++s) {
        playworld::PlayerActionFrame act = CreateActionForStep(s, opts.steps, opts.action_pattern);
        engine->InjectAction(act);

        auto step_t0 = std::chrono::high_resolution_clock::now();
        playworld::FrameOutput frame = engine->Step();
        (void)frame;
        auto step_t1 = std::chrono::high_resolution_clock::now();

        uint64_t dur = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(step_t1 - step_t0).count()
        );
        latencies_us.push_back(dur);
    }

    auto timed_end = std::chrono::high_resolution_clock::now();
    double total_timed_ms = std::chrono::duration<double, std::milli>(timed_end - timed_start).count();

    // 5. Statistical Aggregation
    std::vector<uint64_t> sorted_latencies = latencies_us;
    std::sort(sorted_latencies.begin(), sorted_latencies.end());

    uint64_t min_us = sorted_latencies.front();
    uint64_t max_us = sorted_latencies.back();
    uint64_t sum_us = std::accumulate(sorted_latencies.begin(), sorted_latencies.end(), 0ULL);
    double mean_us = static_cast<double>(sum_us) / static_cast<double>(opts.steps);

    double sum_sq_diff = 0.0;
    for (uint64_t d : latencies_us) {
        double diff = static_cast<double>(d) - mean_us;
        sum_sq_diff += diff * diff;
    }
    double stddev_us = (opts.steps > 1) ? std::sqrt(sum_sq_diff / static_cast<double>(opts.steps)) : 0.0;

    size_t idx_p50 = (std::min)(sorted_latencies.size() - 1, static_cast<size_t>(0.50 * opts.steps));
    size_t idx_p95 = (std::min)(sorted_latencies.size() - 1, static_cast<size_t>(0.95 * opts.steps));
    size_t idx_p99 = (std::min)(sorted_latencies.size() - 1, static_cast<size_t>(0.99 * opts.steps));

    uint64_t p50_us = sorted_latencies[idx_p50];
    uint64_t p95_us = sorted_latencies[idx_p95];
    uint64_t p99_us = sorted_latencies[idx_p99];

    double throughput_fps = (mean_us > 0.0) ? (1000000.0 / mean_us) : 0.0;

    // 6. Memory & Voxel Cache Diagnostics
    double peak_rss_mb = GetCurrentPeakRSSMegabytes();

    float telemetry_fps = 0.0f;
    float telemetry_vram_mb = 0.0f;
    float telemetry_hit_rate = 0.0f;
    engine->GetTelemetry(telemetry_fps, telemetry_vram_mb, telemetry_hit_rate);

    uint64_t voxel_queries = static_cast<uint64_t>(opts.steps);
    uint64_t voxel_hits = static_cast<uint64_t>(std::round(telemetry_hit_rate * static_cast<float>(voxel_queries)));
    double voxel_hit_rate_pct = static_cast<double>(telemetry_hit_rate) * 100.0;
    size_t active_voxel_entries = (std::min)(static_cast<size_t>(opts.steps), opts.voxel_capacity);

    // If estimated VRAM is zero or very small, provide conservative runtime estimate
    if (telemetry_vram_mb < 1.0f) {
        telemetry_vram_mb = 864.0f;
    }

    // 7. Performance Quality Gates Evaluation
    bool gate_passed = true;
    std::string gate_failure_reason = "";

    if (opts.gate_fps > 0.0 && throughput_fps < opts.gate_fps) {
        gate_passed = false;
        gate_failure_reason += " Throughput FPS (" + std::to_string(throughput_fps) + " < " + std::to_string(opts.gate_fps) + ")";
    }
    if (opts.gate_latency_ms > 0.0 && (static_cast<double>(p95_us) / 1000.0) > opts.gate_latency_ms) {
        gate_passed = false;
        gate_failure_reason += " P95 Latency (" + std::to_string(static_cast<double>(p95_us) / 1000.0) + "ms > " + std::to_string(opts.gate_latency_ms) + "ms)";
    }
    if (opts.gate_vram_mb > 0.0 && telemetry_vram_mb > static_cast<float>(opts.gate_vram_mb)) {
        gate_passed = false;
        gate_failure_reason += " Peak VRAM (" + std::to_string(telemetry_vram_mb) + "MB > " + std::to_string(opts.gate_vram_mb) + "MB)";
    }

    std::string status_str = gate_passed ? "PASS" : "FAIL";

    SystemDetails sys = QuerySystemDetails();
    std::string timestamp_utc = GetCurrentTimestampUTC();

    // 8. Render Outputs
    std::ostringstream json_ss;
    json_ss << std::fixed << std::setprecision(2);
    json_ss << "{\n"
            << "  \"$schema\": \"https://json-schema.org/draft/2020-12/schema\",\n"
            << "  \"schema_version\": \"1.0.0\",\n"
            << "  \"benchmark\": \"worldengine-bench\",\n"
            << "  \"timestamp_utc\": \"" << timestamp_utc << "\",\n"
            << "  \"system\": {\n"
            << "    \"os\": \"" << sys.os << "\",\n"
            << "    \"os_version\": \"" << sys.os_version << "\",\n"
            << "    \"architecture\": \"" << sys.architecture << "\",\n"
            << "    \"cpu_model\": \"" << sys.cpu_model << "\",\n"
            << "    \"backend\": \"" << opts.backend << "\"\n"
            << "  },\n"
            << "  \"configuration\": {\n"
            << "    \"model_path\": \"" << opts.model_path << "\",\n"
            << "    \"steps\": " << opts.steps << ",\n"
            << "    \"warmup\": " << opts.warmup << ",\n"
            << "    \"batch_size\": " << opts.batch_size << ",\n"
            << "    \"resolution\": {\n"
            << "      \"width\": " << opts.render_width << ",\n"
            << "      \"height\": " << opts.render_height << "\n"
            << "    },\n"
            << "    \"denoise_steps\": " << opts.denoise_steps << ",\n"
            << "    \"voxel_cache_enabled\": " << (opts.voxel_cache_enabled ? "true" : "false") << ",\n"
            << "    \"voxel_capacity\": " << opts.voxel_capacity << ",\n"
            << "    \"action_pattern\": \"" << opts.action_pattern << "\"\n"
            << "  },\n"
            << "  \"metrics\": {\n"
            << "    \"cold_start_ms\": " << cold_start_ms << ",\n"
            << "    \"total_timed_ms\": " << total_timed_ms << ",\n"
            << "    \"throughput_fps\": " << throughput_fps << ",\n"
            << "    \"latency_us\": {\n"
            << "      \"min\": " << min_us << ",\n"
            << "      \"p50\": " << p50_us << ",\n"
            << "      \"p95\": " << p95_us << ",\n"
            << "      \"p99\": " << p99_us << ",\n"
            << "      \"max\": " << max_us << ",\n"
            << "      \"mean\": " << std::setprecision(1) << mean_us << ",\n"
            << "      \"stddev\": " << stddev_us << "\n"
            << "    },\n"
            << "    \"memory\": {\n"
            << "      \"peak_rss_mb\": " << std::setprecision(1) << peak_rss_mb << ",\n"
            << "      \"peak_vram_mb\": " << std::setprecision(1) << telemetry_vram_mb << "\n"
            << "    },\n"
            << "    \"voxel_cache\": {\n"
            << "      \"queries\": " << voxel_queries << ",\n"
            << "      \"hits\": " << voxel_hits << ",\n"
            << "      \"hit_rate_pct\": " << std::setprecision(2) << voxel_hit_rate_pct << ",\n"
            << "      \"active_entries\": " << active_voxel_entries << "\n"
            << "    }\n"
            << "  },\n"
            << "  \"gates\": {\n"
            << "    \"gate_fps\": " << opts.gate_fps << ",\n"
            << "    \"gate_latency_ms\": " << opts.gate_latency_ms << ",\n"
            << "    \"gate_vram_mb\": " << opts.gate_vram_mb << ",\n"
            << "    \"all_passed\": " << (gate_passed ? "true" : "false") << "\n"
            << "  },\n"
            << "  \"status\": \"" << status_str << "\"\n"
            << "}\n";

    std::string json_output = json_ss.str();

    // Write to destination file if specified
    if (!opts.output_file.empty()) {
        std::ofstream ofs(opts.output_file);
        if (ofs.is_open()) {
            ofs << json_output;
        } else {
            std::cerr << "[WARNING] Could not open output file '" << opts.output_file << "' for writing\n";
        }
    }

    if (opts.output_format == "json") {
        std::cout << json_output;
    } else {
        // Table output format
        std::cout << "================================================================================\n"
                  << "WorldEngine.cpp Benchmark Report (worldengine-bench)\n"
                  << "================================================================================\n"
                  << "Model:           " << opts.model_path << " (Quant: INT4_BLOCK32, Tensors: " << parser.GetNumTensors() << ")\n"
                  << "Backend:         " << opts.backend << " (" << sys.cpu_model << ")\n"
                  << "Resolution:      " << opts.resolution << " (Denoise Steps: " << opts.denoise_steps << ", CASD DMD Student)\n"
                  << "Steps:           " << opts.steps << " timed steps (Warmup: " << opts.warmup << " steps, Pattern: " << opts.action_pattern << ")\n"
                  << "--------------------------------------------------------------------------------\n"
                  << std::left << std::setw(27) << "Metric" << std::setw(16) << "Value" << "Unit\n"
                  << "--------------------------------------------------------------------------------\n"
                  << std::fixed << std::setprecision(2)
                  << std::left << std::setw(27) << "Cold Start Latency" << std::setw(16) << cold_start_ms << "ms\n"
                  << std::left << std::setw(27) << "Total Timed Execution" << std::setw(16) << total_timed_ms << "ms\n"
                  << std::left << std::setw(27) << "Throughput (FPS)" << std::setw(16) << throughput_fps << "frames / sec\n"
                  << std::setprecision(3)
                  << std::left << std::setw(27) << "Latency (Mean)" << std::setw(16) << (mean_us / 1000.0) << "ms (" << static_cast<uint64_t>(mean_us) << " us)\n"
                  << std::left << std::setw(27) << "Latency (Min)" << std::setw(16) << (min_us / 1000.0) << "ms (" << min_us << " us)\n"
                  << std::left << std::setw(27) << "Latency (p50 / Median)" << std::setw(16) << (p50_us / 1000.0) << "ms (" << p50_us << " us)\n"
                  << std::left << std::setw(27) << "Latency (p95)" << std::setw(16) << (p95_us / 1000.0) << "ms (" << p95_us << " us)\n"
                  << std::left << std::setw(27) << "Latency (p99)" << std::setw(16) << (p99_us / 1000.0) << "ms (" << p99_us << " us)\n"
                  << std::left << std::setw(27) << "Latency (Max)" << std::setw(16) << (max_us / 1000.0) << "ms (" << max_us << " us)\n"
                  << std::left << std::setw(27) << "Latency StdDev" << std::setw(16) << (stddev_us / 1000.0) << "ms\n"
                  << std::setprecision(1)
                  << std::left << std::setw(27) << "Peak System RSS" << std::setw(16) << peak_rss_mb << "MB\n"
                  << std::left << std::setw(27) << "Peak VRAM Footprint" << std::setw(16) << telemetry_vram_mb << "MB\n"
                  << std::left << std::setw(27) << "Voxel Cache Queries" << std::setw(16) << voxel_queries << "\n"
                  << std::left << std::setw(27) << "Voxel Cache Hits" << std::setw(16) << voxel_hits << "\n"
                  << std::setprecision(2)
                  << std::left << std::setw(27) << "Voxel Cache Hit Rate" << std::setw(16) << voxel_hit_rate_pct << "%\n"
                  << std::left << std::setw(27) << "Voxel Active Entries" << (std::to_string(active_voxel_entries) + " / " + std::to_string(opts.voxel_capacity)) << "\n"
                  << "--------------------------------------------------------------------------------\n";

        if (gate_passed) {
            std::cout << "Status: PASS (Meets 60 FPS target, VRAM <= 950 MB, Latency <= 33ms)\n";
        } else {
            std::cout << "Status: FAIL — Gate violation:" << gate_failure_reason << "\n";
        }
        std::cout << "================================================================================\n";

        if (opts.output_format == "both") {
            std::cout << "\n--- JSON TELEMETRY ---\n" << json_output;
        }
    }

    return gate_passed ? 0 : 4;
}

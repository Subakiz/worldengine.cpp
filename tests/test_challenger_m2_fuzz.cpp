#include "playworld/pwmf_format.h"
#include "playworld/tensor.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <sstream>
#include <filesystem>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>

using namespace playworld;

// ============================================================================
// Test Categories and Counters
// ============================================================================

enum class FuzzCategory {
    ByteTruncation,
    BitFlips,
    OOBAndIntegerOverflow,
    InvalidEnumsAndDims
};

const char* CategoryToString(FuzzCategory cat) {
    switch (cat) {
        case FuzzCategory::ByteTruncation: return "ByteTruncation";
        case FuzzCategory::BitFlips: return "BitFlips";
        case FuzzCategory::OOBAndIntegerOverflow: return "OOBAndIntegerOverflow";
        case FuzzCategory::InvalidEnumsAndDims: return "InvalidEnumsAndDims";
        default: return "Unknown";
    }
}

struct PermutationResult {
    size_t id{0};
    FuzzCategory category{FuzzCategory::ByteTruncation};
    std::string description;
    bool parse_memory_crashed{false};
    bool parse_memory_accepted{false}; // Unexpected acceptance (true)
    int32_t parse_memory_error{0};
    int parse_memory_signal{0};

    bool open_crashed{false};
    bool open_accepted{false}; // Unexpected acceptance (true)
    int32_t open_error{0};
    int open_signal{0};

    bool passed() const {
        return !parse_memory_crashed && !parse_memory_accepted &&
               !open_crashed && !open_accepted &&
               parse_memory_error != 0 && open_error != 0;
    }
};

// ============================================================================
// Base Model Generators
// ============================================================================

static std::vector<uint8_t> GenerateValidModelSingleTensor() {
    PWMFWriter writer;
    writer.SetMetadata("{\"model_name\": \"fuzz_valid_single\"}");
    std::vector<float> data(64, 1.0f);
    uint32_t shape[5] = {64, 0, 0, 0, 0};
    writer.AddTensor("weights.layer0", 1, shape, QuantType::FP32, data.data(), data.size() * sizeof(float));
    return writer.Serialize();
}

static std::vector<uint8_t> GenerateValidModelMultiTensor() {
    PWMFWriter writer;
    writer.SetMetadata("{\"model_name\": \"fuzz_valid_multi\"}");
    std::vector<float> fp32_data(32, 2.5f);
    uint32_t shape1[5] = {32, 0, 0, 0, 0};
    writer.AddTensor("layer.fp32", 1, shape1, QuantType::FP32, fp32_data.data(), fp32_data.size() * sizeof(float));

    std::vector<uint8_t> int4_data(32 * sizeof(INT4Block32), 0x33);
    uint32_t shape2[5] = {32, 32, 0, 0, 0};
    writer.AddTensor("layer.int4", 2, shape2, QuantType::INT4_BLOCK32, int4_data.data(), int4_data.size());

    std::vector<int8_t> int8_data(128, -7);
    uint32_t shape3[5] = {128, 0, 0, 0, 0};
    writer.AddTensor("layer.int8", 1, shape3, QuantType::INT8_SYMM, int8_data.data(), int8_data.size(), 0.1f);

    return writer.Serialize();
}

// ============================================================================
// Isolated Permutation Runner (Child Process Protection)
// ============================================================================

// Returns exit code:
// 0: Rejected gracefully (false with non-zero error)
// 101: Unexpected acceptance (true returned)
// 102: Returned Success (0) error code
static void ExecuteInChild(const uint8_t* data, size_t size, bool test_open, const std::string& temp_path) {
    PWMFParser parser;
    if (!test_open) {
        bool ok = parser.ParseMemory(data, size);
        if (ok) {
            _exit(101); // Unexpectedly accepted
        }
        if (parser.GetLastError() == PWMFError::Success) {
            _exit(102); // Returned Success on malformed input
        }
        // Encode error code: PWMFError is negative (-1 .. -11).
        // Pass positive integer 1..11 so parent can decode via exit code.
        int code = -static_cast<int32_t>(parser.GetLastError());
        if (code < 1 || code > 50) code = 99;
        _exit(code);
    } else {
        // Write to temp file for Open() test
        std::ofstream out(temp_path, std::ios::binary);
        if (data && size > 0) {
            out.write(reinterpret_cast<const char*>(data), size);
        }
        out.close();

        bool ok = parser.Open(temp_path);
        unlink(temp_path.c_str());

        if (ok) {
            _exit(101); // Unexpectedly accepted
        }
        if (parser.GetLastError() == PWMFError::Success) {
            _exit(102);
        }
        int code = -static_cast<int32_t>(parser.GetLastError());
        if (code < 1 || code > 50) code = 99;
        _exit(code);
    }
}

static void RunPermutation(size_t id, FuzzCategory cat, const std::string& desc,
                           const uint8_t* data, size_t size,
                           PermutationResult& result) {
    result.id = id;
    result.category = cat;
    result.description = desc;

    std::string temp_file_path = "/tmp/challenger_m2_fuzz_" + std::to_string(getpid()) + "_" + std::to_string(id) + ".pwmf";

    // 1. Test ParseMemory in child process
    {
        pid_t pid = fork();
        if (pid == 0) {
            ExecuteInChild(data, size, false, temp_file_path);
        }
        int status = 0;
        waitpid(pid, &status, 0);

        if (WIFSIGNALED(status)) {
            result.parse_memory_crashed = true;
            result.parse_memory_signal = WTERMSIG(status);
        } else if (WIFEXITED(status)) {
            int exit_code = WEXITSTATUS(status);
            if (exit_code == 101) {
                result.parse_memory_accepted = true;
                result.parse_memory_error = 0;
            } else if (exit_code == 102) {
                result.parse_memory_error = 0;
            } else {
                result.parse_memory_error = -exit_code;
            }
        }
    }

    // 2. Test Open in child process
    {
        pid_t pid = fork();
        if (pid == 0) {
            ExecuteInChild(data, size, true, temp_file_path);
        }
        int status = 0;
        waitpid(pid, &status, 0);

        if (WIFSIGNALED(status)) {
            result.open_crashed = true;
            result.open_signal = WTERMSIG(status);
        } else if (WIFEXITED(status)) {
            int exit_code = WEXITSTATUS(status);
            if (exit_code == 101) {
                result.open_accepted = true;
                result.open_error = 0;
            } else if (exit_code == 102) {
                result.open_error = 0;
            } else {
                result.open_error = -exit_code;
            }
        }
    }
}

// ============================================================================
// Main Adversarial Fuzzing Runner
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "================================================================================\n";
    std::cout << "  WORLDENGINE.CPP (PLAYWORLD) — ADVERSARIAL MODEL FUZZING CHALLENGE (M2)\n";
    std::cout << "  EMPIRICAL CHALLENGER: teamwork_preview_challenger_m2_1\n";
    std::cout << "================================================================================\n";

    std::mt19937_64 rng(0xCAFEF00D12345678ULL);

    const std::vector<uint8_t> valid_single = GenerateValidModelSingleTensor();
    const std::vector<uint8_t> valid_multi = GenerateValidModelMultiTensor();

    std::cout << "[INFO] Valid Single Model Size: " << valid_single.size() << " bytes\n";
    std::cout << "[INFO] Valid Multi Model Size:  " << valid_multi.size() << " bytes\n";

    std::vector<PermutationResult> all_results;
    size_t perm_id = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    // ========================================================================
    // Category 1: Random Byte Truncations (Lengths 1 to 500)
    // Target: At least 250 permutations
    // ========================================================================
    std::cout << "\n[RUN] Executing Category 1: Random Byte Truncations (lengths 1 to 500)...\n";
    {
        // Explicit boundary lengths
        const size_t explicit_lengths[] = {
            0, 1, 2, 4, 8, 16, 32, 48, 64, 80, 95, 96, 97,
            112, 127, 128, 129, 200, 256, 300, 384, 400, 450, 499, 500
        };
        for (size_t len : explicit_lengths) {
            const auto& base = (len % 2 == 0) ? valid_single : valid_multi;
            size_t actual_len = std::min(len, base.size());
            std::vector<uint8_t> buf(base.begin(), base.begin() + actual_len);

            PermutationResult res;
            RunPermutation(++perm_id, FuzzCategory::ByteTruncation,
                           "Explicit truncation to " + std::to_string(actual_len) + " bytes",
                           buf.data(), buf.size(), res);
            all_results.push_back(res);
        }

        // Random lengths in [1, 500]
        std::uniform_int_distribution<size_t> dist_1_500(1, 500);
        for (size_t i = 0; i < 235; ++i) {
            size_t target_len = dist_1_500(rng);
            const auto& base = (i % 2 == 0) ? valid_single : valid_multi;
            size_t actual_len = std::min(target_len, base.size() - 1);
            std::vector<uint8_t> buf(base.begin(), base.begin() + actual_len);

            PermutationResult res;
            RunPermutation(++perm_id, FuzzCategory::ByteTruncation,
                           "Random truncation to " + std::to_string(actual_len) + " bytes",
                           buf.data(), buf.size(), res);
            all_results.push_back(res);
        }
    }
    std::cout << "  -> Category 1 completed: " << perm_id << " permutations.\n";

    // ========================================================================
    // Category 2: Random Bit Flips in Valid Models
    // Target: At least 350 permutations
    // ========================================================================
    std::cout << "\n[RUN] Executing Category 2: Random Bit Flips in Valid .PWMF Models...\n";
    {
        size_t cat2_start = perm_id;

        // Subcategory A: Single Bit Flips in Header (bytes 0..95) - 100 permutations
        for (size_t i = 0; i < 100; ++i) {
            auto buf = (i % 2 == 0) ? valid_single : valid_multi;
            std::uniform_int_distribution<size_t> header_byte_dist(0, sizeof(PWMFHeader) - 1);
            std::uniform_int_distribution<int> bit_dist(0, 7);

            size_t byte_idx = header_byte_dist(rng);
            int bit_idx = bit_dist(rng);
            buf[byte_idx] ^= static_cast<uint8_t>(1 << bit_idx);

            PermutationResult res;
            RunPermutation(++perm_id, FuzzCategory::BitFlips,
                           "Header single bit flip at byte " + std::to_string(byte_idx) + " bit " + std::to_string(bit_idx),
                           buf.data(), buf.size(), res);
            all_results.push_back(res);
        }

        // Subcategory B: Single & Multi Bit Flips in Tensor Table - 130 permutations
        for (size_t i = 0; i < 130; ++i) {
            auto buf = (i % 2 == 0) ? valid_single : valid_multi;
            const PWMFHeader* hdr = reinterpret_cast<const PWMFHeader*>(buf.data());
            size_t table_start = hdr->tensor_table_offset;
            size_t table_end = table_start + hdr->tensor_table_length;
            if (table_end > buf.size()) table_end = buf.size();

            std::uniform_int_distribution<size_t> table_dist(table_start, table_end - 1);
            std::uniform_int_distribution<int> bit_dist(0, 7);

            // Flip 1 to 4 bits in table
            int num_flips = 1 + (i % 4);
            for (int f = 0; f < num_flips; ++f) {
                size_t byte_idx = table_dist(rng);
                int bit_idx = bit_dist(rng);
                buf[byte_idx] ^= static_cast<uint8_t>(1 << bit_idx);
            }

            PermutationResult res;
            RunPermutation(++perm_id, FuzzCategory::BitFlips,
                           "Tensor table " + std::to_string(num_flips) + " bit flips",
                           buf.data(), buf.size(), res);
            all_results.push_back(res);
        }

        // Subcategory C: Payload Bit Flips - 120 permutations
        for (size_t i = 0; i < 120; ++i) {
            auto buf = (i % 2 == 0) ? valid_single : valid_multi;
            const PWMFHeader* hdr = reinterpret_cast<const PWMFHeader*>(buf.data());
            size_t payload_start = hdr->weight_data_offset;
            size_t payload_end = payload_start + hdr->weight_data_length;
            if (payload_end > buf.size()) payload_end = buf.size();

            std::uniform_int_distribution<size_t> payload_dist(payload_start, payload_end - 1);
            std::uniform_int_distribution<int> bit_dist(0, 7);

            int num_flips = 1 + (i % 5);
            for (int f = 0; f < num_flips; ++f) {
                size_t byte_idx = payload_dist(rng);
                int bit_idx = bit_dist(rng);
                buf[byte_idx] ^= static_cast<uint8_t>(1 << bit_idx);
            }

            PermutationResult res;
            RunPermutation(++perm_id, FuzzCategory::BitFlips,
                           "Payload " + std::to_string(num_flips) + " bit flips",
                           buf.data(), buf.size(), res);
            all_results.push_back(res);
        }
        std::cout << "  -> Category 2 completed: " << (perm_id - cat2_start) << " permutations.\n";
    }

    // ========================================================================
    // Category 3: Out-of-bounds Offsets & Integer Overflow Attempts
    // Target: At least 300 permutations
    // ========================================================================
    std::cout << "\n[RUN] Executing Category 3: Out-of-bounds Offsets and Integer Overflow Attempts...\n";
    {
        size_t cat3_start = perm_id;

        // Subcategory A: Explicit Boundary & Integer Overflow Extremes - 60 permutations
        const uint64_t extreme_offsets[] = {
            UINT64_MAX,
            UINT64_MAX - 1,
            UINT64_MAX - 63,
            UINT64_MAX - 111,
            UINT64_MAX - 112,
            UINT64_MAX - 500,
            0x8000000000000000ULL,
            0xFFFFFFFF00000000ULL,
            0x0000000100000000ULL,
            UINT32_MAX,
            static_cast<uint64_t>(valid_single.size()) + 1,
            static_cast<uint64_t>(valid_single.size()) + 1000,
            static_cast<uint64_t>(valid_single.size()) * 10ULL
        };

        for (uint64_t off : extreme_offsets) {
            // Corrupt metadata_offset
            {
                auto buf = valid_single;
                PWMFHeader* hdr = reinterpret_cast<PWMFHeader*>(buf.data());
                hdr->metadata_offset = off;
                PermutationResult res;
                RunPermutation(++perm_id, FuzzCategory::OOBAndIntegerOverflow,
                               "metadata_offset = 0x" + [off]() { std::stringstream ss; ss << std::hex << off; return ss.str(); }(),
                               buf.data(), buf.size(), res);
                all_results.push_back(res);
            }
            // Corrupt metadata_length
            {
                auto buf = valid_single;
                PWMFHeader* hdr = reinterpret_cast<PWMFHeader*>(buf.data());
                hdr->metadata_length = off;
                PermutationResult res;
                RunPermutation(++perm_id, FuzzCategory::OOBAndIntegerOverflow,
                               "metadata_length = 0x" + [off]() { std::stringstream ss; ss << std::hex << off; return ss.str(); }(),
                               buf.data(), buf.size(), res);
                all_results.push_back(res);
            }
            // Corrupt tensor_table_offset
            {
                auto buf = valid_single;
                PWMFHeader* hdr = reinterpret_cast<PWMFHeader*>(buf.data());
                hdr->tensor_table_offset = off;
                PermutationResult res;
                RunPermutation(++perm_id, FuzzCategory::OOBAndIntegerOverflow,
                               "tensor_table_offset = 0x" + [off]() { std::stringstream ss; ss << std::hex << off; return ss.str(); }(),
                               buf.data(), buf.size(), res);
                all_results.push_back(res);
            }
            // Corrupt weight_data_offset (aligned and unaligned)
            {
                auto buf = valid_single;
                PWMFHeader* hdr = reinterpret_cast<PWMFHeader*>(buf.data());
                hdr->weight_data_offset = (off & ~63ULL); // aligned
                PermutationResult res;
                RunPermutation(++perm_id, FuzzCategory::OOBAndIntegerOverflow,
                               "weight_data_offset = 0x" + [off]() { std::stringstream ss; ss << std::hex << (off & ~63ULL); return ss.str(); }(),
                               buf.data(), buf.size(), res);
                all_results.push_back(res);
            }
        }

        // Subcategory B: Targeted Wrapping Integer Overflows (offset + length < size mod 2^64) - 80 permutations
        for (size_t i = 0; i < 80; ++i) {
            auto buf = valid_single;
            PWMFHeader* hdr = reinterpret_cast<PWMFHeader*>(buf.data());

            // Deliberately choose offsets near UINT64_MAX such that offset + length wraps to < buf.size()
            uint64_t table_len = hdr->tensor_table_length;
            uint64_t wrap_delta = i % (table_len + 10);
            hdr->tensor_table_offset = UINT64_MAX - table_len + wrap_delta + 1;

            PermutationResult res;
            RunPermutation(++perm_id, FuzzCategory::OOBAndIntegerOverflow,
                           "Wrapping tensor_table_offset (sum wraps to " + std::to_string(wrap_delta) + ")",
                           buf.data(), buf.size(), res);
            all_results.push_back(res);
        }

        // Subcategory C: Tensor Descriptor data_offset & data_bytes wrapping & OOB - 80 permutations
        for (size_t i = 0; i < 80; ++i) {
            auto buf = valid_single;
            PWMFHeader* hdr = reinterpret_cast<PWMFHeader*>(buf.data());
            PWMFTensorDescriptor* desc = reinterpret_cast<PWMFTensorDescriptor*>(buf.data() + hdr->tensor_table_offset);

            if (i % 2 == 0) {
                // Out-of-bounds offset
                desc->data_offset = hdr->weight_data_length + 100 + i * 1000;
            } else {
                // Wrapping offset: data_offset + data_bytes wraps mod 2^64
                desc->data_offset = UINT64_MAX - desc->data_bytes + 1 + (i % 30);
            }

            PermutationResult res;
            RunPermutation(++perm_id, FuzzCategory::OOBAndIntegerOverflow,
                           "Descriptor data_offset integer overflow / OOB test #" + std::to_string(i),
                           buf.data(), buf.size(), res);
            all_results.push_back(res);
        }

        // Subcategory D: num_tensors overflows and table length mismatches - 88 permutations
        for (size_t i = 0; i < 88; ++i) {
            auto buf = valid_single;
            PWMFHeader* hdr = reinterpret_cast<PWMFHeader*>(buf.data());
            uint32_t fake_count = (i == 0) ? UINT32_MAX : (i == 1) ? 0x7FFFFFFF : (1000000 + i * 10000);
            hdr->num_tensors = fake_count;
            if (i % 2 == 0) {
                hdr->tensor_table_length = static_cast<uint64_t>(fake_count) * sizeof(PWMFTensorDescriptor);
            }

            PermutationResult res;
            RunPermutation(++perm_id, FuzzCategory::OOBAndIntegerOverflow,
                           "num_tensors = " + std::to_string(fake_count),
                           buf.data(), buf.size(), res);
            all_results.push_back(res);
        }

        std::cout << "  -> Category 3 completed: " << (perm_id - cat3_start) << " permutations.\n";
    }

    // ========================================================================
    // Category 4: Random Invalid Enum Values in QuantType and ndims
    // Target: At least 250 permutations
    // ========================================================================
    std::cout << "\n[RUN] Executing Category 4: Random Invalid Enum Values in QuantType and ndims...\n";
    {
        size_t cat4_start = perm_id;

        // Subcategory A: Invalid QuantType Enums (> 7) - 100 permutations
        for (size_t i = 0; i < 100; ++i) {
            auto buf = valid_single;
            PWMFHeader* hdr = reinterpret_cast<PWMFHeader*>(buf.data());
            PWMFTensorDescriptor* desc = reinterpret_cast<PWMFTensorDescriptor*>(buf.data() + hdr->tensor_table_offset);

            uint8_t bad_qt = static_cast<uint8_t>(8 + (i % 248)); // 8 .. 255
            desc->quant_type = static_cast<QuantType>(bad_qt);

            PermutationResult res;
            RunPermutation(++perm_id, FuzzCategory::InvalidEnumsAndDims,
                           "Invalid QuantType enum = " + std::to_string(static_cast<int>(bad_qt)),
                           buf.data(), buf.size(), res);
            all_results.push_back(res);
        }

        // Subcategory B: Invalid ndims (< 1 or > 5) - 100 permutations
        for (size_t i = 0; i < 100; ++i) {
            auto buf = valid_single;
            PWMFHeader* hdr = reinterpret_cast<PWMFHeader*>(buf.data());
            PWMFTensorDescriptor* desc = reinterpret_cast<PWMFTensorDescriptor*>(buf.data() + hdr->tensor_table_offset);

            uint32_t bad_ndims;
            if (i == 0) bad_ndims = 0;
            else if (i == 1) bad_ndims = 6;
            else if (i == 2) bad_ndims = UINT32_MAX;
            else bad_ndims = 7 + (i * 37) % 100000;

            desc->ndims = bad_ndims;

            PermutationResult res;
            RunPermutation(++perm_id, FuzzCategory::InvalidEnumsAndDims,
                           "Invalid ndims = " + std::to_string(bad_ndims),
                           buf.data(), buf.size(), res);
            all_results.push_back(res);
        }

        // Subcategory C: Combined invalid QuantType + ndims + corrupted shapes - 50 permutations
        for (size_t i = 0; i < 50; ++i) {
            auto buf = valid_multi;
            PWMFHeader* hdr = reinterpret_cast<PWMFHeader*>(buf.data());
            // Target first or second tensor
            size_t tensor_idx = i % hdr->num_tensors;
            PWMFTensorDescriptor* desc = reinterpret_cast<PWMFTensorDescriptor*>(
                buf.data() + hdr->tensor_table_offset + tensor_idx * sizeof(PWMFTensorDescriptor));

            desc->quant_type = static_cast<QuantType>(128 + i);
            desc->ndims = 10 + i;
            desc->shape[0] = 0; // zero dimension

            PermutationResult res;
            RunPermutation(++perm_id, FuzzCategory::InvalidEnumsAndDims,
                           "Combined invalid quant (" + std::to_string(128 + i) + ") and ndims (" + std::to_string(10 + i) + ")",
                           buf.data(), buf.size(), res);
            all_results.push_back(res);
        }

        std::cout << "  -> Category 4 completed: " << (perm_id - cat4_start) << " permutations.\n";
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // ========================================================================
    // Analysis and Statistics
    // ========================================================================
    size_t total_tests = all_results.size();
    size_t total_passed = 0;
    size_t total_crashed = 0;
    size_t total_accepted = 0;
    size_t total_invalid_error = 0;

    size_t cat_tests[4]{0};
    size_t cat_passed[4]{0};
    size_t cat_crashed[4]{0};
    size_t cat_accepted[4]{0};

    std::vector<PermutationResult> failing_cases;

    for (const auto& res : all_results) {
        size_t c_idx = static_cast<size_t>(res.category);
        cat_tests[c_idx]++;

        bool crashed = res.parse_memory_crashed || res.open_crashed;
        bool accepted = res.parse_memory_accepted || res.open_accepted;
        bool bad_err = (res.parse_memory_error == 0) || (res.open_error == 0);

        if (crashed) {
            total_crashed++;
            cat_crashed[c_idx]++;
            failing_cases.push_back(res);
        } else if (accepted) {
            total_accepted++;
            cat_accepted[c_idx]++;
            failing_cases.push_back(res);
        } else if (bad_err) {
            total_invalid_error++;
            failing_cases.push_back(res);
        } else {
            total_passed++;
            cat_passed[c_idx]++;
        }
    }

    std::cout << "\n================================================================================\n";
    std::cout << "                         ADVERSARIAL FUZZING AUDIT RESULTS                      \n";
    std::cout << "================================================================================\n";
    std::cout << "Total Permutations Generated:  " << total_tests << "\n";
    std::cout << "Total Gracefully Rejected:     " << total_passed << " ("
              << std::fixed << std::setprecision(2) << (100.0 * total_passed / total_tests) << "%)\n";
    std::cout << "Total Crashes / Faults:        " << total_crashed << "\n";
    std::cout << "Total False Acceptances:       " << total_accepted << "\n";
    std::cout << "Total Invalid Error Codes:     " << total_invalid_error << "\n";
    std::cout << "Total Execution Duration:      " << total_ms << " ms ("
              << (total_ms / 1000.0) << " s)\n";
    std::cout << "================================================================================\n";

    std::cout << "\nBreakdown by Category:\n";
    for (size_t c = 0; c < 4; ++c) {
        FuzzCategory cat = static_cast<FuzzCategory>(c);
        std::cout << "  - " << std::left << std::setw(24) << CategoryToString(cat)
                  << " Total: " << std::setw(5) << cat_tests[c]
                  << " Passed: " << std::setw(5) << cat_passed[c]
                  << " Crashed: " << std::setw(3) << cat_crashed[c]
                  << " Accepted: " << std::setw(3) << cat_accepted[c]
                  << "\n";
    }

    if (!failing_cases.empty()) {
        std::cout << "\n================================================================================\n";
        std::cout << "                         FAILURES & VULNERABILITIES IDENTIFIED                 \n";
        std::cout << "================================================================================\n";
        std::cout << "Displaying first " << std::min(failing_cases.size(), size_t(10)) << " failures:\n";
        for (size_t i = 0; i < std::min(failing_cases.size(), size_t(10)); ++i) {
            const auto& f = failing_cases[i];
            std::cout << "  #" << f.id << " [" << CategoryToString(f.category) << "] " << f.description << "\n";
            if (f.parse_memory_crashed) {
                std::cout << "     -> ParseMemory CRASHED with signal " << f.parse_memory_signal
                          << " (" << (f.parse_memory_signal == 11 ? "SIGSEGV" : f.parse_memory_signal == 10 ? "SIGBUS" : "SIGNAL") << ")\n";
            }
            if (f.open_crashed) {
                std::cout << "     -> Open CRASHED with signal " << f.open_signal
                          << " (" << (f.open_signal == 11 ? "SIGSEGV" : f.open_signal == 10 ? "SIGBUS" : "SIGNAL") << ")\n";
            }
            if (f.parse_memory_accepted) {
                std::cout << "     -> ParseMemory FALSE ACCEPTANCE: returned true, error=" << f.parse_memory_error << "\n";
            }
            if (f.open_accepted) {
                std::cout << "     -> Open FALSE ACCEPTANCE: returned true, error=" << f.open_error << "\n";
            }
        }
    }

    std::cout << "\n================================================================================\n";
    if (total_crashed == 0 && total_accepted == 0 && total_invalid_error == 0 && total_passed == total_tests) {
        std::cout << "  FINAL VERDICT: CONFIRMED\n";
        std::cout << "  100% of " << total_tests << " adversarial malformed inputs were gracefully rejected.\n";
        std::cout << "  Zero crashes, zero aborts, zero memory faults detected.\n";
        std::cout << "================================================================================\n";
        return 0;
    } else {
        std::cout << "  FINAL VERDICT: FAILED\n";
        std::cout << "  Parser resilience challenge FAILED: " << failing_cases.size() << " failures detected!\n";
        std::cout << "  - Crashes: " << total_crashed << "\n";
        std::cout << "  - False Acceptances: " << total_accepted << "\n";
        std::cout << "  - Invalid Error Codes: " << total_invalid_error << "\n";
        std::cout << "================================================================================\n";
        return 1;
    }
}

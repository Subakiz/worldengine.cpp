#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace playworld::test {

// ANSI Color Escape Sequences
inline constexpr const char* COLOR_RESET   = "\033[0m";
inline constexpr const char* COLOR_RED     = "\033[31m";
inline constexpr const char* COLOR_GREEN   = "\033[32m";
inline constexpr const char* COLOR_YELLOW  = "\033[33m";
inline constexpr const char* COLOR_BLUE    = "\033[34m";
inline constexpr const char* COLOR_MAGENTA = "\033[35m";
inline constexpr const char* COLOR_CYAN    = "\033[36m";
inline constexpr const char* COLOR_BOLD    = "\033[1m";

class AssertionException : public std::exception {
public:
    explicit AssertionException(std::string message) : msg_(std::move(message)) {}
    [[nodiscard]] const char* what() const noexcept override { return msg_.c_str(); }
private:
    std::string msg_;
};

struct FailureInfo {
    std::string file;
    int line;
    std::string expression;
    std::string message;
    bool is_fatal;
};

struct TestResult {
    std::string suite_name;
    std::string test_name;
    bool passed{true};
    uint64_t duration_us{0};
    std::vector<FailureInfo> failures;
};

class TestCase {
public:
    virtual ~TestCase() = default;
    virtual void Run() = 0;
};

class TestRegistry {
public:
    using TestFactory = std::function<std::unique_ptr<TestCase>()>;

    struct TestEntry {
        std::string suite_name;
        std::string test_name;
        TestFactory factory;
    };

    static TestRegistry& Instance() {
        static TestRegistry instance;
        return instance;
    }

    void RegisterTest(std::string suite, std::string name, TestFactory factory) {
        tests_.push_back(TestEntry{std::move(suite), std::move(name), std::move(factory)});
    }

    void AddFailure(std::string file, int line, std::string expr, std::string msg, bool fatal) {
        if (current_result_) {
            current_result_->passed = false;
            current_result_->failures.push_back(FailureInfo{
                std::move(file), line, std::move(expr), std::move(msg), fatal
            });
        }
        if (fatal) {
            throw AssertionException(msg);
        }
    }

    int RunAll(int argc = 0, char** argv = nullptr) {
        std::string filter;
        std::string json_path;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if ((arg == "--filter" || arg == "-f") && i + 1 < argc) {
                filter = argv[++i];
            } else if (arg.rfind("--filter=", 0) == 0) {
                filter = arg.substr(9);
            } else if ((arg == "--json" || arg == "-j") && i + 1 < argc) {
                json_path = argv[++i];
            } else if (arg.rfind("--json=", 0) == 0) {
                json_path = arg.substr(7);
            }
        }

        std::vector<TestEntry> to_run;
        for (const auto& entry : tests_) {
            std::string full_name = entry.suite_name + "." + entry.test_name;
            if (filter.empty() || full_name.find(filter) != std::string::npos) {
                to_run.push_back(entry);
            }
        }

        std::cout << COLOR_GREEN << "[==========] " << COLOR_RESET
                  << "Running " << to_run.size() << " tests from " << tests_.size() << " registered.\n";

        auto suite_start = std::chrono::high_resolution_clock::now();
        std::vector<TestResult> results;
        int passed_count = 0;
        int failed_count = 0;

        for (const auto& entry : to_run) {
            std::string full_name = entry.suite_name + "." + entry.test_name;
            std::cout << COLOR_GREEN << "[ RUN      ] " << COLOR_RESET << full_name << "\n";

            TestResult result;
            result.suite_name = entry.suite_name;
            result.test_name = entry.test_name;
            result.passed = true;
            current_result_ = &result;

            auto t0 = std::chrono::high_resolution_clock::now();
            try {
                auto test_instance = entry.factory();
                test_instance->Run();
            } catch (const AssertionException& e) {
                result.passed = false;
            } catch (const std::exception& e) {
                result.passed = false;
                result.failures.push_back(FailureInfo{
                    __FILE__, __LINE__, "Unhandled Exception", e.what(), true
                });
            } catch (...) {
                result.passed = false;
                result.failures.push_back(FailureInfo{
                    __FILE__, __LINE__, "Unknown Exception", "Caught non-std::exception", true
                });
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            result.duration_us = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
            );

            current_result_ = nullptr;
            results.push_back(result);

            if (result.passed) {
                passed_count++;
                std::cout << COLOR_GREEN << "[       OK ] " << COLOR_RESET << full_name
                          << " (" << (result.duration_us / 1000.0) << " ms, "
                          << result.duration_us << " us)\n";
            } else {
                failed_count++;
                std::cout << COLOR_RED << "[  FAILED  ] " << COLOR_RESET << full_name
                          << " (" << (result.duration_us / 1000.0) << " ms)\n";
                for (const auto& fail : result.failures) {
                    std::cout << "  " << COLOR_BOLD << fail.file << ":" << fail.line << COLOR_RESET
                              << ": Failure in `" << fail.expression << "`\n"
                              << "    " << fail.message << "\n";
                }
            }
        }

        auto suite_end = std::chrono::high_resolution_clock::now();
        uint64_t total_duration_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(suite_end - suite_start).count()
        );

        std::cout << COLOR_GREEN << "[==========] " << COLOR_RESET
                  << to_run.size() << " tests ran. ("
                  << (total_duration_us / 1000.0) << " ms total)\n";
        std::cout << COLOR_GREEN << "[  PASSED  ] " << COLOR_RESET << passed_count << " tests.\n";
        if (failed_count > 0) {
            std::cout << COLOR_RED << "[  FAILED  ] " << COLOR_RESET << failed_count << " tests.\n";
            for (const auto& res : results) {
                if (!res.passed) {
                    std::cout << COLOR_RED << "[  FAILED  ] " << COLOR_RESET
                              << res.suite_name << "." << res.test_name << "\n";
                }
            }
        }

        if (!json_path.empty()) {
            WriteJson(json_path, results, passed_count, failed_count, total_duration_us);
        }

        return (failed_count == 0) ? 0 : 1;
    }

    const std::vector<TestEntry>& GetTests() const { return tests_; }

private:
    TestRegistry() = default;
    std::vector<TestEntry> tests_;
    TestResult* current_result_{nullptr};

    static void WriteJson(const std::string& path,
                          const std::vector<TestResult>& results,
                          int passed, int failed, uint64_t duration_us) {
        std::ofstream ofs(path);
        if (!ofs.is_open()) return;

        ofs << "{\n";
        ofs << "  \"total\": " << results.size() << ",\n";
        ofs << "  \"passed\": " << passed << ",\n";
        ofs << "  \"failed\": " << failed << ",\n";
        ofs << "  \"duration_us\": " << duration_us << ",\n";
        ofs << "  \"tests\": [\n";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            ofs << "    {\n";
            ofs << "      \"suite\": \"" << r.suite_name << "\",\n";
            ofs << "      \"name\": \"" << r.test_name << "\",\n";
            ofs << "      \"status\": \"" << (r.passed ? "PASSED" : "FAILED") << "\",\n";
            ofs << "      \"duration_us\": " << r.duration_us << "\n";
            ofs << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
        }
        ofs << "  ]\n";
        ofs << "}\n";
    }
};

template <typename T>
struct TestRegistrar {
    TestRegistrar(std::string suite, std::string name) {
        TestRegistry::Instance().RegisterTest(
            std::move(suite), std::move(name),
            []() -> std::unique_ptr<TestCase> { return std::make_unique<T>(); }
        );
    }
};

inline void RecordFailure(const char* file, int line, const char* expr, const std::string& msg, bool fatal) {
    TestRegistry::Instance().AddFailure(file, line, expr, msg, fatal);
}

template <typename T>
inline void PrintValue(std::ostream& os, const T& val) {
    if constexpr (std::is_enum_v<T>) {
        os << static_cast<std::underlying_type_t<T>>(val);
    } else if constexpr (requires { os << val; }) {
        os << val;
    } else {
        os << "<custom-type>";
    }
}

template <typename A, typename B>
inline bool ValuesEqual(const A& a, const B& b) {
    return a == b;
}

inline bool ValuesNear(double a, double b, double eps) {
    return std::fabs(a - b) <= eps;
}

} // namespace playworld::test

// Test Definition Macro
#define TEST(SuiteName, TestName) \
    class SuiteName##_##TestName##_Test : public ::playworld::test::TestCase { \
    public: \
        void Run() override; \
    }; \
    static ::playworld::test::TestRegistrar<SuiteName##_##TestName##_Test> \
        g_registrar_##SuiteName##_##TestName(#SuiteName, #TestName); \
    void SuiteName##_##TestName##_Test::Run()

// Also alias TEST_CASE for Catch2-style familiarity if needed
#define TEST_CASE(TestName) TEST(DefaultSuite, TestName)

// Assertion Helpers
#define PW_CHECK_BOOL(expr, expected, fatal) \
    do { \
        bool _pw_val = static_cast<bool>(expr); \
        if (_pw_val != (expected)) { \
            std::ostringstream _pw_oss; \
            _pw_oss << "Expected " << #expr << " to be " << ((expected) ? "true" : "false") \
                    << ", but was " << (_pw_val ? "true" : "false"); \
            ::playworld::test::RecordFailure(__FILE__, __LINE__, #expr, _pw_oss.str(), (fatal)); \
        } \
    } while (0)

#define PW_CHECK_BINARY(a, b, op, fatal) \
    do { \
        const auto& _pw_a = (a); \
        const auto& _pw_b = (b); \
        if (!(_pw_a op _pw_b)) { \
            std::ostringstream _pw_oss; \
            _pw_oss << "Comparison (" << #a << " " << #op << " " << #b << ") failed: ["; \
            ::playworld::test::PrintValue(_pw_oss, _pw_a); \
            _pw_oss << "] vs ["; \
            ::playworld::test::PrintValue(_pw_oss, _pw_b); \
            _pw_oss << "]"; \
            ::playworld::test::RecordFailure(__FILE__, __LINE__, #a " " #op " " #b, _pw_oss.str(), (fatal)); \
        } \
    } while (0)

#define PW_CHECK_NEAR(a, b, eps, fatal) \
    do { \
        double _pw_a = static_cast<double>(a); \
        double _pw_b = static_cast<double>(b); \
        double _pw_eps = static_cast<double>(eps); \
        double _pw_diff = std::fabs(_pw_a - _pw_b); \
        if (_pw_diff > _pw_eps) { \
            std::ostringstream _pw_oss; \
            _pw_oss << "Difference between (" << #a << ")=" << _pw_a \
                    << " and (" << #b << ")=" << _pw_b << " is " << _pw_diff \
                    << ", which exceeds epsilon " << #eps << "=" << _pw_eps; \
            ::playworld::test::RecordFailure(__FILE__, __LINE__, "NEAR(" #a ", " #b ")", _pw_oss.str(), (fatal)); \
        } \
    } while (0)

// Fatal Assertions (ASSERT_*)
#define ASSERT_TRUE(expr)              PW_CHECK_BOOL(expr, true, true)
#define ASSERT_FALSE(expr)             PW_CHECK_BOOL(expr, false, true)
#define ASSERT_EQ(a, b)                PW_CHECK_BINARY(a, b, ==, true)
#define ASSERT_NE(a, b)                PW_CHECK_BINARY(a, b, !=, true)
#define ASSERT_LT(a, b)                PW_CHECK_BINARY(a, b, <, true)
#define ASSERT_LE(a, b)                PW_CHECK_BINARY(a, b, <=, true)
#define ASSERT_GT(a, b)                PW_CHECK_BINARY(a, b, >, true)
#define ASSERT_GE(a, b)                PW_CHECK_BINARY(a, b, >=, true)
#define ASSERT_NEAR(a, b, eps)         PW_CHECK_NEAR(a, b, eps, true)

#define ASSERT_THROWS(statement, ExceptionType) \
    do { \
        bool _pw_threw = false; \
        try { \
            statement; \
        } catch (const ExceptionType&) { \
            _pw_threw = true; \
        } catch (...) { \
            ::playworld::test::RecordFailure(__FILE__, __LINE__, #statement, \
                "Expected exception " #ExceptionType ", but different exception thrown", true); \
        } \
        if (!_pw_threw) { \
            ::playworld::test::RecordFailure(__FILE__, __LINE__, #statement, \
                "Expected exception " #ExceptionType ", but no exception thrown", true); \
        } \
    } while (0)

#define ASSERT_NO_THROW(statement) \
    do { \
        try { \
            statement; \
        } catch (const std::exception& _pw_e) { \
            ::playworld::test::RecordFailure(__FILE__, __LINE__, #statement, \
                std::string("Unexpected exception thrown: ") + _pw_e.what(), true); \
        } catch (...) { \
            ::playworld::test::RecordFailure(__FILE__, __LINE__, #statement, \
                "Unexpected unknown exception thrown", true); \
        } \
    } while (0)

// Non-fatal Expectations (EXPECT_*)
#define EXPECT_TRUE(expr)              PW_CHECK_BOOL(expr, true, false)
#define EXPECT_FALSE(expr)             PW_CHECK_BOOL(expr, false, false)
#define EXPECT_EQ(a, b)                PW_CHECK_BINARY(a, b, ==, false)
#define EXPECT_NE(a, b)                PW_CHECK_BINARY(a, b, !=, false)
#define EXPECT_LT(a, b)                PW_CHECK_BINARY(a, b, <, false)
#define EXPECT_LE(a, b)                PW_CHECK_BINARY(a, b, <=, false)
#define EXPECT_GT(a, b)                PW_CHECK_BINARY(a, b, >, false)
#define EXPECT_GE(a, b)                PW_CHECK_BINARY(a, b, >=, false)
#define EXPECT_LE(a, b)                PW_CHECK_BINARY(a, b, <=, false)
#define EXPECT_NEAR(a, b, eps)         PW_CHECK_NEAR(a, b, eps, false)

// Test Runner Driver Macro
#define TEST_RUNNER_MAIN() \
    int main(int argc, char** argv) { \
        return ::playworld::test::TestRegistry::Instance().RunAll(argc, argv); \
    }

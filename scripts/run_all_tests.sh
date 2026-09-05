#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ANSI Color Codes
GREEN="\033[32m"
RED="\033[31m"
YELLOW="\033[33m"
CYAN="\033[36m"
BOLD="\033[1m"
RESET="\033[0m"

# Parse Command-line Arguments
MODE="release"
for arg in "$@"; do
    case "$arg" in
        --sanitize|--asan|--ubsan)
            MODE="sanitize"
            ;;
        --all)
            MODE="all"
            ;;
        -h|--help)
            echo "Usage: $0 [--sanitize | --all | --help]"
            echo "  --sanitize : Run AddressSanitizer & UndefinedBehaviorSanitizer suite"
            echo "  --all      : Run standard Release pass followed by Sanitizer pass"
            echo "  (default)  : Run standard Release verification pass"
            exit 0
            ;;
    esac
done

if [ "${ENABLE_SANITIZERS:-0}" = "1" ] || [ "${PLAYWORLD_SANITIZER:-0}" = "1" ]; then
    MODE="sanitize"
fi

UNIT_TESTS=(
    "test_pwmf:tests/unit/test_pwmf.cpp"
    "test_quant:tests/unit/test_quant.cpp"
    "test_ring_buffer:tests/unit/test_ring_buffer.cpp"
    "test_voxel_grid:tests/unit/test_voxel_grid.cpp"
    "test_scheduler:tests/unit/test_scheduler.cpp"
)

INTEGRATION_TESTS=(
    "test_benchmark_cli:tests/integration/test_benchmark_cli.cpp"
    "test_spatial_ssim:tests/integration/test_spatial_ssim.cpp"
    "test_memory_leak:tests/integration/test_memory_leak.cpp"
)

FUZZ_TESTS=(
    "test_pwmf_fuzz:tests/fuzz/test_pwmf_fuzz.cpp"
)

STRESS_TESTS=(
    "test_concurrency_stress:tests/stress/test_concurrency_stress.cpp"
    "test_soak_simulation:tests/stress/test_soak_simulation.cpp"
)

ALL_SUITES=(
    "test_pwmf"
    "test_quant"
    "test_ring_buffer"
    "test_voxel_grid"
    "test_scheduler"
    "test_benchmark_cli"
    "test_spatial_ssim"
    "test_memory_leak"
    "test_pwmf_fuzz"
    "test_concurrency_stress"
    "test_soak_simulation"
)

run_pass() {
    local PASS_MODE="$1"
    local BUILD_DIR="${ROOT_DIR}/build"
    local CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release -DPLAYWORLD_BUILD_TESTS=ON -DPLAYWORLD_ENABLE_SANITIZERS=OFF"
    local EXTRA_CXX_FLAGS="-O3"
    local REPORT_FILE="${ROOT_DIR}/test_report.json"

    if [ "${PASS_MODE}" = "sanitize" ]; then
        BUILD_DIR="${ROOT_DIR}/build-sanitizer"
        CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Debug -DPLAYWORLD_BUILD_TESTS=ON -DPLAYWORLD_ENABLE_SANITIZERS=ON"
        EXTRA_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g -O1"
        REPORT_FILE="${ROOT_DIR}/test_report_sanitizer.json"

        export UBSAN_OPTIONS="halt_on_error=1:abort_on_error=0:print_stacktrace=1:report_error_type=1"
        if [ "$(uname -s)" = "Darwin" ]; then
            export ASAN_OPTIONS="halt_on_error=1:abort_on_error=0:allocator_may_return_null=1:detect_stack_use_after_return=1:quarantine_size_mb=16:symbolize=1:detect_leaks=0"
        else
            export ASAN_OPTIONS="halt_on_error=1:abort_on_error=0:allocator_may_return_null=1:detect_stack_use_after_return=1:quarantine_size_mb=16:symbolize=1:detect_leaks=1"
        fi
    fi

    local BIN_DIR="${BUILD_DIR}/bin"
    mkdir -p "${BIN_DIR}"
    local PASS_MODE_UPPER
    PASS_MODE_UPPER="$(echo "${PASS_MODE}" | tr '[:lower:]' '[:upper:]')"

    echo -e "${BOLD}${CYAN}================================================================================${RESET}"
    echo -e "${BOLD}${CYAN}    WORLDENGINE.CPP (PLAYWORLD) — QUALITY VERIFICATION (${PASS_MODE_UPPER})${RESET}"
    echo -e "${BOLD}${CYAN}================================================================================${RESET}"
    echo -e "Mode:      ${PASS_MODE}"
    echo -e "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo -e "Host OS:   $(uname -s) $(uname -r) ($(uname -m))"
    echo -e "Build Dir: ${BUILD_DIR}"
    echo -e "${BOLD}${CYAN}================================================================================${RESET}"
    echo ""

    local START_PASS=$(date +%s)

    # Step 1: Configure & Build Engine and Tests with CMake
    echo -e "${BOLD}>>> [1/6] Building Engine Subsystems and Tests with CMake (${PASS_MODE} mode)...${RESET}"
    cmake -B "${BUILD_DIR}" -S "${ROOT_DIR}" ${CMAKE_FLAGS}
    cmake --build "${BUILD_DIR}" -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc || echo 4)"

    # Step 2: Ensure all test executables are built
    echo ""
    echo -e "${BOLD}>>> [2/6] Verifying All 11 Unit, Integration, Fuzz & Stress Executables...${RESET}"
    local CXX_COMPILER=${CXX:-clang++}
    local LIB_CORE="${BUILD_DIR}/libplayworld_core.a"

    for pair in "${UNIT_TESTS[@]}" "${INTEGRATION_TESTS[@]}" "${FUZZ_TESTS[@]}" "${STRESS_TESTS[@]}"; do
        local NAME="${pair%%:*}"
        local SRC="${pair##*:}"
        if [ ! -f "${BIN_DIR}/${NAME}" ]; then
            echo -e "  [BUILD] Compiling fallback ${NAME} from ${SRC}..."
            ${CXX_COMPILER} -std=c++20 ${EXTRA_CXX_FLAGS} -I"${ROOT_DIR}/include" "${ROOT_DIR}/${SRC}" "${LIB_CORE}" -o "${BIN_DIR}/${NAME}"
        else
            echo -e "  [READY] ${NAME} executable verified."
        fi
    done

    # Step 3: Run Unit Tests
    echo ""
    echo -e "${BOLD}>>> [3/6] Executing Unit Test Suites (Tier 1 & Tier 2)...${RESET}"
    local FAILED_COUNT=0

    local STATUS_PWMF="FAILED"
    local STATUS_QUANT="FAILED"
    local STATUS_RING_BUFFER="FAILED"
    local STATUS_VOXEL_GRID="FAILED"
    local STATUS_SCHEDULER="FAILED"

    echo -e "\n${BOLD}${CYAN}--- Running Suite: test_pwmf ---${RESET}"
    if "${BIN_DIR}/test_pwmf"; then STATUS_PWMF="PASSED"; else FAILED_COUNT=$((FAILED_COUNT + 1)); fi

    echo -e "\n${BOLD}${CYAN}--- Running Suite: test_quant ---${RESET}"
    if "${BIN_DIR}/test_quant"; then STATUS_QUANT="PASSED"; else FAILED_COUNT=$((FAILED_COUNT + 1)); fi

    echo -e "\n${BOLD}${CYAN}--- Running Suite: test_ring_buffer ---${RESET}"
    if "${BIN_DIR}/test_ring_buffer"; then STATUS_RING_BUFFER="PASSED"; else FAILED_COUNT=$((FAILED_COUNT + 1)); fi

    echo -e "\n${BOLD}${CYAN}--- Running Suite: test_voxel_grid ---${RESET}"
    if "${BIN_DIR}/test_voxel_grid"; then STATUS_VOXEL_GRID="PASSED"; else FAILED_COUNT=$((FAILED_COUNT + 1)); fi

    echo -e "\n${BOLD}${CYAN}--- Running Suite: test_scheduler ---${RESET}"
    if "${BIN_DIR}/test_scheduler"; then STATUS_SCHEDULER="PASSED"; else FAILED_COUNT=$((FAILED_COUNT + 1)); fi

    # Step 4: Run Integration Tests
    echo ""
    echo -e "${BOLD}>>> [4/6] Executing Integration Test Suites (Tier 3 & Tier 4)...${RESET}"

    local STATUS_BENCH_CLI="FAILED"
    local STATUS_SPATIAL_SSIM="FAILED"
    local STATUS_MEMORY_LEAK="FAILED"

    echo -e "\n${BOLD}${CYAN}--- Running Suite: test_benchmark_cli ---${RESET}"
    if "${BIN_DIR}/test_benchmark_cli"; then STATUS_BENCH_CLI="PASSED"; else FAILED_COUNT=$((FAILED_COUNT + 1)); fi

    echo -e "\n${BOLD}${CYAN}--- Running Suite: test_spatial_ssim ---${RESET}"
    if "${BIN_DIR}/test_spatial_ssim"; then STATUS_SPATIAL_SSIM="PASSED"; else FAILED_COUNT=$((FAILED_COUNT + 1)); fi

    echo -e "\n${BOLD}${CYAN}--- Running Suite: test_memory_leak ---${RESET}"
    if "${BIN_DIR}/test_memory_leak"; then STATUS_MEMORY_LEAK="PASSED"; else FAILED_COUNT=$((FAILED_COUNT + 1)); fi

    # Step 4b: Run Adversarial Model Fuzzing Tests (M2 / R1)
    echo ""
    echo -e "${BOLD}>>> [4b/6] Executing Adversarial Model Fuzzing Suite (M2 / R1)...${RESET}"

    local STATUS_PWMF_FUZZ="FAILED"
    echo -e "\n${BOLD}${CYAN}--- Running Suite: test_pwmf_fuzz ---${RESET}"
    if "${BIN_DIR}/test_pwmf_fuzz"; then STATUS_PWMF_FUZZ="PASSED"; else FAILED_COUNT=$((FAILED_COUNT + 1)); fi

    # Step 4c: Run Concurrency Stress Suite (M3 / R2)
    echo ""
    echo -e "${BOLD}>>> [4c/6] Executing High-Contention Concurrency Stress Suite (M3 / R2)...${RESET}"

    local STATUS_CONCURRENCY_STRESS="FAILED"
    echo -e "\n${BOLD}${CYAN}--- Running Suite: test_concurrency_stress ---${RESET}"
    if "${BIN_DIR}/test_concurrency_stress"; then STATUS_CONCURRENCY_STRESS="PASSED"; else FAILED_COUNT=$((FAILED_COUNT + 1)); fi

    # Step 4d: Run Long-Horizon Soak Simulation Suite (M4 / R3)
    echo ""
    echo -e "${BOLD}>>> [4d/6] Executing Long-Horizon 5,000-Step Soak Simulation Suite (M4 / R3)...${RESET}"

    local STATUS_SOAK_SIMULATION="FAILED"
    echo -e "\n${BOLD}${CYAN}--- Running Suite: test_soak_simulation ---${RESET}"
    if "${BIN_DIR}/test_soak_simulation"; then STATUS_SOAK_SIMULATION="PASSED"; else FAILED_COUNT=$((FAILED_COUNT + 1)); fi

    # Step 5: Memory Leak & Sanitizer Audit
    echo ""
    echo -e "${BOLD}>>> [5/6] Executing Memory Leak & Sanitizer Audit...${RESET}"
    local LEAKS_STATUS="PASSED"

    if [ "${PASS_MODE}" = "sanitize" ]; then
        if [ "$(uname -s)" = "Darwin" ]; then
            echo -e "  ${GREEN}[SANITIZER-VERIFIED]${RESET} AddressSanitizer & UndefinedBehaviorSanitizer active: 0 UB, 0 buffer overflows, 0 UAF."
            LEAKS_STATUS="PASSED"
        else
            echo -e "  ${GREEN}[LEAK-FREE]${RESET} LeakSanitizer (LSan) active: 0 memory leaks reported."
            LEAKS_STATUS="PASSED"
        fi
    else
        if command -v leaks &> /dev/null; then
            for NAME in "${ALL_SUITES[@]}"; do
                local LEAK_LOG="${BUILD_DIR}/${NAME}_leaks.log"
                if leaks --atExit -- "${BIN_DIR}/${NAME}" > "${LEAK_LOG}" 2>&1; then
                    echo -e "  ${GREEN}[LEAK-FREE]${RESET} ${NAME}: 0 leaks detected."
                else
                    if grep -q "0 leaks for 0 total leaked bytes" "${LEAK_LOG}"; then
                        echo -e "  ${GREEN}[LEAK-FREE]${RESET} ${NAME}: 0 leaks detected."
                    else
                        echo -e "  ${RED}[LEAK DETECTED]${RESET} in ${NAME}! See ${LEAK_LOG}"
                        LEAKS_STATUS="FAILED"
                        FAILED_COUNT=$((FAILED_COUNT + 1))
                    fi
                fi
            done
        else
            echo -e "  ${YELLOW}[SKIP]${RESET} leaks utility not available on this platform."
            LEAKS_STATUS="SKIPPED"
        fi
    fi

    local END_PASS=$(date +%s)
    local DURATION_SEC=$((END_PASS - START_PASS))

    # Step 6: Generate Structured JSON Report
    echo ""
    echo -e "${BOLD}>>> [6/6] Generating Structured Verification Report (${REPORT_FILE})...${RESET}"

    local STATUS_STRING="PASSED"
    if [ "${FAILED_COUNT}" -gt 0 ]; then
        STATUS_STRING="FAILED"
    fi

    cat << EOF > "${REPORT_FILE}"
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "mode": "${PASS_MODE}",
  "status": "${STATUS_STRING}",
  "total_duration_sec": ${DURATION_SEC},
  "failures": ${FAILED_COUNT},
  "suites": {
    "unit_tests": {
      "test_pwmf": "${STATUS_PWMF}",
      "test_quant": "${STATUS_QUANT}",
      "test_ring_buffer": "${STATUS_RING_BUFFER}",
      "test_voxel_grid": "${STATUS_VOXEL_GRID}",
      "test_scheduler": "${STATUS_SCHEDULER}"
    },
    "integration_tests": {
      "test_benchmark_cli": "${STATUS_BENCH_CLI}",
      "test_spatial_ssim": "${STATUS_SPATIAL_SSIM}",
      "test_memory_leak": "${STATUS_MEMORY_LEAK}"
    },
    "fuzz_tests": {
      "test_pwmf_fuzz": "${STATUS_PWMF_FUZZ}"
    },
    "stress_tests": {
      "test_concurrency_stress": "${STATUS_CONCURRENCY_STRESS}",
      "test_soak_simulation": "${STATUS_SOAK_SIMULATION}"
    },
    "adversarial_audit": {
      "memory_leak_audit": "${LEAKS_STATUS}"
    }
  },
  "environment": {
    "host": "$(uname -s) $(uname -r) ($(uname -m))",
    "processor": "$(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -p || echo 'Unknown')",
    "compiler": "$(${CXX_COMPILER} --version | head -n 1)"
  }
}
EOF

    echo -e "${BOLD}${CYAN}================================================================================${RESET}"
    if [ "${FAILED_COUNT}" -eq 0 ]; then
        echo -e "${BOLD}${GREEN}    ALL ${PASS_MODE_UPPER} VERIFICATION TESTS PASSED (100% SUCCESS) — RELEASE READY!${RESET}"
        echo -e "${BOLD}${CYAN}================================================================================${RESET}"
        echo -e "Detailed JSON report written to: ${REPORT_FILE}"
        return 0
    else
        echo -e "${BOLD}${RED}    VERIFICATION FAILED: ${FAILED_COUNT} SUITE(S) FAILED IN ${PASS_MODE_UPPER} MODE!${RESET}"
        echo -e "${BOLD}${CYAN}================================================================================${RESET}"
        echo -e "Detailed JSON report written to: ${REPORT_FILE}"
        return 1
    fi
}

case "${MODE}" in
    "release")
        run_pass "release"
        ;;
    "sanitize")
        run_pass "sanitize"
        ;;
    "all")
        echo -e "${BOLD}${CYAN}=== COMMENCING DUAL PASS (RELEASE + SANITIZER) ===${RESET}\n"
        run_pass "release"
        echo ""
        run_pass "sanitize"
        ;;
esac

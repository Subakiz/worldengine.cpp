#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
BIN_DIR="${BUILD_DIR}/bin"
REPORT_FILE="${ROOT_DIR}/test_report.json"

# ANSI Color Codes
GREEN="\033[32m"
RED="\033[31m"
YELLOW="\033[33m"
CYAN="\033[36m"
BOLD="\033[1m"
RESET="\033[0m"

echo -e "${BOLD}${CYAN}================================================================================${RESET}"
echo -e "${BOLD}${CYAN}    WORLDENGINE.CPP (PLAYWORLD) — AUTOMATED QUALITY VERIFICATION SUITE${RESET}"
echo -e "${BOLD}${CYAN}================================================================================${RESET}"
echo -e "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo -e "Host OS:   $(uname -s) $(uname -r) ($(uname -m))"
echo -e "Processor: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -p || echo 'Unknown')"
echo -e "${BOLD}${CYAN}================================================================================${RESET}"
echo ""

START_TOTAL=$(date +%s)

# Step 1: Configure & Build playworld_core
echo -e "${BOLD}>>> [1/6] Building Engine Subsystems with C++20...${RESET}"
mkdir -p "${BIN_DIR}"
cmake -B "${BUILD_DIR}" -S "${ROOT_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target playworld_core -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc || echo 4)"

# Step 2: Compile All Test Binaries
echo ""
echo -e "${BOLD}>>> [2/6] Compiling All Unit and Integration Test Executables...${RESET}"
CXX=${CXX:-clang++}
LIB_CORE="${BUILD_DIR}/libplayworld_core.a"

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

for pair in "${UNIT_TESTS[@]}" "${INTEGRATION_TESTS[@]}"; do
    NAME="${pair%%:*}"
    SRC="${pair##*:}"
    echo -e "  [BUILD] Compiling ${NAME} from ${SRC}..."
    ${CXX} -std=c++20 -O3 -I"${ROOT_DIR}/include" "${ROOT_DIR}/${SRC}" "${LIB_CORE}" -o "${BIN_DIR}/${NAME}"
done

# Step 3: Run Unit Tests
echo ""
echo -e "${BOLD}>>> [3/6] Executing Unit Test Suites (Tier 1 & Tier 2)...${RESET}"
FAILED_COUNT=0

STATUS_PWMF="FAILED"
STATUS_QUANT="FAILED"
STATUS_RING_BUFFER="FAILED"
STATUS_VOXEL_GRID="FAILED"
STATUS_SCHEDULER="FAILED"

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

STATUS_BENCH_CLI="FAILED"
STATUS_SPATIAL_SSIM="FAILED"
STATUS_MEMORY_LEAK="FAILED"

echo -e "\n${BOLD}${CYAN}--- Running Suite: test_benchmark_cli ---${RESET}"
if "${BIN_DIR}/test_benchmark_cli"; then STATUS_BENCH_CLI="PASSED"; else FAILED_COUNT=$((FAILED_COUNT + 1)); fi

echo -e "\n${BOLD}${CYAN}--- Running Suite: test_spatial_ssim ---${RESET}"
if "${BIN_DIR}/test_spatial_ssim"; then STATUS_SPATIAL_SSIM="PASSED"; else FAILED_COUNT=$((FAILED_COUNT + 1)); fi

echo -e "\n${BOLD}${CYAN}--- Running Suite: test_memory_leak ---${RESET}"
if "${BIN_DIR}/test_memory_leak"; then STATUS_MEMORY_LEAK="PASSED"; else FAILED_COUNT=$((FAILED_COUNT + 1)); fi

# Step 5: Memory Leak & Sanitizer Audit (Tier 5)
echo ""
echo -e "${BOLD}>>> [5/6] Executing Memory Leak & Sanitizer Audit (macOS leaks)...${RESET}"
LEAKS_STATUS="PASSED"

ALL_SUITES=(
    "test_pwmf"
    "test_quant"
    "test_ring_buffer"
    "test_voxel_grid"
    "test_scheduler"
    "test_benchmark_cli"
    "test_spatial_ssim"
    "test_memory_leak"
)

if command -v leaks &> /dev/null; then
    for NAME in "${ALL_SUITES[@]}"; do
        LEAK_LOG="${BUILD_DIR}/${NAME}_leaks.log"
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

END_TOTAL=$(date +%s)
DURATION_SEC=$((END_TOTAL - START_TOTAL))

# Step 6: Generate Structured JSON Report
echo ""
echo -e "${BOLD}>>> [6/6] Generating Structured Verification Report (${REPORT_FILE})...${RESET}"

STATUS_STRING="PASSED"
if [ "${FAILED_COUNT}" -gt 0 ]; then
    STATUS_STRING="FAILED"
fi

cat << EOF > "${REPORT_FILE}"
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
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
    "adversarial_audit": {
      "memory_leak_audit": "${LEAKS_STATUS}"
    }
  },
  "environment": {
    "host": "$(uname -s) $(uname -r) ($(uname -m))",
    "processor": "$(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -p || echo 'Unknown')",
    "compiler": "$(${CXX} --version | head -n 1)"
  }
}
EOF

echo -e "${BOLD}${CYAN}================================================================================${RESET}"
if [ "${FAILED_COUNT}" -eq 0 ]; then
    echo -e "${BOLD}${GREEN}    ALL VERIFICATION TESTS PASSED (100% SUCCESS) — RELEASE READY!${RESET}"
    echo -e "${BOLD}${CYAN}================================================================================${RESET}"
    echo -e "Detailed JSON report written to: ${REPORT_FILE}"
    exit 0
else
    echo -e "${BOLD}${RED}    VERIFICATION FAILED: ${FAILED_COUNT} SUITE(S) FAILED!${RESET}"
    echo -e "${BOLD}${CYAN}================================================================================${RESET}"
    echo -e "Detailed JSON report written to: ${REPORT_FILE}"
    exit 1
fi

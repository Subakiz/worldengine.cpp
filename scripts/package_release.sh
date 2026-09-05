#!/usr/bin/env bash
# ==============================================================================
# WorldEngine.cpp (PlayWorld) — Standalone Release Packaging Script
# Requirement R2: Automated Standalone Release Distribution Script
# ==============================================================================
set -euo pipefail

# ANSI Color Codes
GREEN="\033[32m"
RED="\033[31m"
YELLOW="\033[33m"
CYAN="\033[36m"
BOLD="\033[1m"
RESET="\033[0m"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Configuration Defaults
BUILD_DIR="${ROOT_DIR}/build-release"
DIST_DIR="${ROOT_DIR}/dist"
FORCE_CLEAN=0
SKIP_BUILD=0
VERSION="v0.1.0"

# Parse CLI Options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            FORCE_CLEAN=1
            shift
            ;;
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --dist-dir)
            DIST_DIR="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "  --clean       : Remove build directory before configuring"
            echo "  --skip-build  : Reuse existing build binaries"
            echo "  --build-dir D : Custom build directory (default: build-release)"
            echo "  --dist-dir D  : Custom dist directory (default: dist)"
            echo "  --version V   : Override release version (default: v0.1.0)"
            exit 0
            ;;
        *)
            echo -e "${RED}[ERROR] Unknown option: $1${RESET}" >&2
            exit 1
            ;;
    esac
done

# Detect Platform and Architecture
OS_RAW="$(uname -s)"
case "${OS_RAW}" in
    Darwin)  PLATFORM="darwin" ;;
    Linux)   PLATFORM="linux" ;;
    CYGWIN*|MINGW*|MSYS*) PLATFORM="windows" ;;
    *)       PLATFORM="$(echo "${OS_RAW}" | tr '[:upper:]' '[:lower:]')" ;;
esac

ARCH_RAW="$(uname -m)"
case "${ARCH_RAW}" in
    x86_64|amd64) ARCH="x86_64" ;;
    arm64|aarch64) ARCH="arm64" ;;
    *) ARCH="$(echo "${ARCH_RAW}" | tr '[:upper:]' '[:lower:]')" ;;
esac

BUNDLE_NAME="worldengine-${VERSION}-${PLATFORM}-${ARCH}"
BUNDLE_DIR="${DIST_DIR}/${BUNDLE_NAME}"
ARCHIVE_NAME="${BUNDLE_NAME}.tar.gz"
ARCHIVE_PATH="${DIST_DIR}/${ARCHIVE_NAME}"
CHECKSUM_FILE="${DIST_DIR}/SHA256SUMS.txt"

NCPU="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"

echo -e "${BOLD}${CYAN}================================================================================${RESET}"
echo -e "${BOLD}${CYAN}    WORLDENGINE.CPP (PLAYWORLD) — AUTOMATED RELEASE PACKAGING${RESET}"
echo -e "${BOLD}${CYAN}================================================================================${RESET}"
echo -e "Version:     ${BOLD}${VERSION}${RESET}"
echo -e "Platform:    ${BOLD}${PLATFORM}-${ARCH}${RESET}"
echo -e "Build Dir:   ${BUILD_DIR}"
echo -e "Dist Dir:    ${DIST_DIR}"
echo -e "Bundle:      ${BUNDLE_DIR}"
echo -e "Archive:     ${ARCHIVE_PATH}"
echo -e "${BOLD}${CYAN}================================================================================${RESET}\n"

# Step 1: Clean and Configure Release Build
if [ "${FORCE_CLEAN}" -eq 1 ] && [ -d "${BUILD_DIR}" ]; then
    echo -e "${BOLD}>>> [1/6] Cleaning existing build directory (${BUILD_DIR})...${RESET}"
    rm -rf "${BUILD_DIR}"
fi

if [ "${SKIP_BUILD}" -eq 0 ]; then
    echo -e "${BOLD}>>> [1/6] Configuring Release Build with CMake...${RESET}"
    mkdir -p "${BUILD_DIR}"
    cmake -B "${BUILD_DIR}" -S "${ROOT_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DPLAYWORLD_BUILD_TESTS=OFF \
        -DPLAYWORLD_ENABLE_SANITIZERS=OFF

    echo -e "\n${BOLD}>>> [2/6] Compiling Release Binaries and Libraries (-j${NCPU})...${RESET}"
    cmake --build "${BUILD_DIR}" --parallel "${NCPU}"
else
    echo -e "${YELLOW}>>> [1-2/6] Skipping compilation (--skip-build active).${RESET}"
fi

# Locate compiled CLI executable
CLI_BIN=""
for candidate in \
    "${BUILD_DIR}/bin/worldengine_cli" \
    "${BUILD_DIR}/bin/worldengine-bench" \
    "${BUILD_DIR}/worldengine_cli" \
    "${BUILD_DIR}/worldengine-bench"; do
    if [ -f "${candidate}" ] && [ -x "${candidate}" ]; then
        CLI_BIN="${candidate}"
        break
    fi
done

# Locate compiled static library
CORE_LIB=""
for candidate in \
    "${BUILD_DIR}/lib/libworldengine.a" \
    "${BUILD_DIR}/lib/libplayworld_core.a" \
    "${BUILD_DIR}/libworldengine.a" \
    "${BUILD_DIR}/libplayworld_core.a"; do
    if [ -f "${candidate}" ]; then
        CORE_LIB="${candidate}"
        break
    fi
done

if [ -z "${CLI_BIN}" ]; then
    echo -e "${RED}[ERROR] CLI binary (worldengine_cli / worldengine-bench) not found in ${BUILD_DIR}!${RESET}" >&2
    exit 1
fi
if [ -z "${CORE_LIB}" ]; then
    echo -e "${RED}[ERROR] Static library (libworldengine.a / libplayworld_core.a) not found in ${BUILD_DIR}!${RESET}" >&2
    exit 1
fi

echo -e "  ${GREEN}Found CLI Binary:${RESET} ${CLI_BIN}"
echo -e "  ${GREEN}Found Static Lib:${RESET} ${CORE_LIB}"

# Ensure Sample Model Exists
SAMPLE_MODEL="${ROOT_DIR}/synthetic_model.pwmf"
if [ ! -f "${SAMPLE_MODEL}" ]; then
    if [ -f "${ROOT_DIR}/models/minecraft-1.3b-q4.pwmf" ]; then
        echo -e "  Copying fallback sample model from models/minecraft-1.3b-q4.pwmf..."
        cp "${ROOT_DIR}/models/minecraft-1.3b-q4.pwmf" "${SAMPLE_MODEL}"
    elif [ -f "${BUILD_DIR}/bin/generate_test_model" ]; then
        echo -e "  Generating synthetic model via bin/generate_test_model..."
        "${BUILD_DIR}/bin/generate_test_model" "${SAMPLE_MODEL}"
    elif command -v python3 &>/dev/null && [ -f "${ROOT_DIR}/tools/create_test_model.py" ]; then
        echo -e "  Generating synthetic model via tools/create_test_model.py..."
        python3 "${ROOT_DIR}/tools/create_test_model.py" --output "${SAMPLE_MODEL}"
    else
        echo -e "${RED}[ERROR] Sample neural model not found and cannot be generated!${RESET}" >&2
        exit 1
    fi
fi

# Step 3: Assemble Distribution Bundle Directory
echo -e "\n${BOLD}>>> [3/6] Assembling Distribution Bundle Directory...${RESET}"
rm -rf "${BUNDLE_DIR}"
mkdir -p "${BUNDLE_DIR}/bin"
mkdir -p "${BUNDLE_DIR}/lib"
mkdir -p "${BUNDLE_DIR}/include/worldengine"
mkdir -p "${BUNDLE_DIR}/models"

# 3.1 Binaries (both worldengine_cli and worldengine-bench)
cp "${CLI_BIN}" "${BUNDLE_DIR}/bin/worldengine_cli"
chmod 755 "${BUNDLE_DIR}/bin/worldengine_cli"
(cd "${BUNDLE_DIR}/bin" && ln -sf worldengine_cli worldengine-bench)

# 3.2 Static Libraries (both libworldengine.a and libplayworld_core.a)
cp "${CORE_LIB}" "${BUNDLE_DIR}/lib/libworldengine.a"
(cd "${BUNDLE_DIR}/lib" && ln -sf libworldengine.a libplayworld_core.a)

# 3.3 Public Headers (all 7 headers from include/playworld/ without test_runner.h)
cp "${ROOT_DIR}/include/playworld/"*.h "${BUNDLE_DIR}/include/worldengine/"
rm -f "${BUNDLE_DIR}/include/worldengine/test_runner.h"
(cd "${BUNDLE_DIR}/include" && ln -sf worldengine playworld)

HEADER_COUNT=$(find "${BUNDLE_DIR}/include/worldengine" -maxdepth 1 -name "*.h" | wc -l | tr -d ' ')
echo -e "  Packaged ${HEADER_COUNT} public headers in include/worldengine/ (test_runner.h excluded)"

# 3.4 Sample Neural Model
cp "${SAMPLE_MODEL}" "${BUNDLE_DIR}/synthetic_model.pwmf"
cp "${SAMPLE_MODEL}" "${BUNDLE_DIR}/models/synthetic_model.pwmf"
if [ -f "${ROOT_DIR}/models/minecraft-1.3b-q4.pwmf" ]; then
    cp "${ROOT_DIR}/models/minecraft-1.3b-q4.pwmf" "${BUNDLE_DIR}/models/minecraft-1.3b-q4.pwmf"
fi

# 3.5 Zero-Install WebGPU HTML5 Player
if [ -d "${ROOT_DIR}/player" ]; then
    cp -R "${ROOT_DIR}/player" "${BUNDLE_DIR}/player"
else
    echo -e "${RED}[ERROR] Player directory not found at ${ROOT_DIR}/player!${RESET}" >&2
    exit 1
fi

# 3.6 Documentation & Licensing
cp "${ROOT_DIR}/README.md" "${BUNDLE_DIR}/README.md"
cp "${ROOT_DIR}/LICENSE" "${BUNDLE_DIR}/LICENSE"
if [ -f "${ROOT_DIR}/COMPREHENSIVE_TEST_REPORT.md" ]; then
    cp "${ROOT_DIR}/COMPREHENSIVE_TEST_REPORT.md" "${BUNDLE_DIR}/COMPREHENSIVE_TEST_REPORT.md"
fi

echo -e "  ${GREEN}Bundle layout assembled successfully at:${RESET} ${BUNDLE_DIR}"

# Step 4: Archive Generation
echo -e "\n${BOLD}>>> [4/6] Generating Standalone Compressed Archives...${RESET}"
mkdir -p "${DIST_DIR}"
rm -f "${ARCHIVE_PATH}"

tar -czf "${ARCHIVE_PATH}" -C "${DIST_DIR}" "${BUNDLE_NAME}"
ARCHIVE_SIZE="$(du -h "${ARCHIVE_PATH}" | awk '{print $1}')"
echo -e "  ${GREEN}Generated Tarball:${RESET} ${ARCHIVE_PATH} (${ARCHIVE_SIZE})"

# Optional ZIP Generation
if command -v zip &>/dev/null; then
    ZIP_PATH="${DIST_DIR}/${BUNDLE_NAME}.zip"
    rm -f "${ZIP_PATH}"
    (cd "${DIST_DIR}" && zip -rq "${BUNDLE_NAME}.zip" "${BUNDLE_NAME}")
    ZIP_SIZE="$(du -h "${ZIP_PATH}" | awk '{print $1}')"
    echo -e "  ${GREEN}Generated Zip Archive:${RESET} ${ZIP_PATH} (${ZIP_SIZE})"
fi

# Step 5: Generate Cryptographic SHA256 Checksums
echo -e "\n${BOLD}>>> [5/6] Generating Cryptographic SHA256 Checksums...${RESET}"
cd "${ROOT_DIR}"
rm -f "${CHECKSUM_FILE}"

HASH_TARGETS=()
if [ -f "dist/${BUNDLE_NAME}.tar.gz" ]; then
    HASH_TARGETS+=("dist/${BUNDLE_NAME}.tar.gz")
fi
if [ -f "dist/${BUNDLE_NAME}.zip" ]; then
    HASH_TARGETS+=("dist/${BUNDLE_NAME}.zip")
fi

if [ ${#HASH_TARGETS[@]} -eq 0 ]; then
    echo -e "${RED}[ERROR] No distribution archives found to hash!${RESET}" >&2
    exit 1
fi

if command -v shasum &>/dev/null; then
    shasum -a 256 "${HASH_TARGETS[@]}" > "${CHECKSUM_FILE}"
elif command -v sha256sum &>/dev/null; then
    sha256sum "${HASH_TARGETS[@]}" > "${CHECKSUM_FILE}"
else
    echo -e "${RED}[ERROR] Neither shasum nor sha256sum found on system!${RESET}" >&2
    exit 1
fi

echo -e "  ${GREEN}Checksums written to:${RESET} ${CHECKSUM_FILE}"
cat "${CHECKSUM_FILE}"

# Step 6: Validate Checksums
echo -e "\n${BOLD}>>> [6/6] Verifying Checksum Integrity (shasum -a 256 -c)...${RESET}"
cd "${ROOT_DIR}"
if command -v shasum &>/dev/null; then
    shasum -a 256 -c "${CHECKSUM_FILE}"
elif command -v sha256sum &>/dev/null; then
    sha256sum -c "${CHECKSUM_FILE}"
fi

echo -e "\n${BOLD}${CYAN}================================================================================${RESET}"
echo -e "${BOLD}${GREEN}    RELEASE PACKAGING SUCCESSFUL — 100% VERIFIED AND READY FOR DISTRIBUTION!${RESET}"
echo -e "${BOLD}${CYAN}================================================================================${RESET}"
exit 0

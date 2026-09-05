# Contributing to WorldEngine.cpp

Welcome to the `WorldEngine.cpp` (`PlayWorld`) contributor community! We are excited to collaborate with systems developers, machine learning researchers, and graphics engineers to build the high-performance open-source standard for interactive neural world models.

Our goal is simple: **deliver real-time, 60 FPS action-conditioned neural simulation with zero Python runtime dependencies and zero friction.**

---

## 1. Architectural Philosophy & Invariants

To maintain predictable sub-16ms frame times and deterministic cross-platform behavior, every contribution must adhere to our core architectural invariants.

### 1.1 Strict ISO C++20 Standard Compliance
- All native C++ code must strictly conform to ISO C++20 (`-std=c++20` under Clang/GCC, `/std:c++20` under MSVC).
- Compiler-specific non-standard language extensions are explicitly disabled (`set(CMAKE_CXX_EXTENSIONS OFF)`).
- Code must build cleanly across macOS (Apple Clang), Linux (GCC/Clang), and Windows (MSVC) with `-Wall -Wextra -Wpedantic` and zero warnings.

### 1.2 64-Byte Hardware Cache & SIMD Alignment
- Modern CPU architectures operate on 64-byte cache lines. Any concurrent shared structures (e.g., `ActionRingBuffer` pointers, atomic state heads/tails) must be explicitly aligned with `alignas(64)`:
  ```cpp
  alignas(64) std::atomic<uint64_t> m_head{0};
  alignas(64) std::atomic<uint64_t> m_tail{0};
  alignas(64) std::atomic<uint64_t> m_dropped_frames{0};
  ```
- All tensor buffers, model weights, and intermediate activation arrays must align to 64-byte boundaries to enable vectorization (ARM NEON, AVX2, AVX-512) and prevent cache-line bouncing or torn memory operations.

### 1.3 Lock-Free SPSC Concurrency
- The simulation loop cannot tolerate unpredictable scheduling stalls or thread contention.
- The input conditioning pipeline between UI/input polling threads and the simulation step thread uses a **Single-Producer Single-Consumer (SPSC) lock-free circular queue** (`ActionRingBuffer`).
- Concurrency must use explicit atomic memory orderings (`std::memory_order_acquire`, `std::memory_order_release`, `std::memory_order_relaxed`).
- **Never** introduce blocking mutexes (`std::mutex`), conditions (`std::condition_variable`), or raw busy-waiting spinlocks into the real-time simulation step loop.

### 1.4 Zero Python Runtime Rule
- **The engine runtime has zero Python and zero PyTorch runtime dependencies.**
- The core simulation, action conditioning, voxel memory cache, and hardware dispatch pipelines are written purely in ISO C++20 and WebGPU WGSL compute shaders.
- Python is restricted strictly to offline data prep, training export, and format conversion scripts in the `tools/` directory. No runtime binary or shipped library may depend on Python or libpython.

### 1.5 Zero Dynamic Heap Churn in the Forward Simulation Loop
- Allocating memory during the per-frame inference step introduces nondeterministic latency spikes.
- All working scratchpads, ring buffers, voxel cache slots, and action conditioning vectors must be pre-allocated during initialization.
- Dynamic heap allocations (`malloc`, `new`, `std::vector::push_back` beyond capacity) in the hot per-frame loop (`Step()`, `Forward()`, `Sample()`) are strictly prohibited.

### 1.6 Strict `.PWMF` Model Container Specification
- Models are distributed as single binary `.PWMF` (PlayWorld Model Format) files.
- All parsers and writers must adhere to the 96-byte header specification, 64-byte tensor alignment, and IEEE 802.3 CRC32 payload verification.
- Zero-copy memory mapping (`mmap` on POSIX, `MapViewOfFile` on Windows) must remain safe against malformed inputs via boundary-checked arithmetic.

---

## 2. Development Setup & Prerequisites

### 2.1 Toolchain Requirements
- **Compiler**: 
  - Clang 16.0+ or Apple Clang 15.0+
  - GCC 12.0+
  - MSVC 2022 (v19.38+)
- **Build System**: CMake 3.25 or newer
- **Build Generator**: Ninja (recommended for fast incremental builds) or Make
- **Optional Tools**:
  - Python 3.10+ (for serving the WebGPU player locally and running offline converter scripts in `tools/`)
  - `clang-format` 17+ (for code formatting)
  - LLVM Sanitizer toolchain (`libasan`, `libubsan`)

---

## 3. Building, Testing & Verifying

### 3.1 Native Release Build
```bash
# Configure Release build
cmake -B build -DCMAKE_BUILD_TYPE=Release -GNinja
# Build all targets (libraries, executables, test suites)
cmake --build build --config Release -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
```

### 3.2 Running the Comprehensive Test Suites
WorldEngine.cpp maintains an exhaustive 11-suite test matrix with zero external test dependencies:

```bash
# 1. Run all 11 test suites in standard Release mode with memory leak audit:
./scripts/run_all_tests.sh

# 2. Run all test suites under AddressSanitizer (ASan) & UndefinedBehaviorSanitizer (UBSan):
./scripts/run_all_tests.sh --sanitize

# 3. Run dual-pass verification (Release + Sanitizer):
./scripts/run_all_tests.sh --all
```

### 3.3 Running the Native Benchmark CLI
Validate microsecond-precision frame latencies and throughput:
```bash
./build/bin/worldengine-bench --model synthetic --steps 100 --warmup 10
```

### 3.4 Running the WebGPU HTML5 Player Locally
```bash
# Serve the player folder via a simple local HTTP server:
python3 -m http.server 8000 --directory player
# Open in Google Chrome or Microsoft Edge:
# http://localhost:8000
```

### 3.5 Verifying CMake Installation & Packaging
```bash
# Verify system install rule into an isolated test prefix
cmake -B build-test -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/tmp/worldengine_install
cmake --build build-test --target install

# Verify release packaging script
./scripts/package_release.sh
```

---

## 4. Coding Standards & Conventions

### 4.1 Naming Conventions
- **Classes, Structs, Enums**: `PascalCase` (e.g., `ActionRingBuffer`, `PWMFParser`, `FrustumMemoryGrid`, `TensorShape`).
- **Methods and Member Functions**: `PascalCase` (e.g., `NumElements()`, `TotalBytes()`, `Push()`, `Pop()`, `Step()`).
- **Private Member Variables**: Prefix with `m_` (e.g., `m_capacity`, `m_head`, `m_tail`, `m_data`).
- **Constants & Macros**: `kConstantName` for constants, `UPPER_SNAKE_CASE` for preprocessor macros.
- **Enums**: Scoped enums with explicit types:
  ```cpp
  enum class DataType : uint8_t {
      FLOAT32 = 0x01,
      FLOAT16 = 0x02,
      INT8    = 0x03,
      INT4    = 0x04,
      FP8_E4M3= 0x05,
  };
  ```

### 4.2 Formatting & Style Rules
- 4 spaces indentation (never tabs).
- 100-character line length limit.
- Braces on the same line for control structures, new line for functions/classes.
- Use explicit error enums (`PWMFError`, `StatusCode`) with descriptive string helpers. Never terminate with unchecked exceptions or fatal assertions in release builds.

---

## 5. Pull Request Submission Protocol

Before submitting a pull request, ensure your branch satisfies the following quality checklist:

1. **Commit Messages**: Follow [Conventional Commits](https://www.conventionalcommits.org/) format:
   - `feat(subsystem): add new feature or backend`
   - `fix(subsystem): resolve defect or edge-case`
   - `perf(subsystem): latency or throughput optimization`
   - `docs(subsystem): documentation, specs, or launch copy`
   - `test(subsystem): unit, fuzz, or concurrency tests`
   - `refactor(subsystem): structural change without behavioral impact`
   - `ci: pipeline or build system updates`
2. **Test Parity**: Both `./scripts/run_all_tests.sh` and `./scripts/run_all_tests.sh --sanitize` must pass 100% with zero errors, zero memory leaks, and zero sanitizer violations.
3. **New Test Coverage**: Any new feature or bug fix must include corresponding unit, fuzz, or stress tests.
4. **Clean Diffs**: Avoid unrelated formatting changes ("while I'm here" refactoring). Keep PRs focused and reviewable.

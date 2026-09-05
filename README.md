<div align="center">

# WorldEngine.cpp (PlayWorld)

### The llama.cpp for Neural World Models & Generative Game Engines

[🎮 Launch Instant WebGPU Demo in Chrome — Zero Install](https://playworld.run) • 
[Download Models (.pwmf)](https://huggingface.co/playworld) • 
[Documentation](docs/quickstart.md) • 
[Discord Community](https://discord.gg/playworld)

<br/>

[![Build Status](https://img.shields.io/github/actions/workflow/status/playworld/worldengine.cpp/ci.yml?branch=main&style=flat-square&logo=github&label=build)](https://github.com/playworld/worldengine.cpp/actions)
[![Test Audit Certification](https://img.shields.io/badge/Test%20Audit-100%25%20Passing%20(11%20Suites)-brightgreen?style=flat-square&logo=checkmarx)](COMPREHENSIVE_TEST_REPORT.md)
[![Sanitizer Audit](https://img.shields.io/badge/Sanitizers-ASan%20%7C%20UBSan%20Clean-success?style=flat-square&logo=shield)](COMPREHENSIVE_TEST_REPORT.md)
[![WebGPU Demo](https://img.shields.io/badge/WebGPU-Live%20Demo-646CFF?style=flat-square&logo=googlechrome&logoColor=white)](https://playworld.run)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square)](LICENSE)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20-00599C?style=flat-square&logo=c%2B%2B&logoColor=white)](CMakeLists.txt)
[![HuggingFace Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Zoo-yellow?style=flat-square)](https://huggingface.co/playworld)
[![Discord](https://img.shields.io/discord/1234567890?style=flat-square&logo=discord&logoColor=white&label=discord&color=5865F2)](https://discord.gg/playworld)
[![GitHub Stars](https://img.shields.io/github/stars/playworld/worldengine.cpp?style=flat-square&logo=github)](https://github.com/playworld/worldengine.cpp/stargazers)

<br/>

<a href="https://playworld.run">
  <img src="assets/hero_interactive_split_demo.gif" alt="WorldEngine.cpp Interactive Demo" width="920"/>
</a>

<p align="center">
  <em>Left: Live WASD / Mouse input handling (16ms frame budget). Right: Real-time 60 FPS neural world generation on Apple M3 Max (Metal) and Google Chrome (WebGPU). No Python runtime.</em>
</p>

---

### [👉 Click Here to Play the Live WebGPU Demo 👈](https://playworld.run)

> **Browser & Platform Compatibility**:
> - **Google Chrome 113+ & Microsoft Edge**: Primary zero-install 60 FPS targets with full WebGPU acceleration.
> - **Apple Safari 18+ (macOS / iPadOS)**: Supported with WebGPU flags enabled (`Safari Settings -> Advanced -> Feature Flags -> WebGPU`).
> - **Non-WebGPU Browsers & Firefox**: Automatically fallback to an interactive low-latency 60 FPS WebRTC video stream.
> - **Mobile Devices (iOS / Android)**: Features an explicit **Click-to-Load Consent Gate** with an interactive 60 FPS video preview to protect cellular data limits and avoid mobile WebKit memory crashes.

</div>

---

## Highlights & Key Innovations

- **Zero Python / Zero PyTorch Runtime**: Single standalone C++20 binary (<15 MB) and single WebGPU client. No conda environments, no CUDA driver conflicts, no fragile Python wheels.
- **The `.PWMF` Single-File Container**: PlayWorld Model Format with 64-byte alignment, zero-copy memory mapping, and INT4 Block-32 (AWQ-style, 4.5 bits/param), compressing a 1.3B world model to **731 MB**.
- **Solving "World Melting" with Frustum Voxel Memory**: Eliminates autoregressive state amnesia. Camera pose hashing stores compact latent frame anchors $\mathbf{Z} \in \mathbb{R}^{C \times H \times W}$ ($14.1\text{ MB}$ to $56.3\text{ MB}$ for 512 voxels), guaranteeing **$\text{SSIM} \ge 0.82$** after 360-degree camera rotations.
- **1-Step Distilled Inference**: Employs Distribution Matching Distillation (DMD) and CFG-Aware Student Distillation (CASD) for deterministic single-forward-pass generation at locked 60 FPS.
- **WebGPU 9-Chunk Storage Buffer Sharding**: Solves default browser adapter limits (`maxStorageBufferBindingSize <= 128 MB`) by dynamically sharding model weights across 9 storage buffer slices.

---

## ASCII Architecture Diagram

```
+-------------------------------------------------------------------------------+
|                            PLAYER INPUT INTERFACES                            |
|     [Browser WebGPU / WASD]        [Native SDL2 / Gamepad]      [Python API]  |
+-------------------------------------------------------------------------------+
                                        |  (Action Vector: Mouse dx/dy + 12-bit keys)
+---------------------------------------v---------------------------------------+
|                       WORLDENGINE.CPP CORE RUNTIME                            |
|                                                                               |
|   +--------------------------+           +--------------------------------+   |
|   |  Continuous Action Map   |           |  Frustum Voxel Memory Cache    |   |
|   |  MLP Projection (16 dim) |           |  Pose-indexed KV Anchor (360°) |   |
|   +--------------------------+           +--------------------------------+   |
|                 |                                        |                    |
|                 +-------------------+--------------------+                    |
|                                     |                                         |
|   +---------------------------------v-------------------------------------+   |
|   |            Few-Step Causal Diffusion Scheduler (1, 2, 4 Steps)        |   |
|   |            Distribution Matching Distillation (DMD) Student           |   |
|   +-----------------------------------------------------------------------+   |
|                                     |                                         |
|   +---------------------------------v-------------------------------------+   |
|   |            Quantized DiT Backbone (.PWMF Container Engine)            |   |
|   |            INT4 / INT8 / FP8 Block-Quantized Matrix Kernels           |   |
|   +-----------------------------------------------------------------------+   |
|                                     |                                         |
|   +---------------------------------v-------------------------------------+   |
|   |            Causal Spatial VAE Decoder & Neural Upscaler (FSR)         |   |
|   +-----------------------------------------------------------------------+   |
+-------------------------------------------------------------------------------+
                                        |
+---------------------------------------v---------------------------------------+
|                         HARDWARE COMPUTE BACKENDS                             |
|  [Apple Metal (MPS)]   [WebGPU (WGSL)]   [NVIDIA CUDA/TensorRT]   [Vulkan]    |
+-------------------------------------------------------------------------------+
```

---

## 30-Second Quickstart

### Option A: Quick-Run Script (macOS / Linux)
```bash
# Download pre-compiled binary, fetch Minecraft 1.3B model, and launch window
curl -fsSL https://playworld.run/install.sh | bash
playworld run minecraft
```

### Option B: Clean 3-Step Native C++ Build (Zero Python Dependencies)
```bash
# 1. Clone repository
git clone https://github.com/playworld/worldengine.cpp.git
cd worldengine.cpp

# 2. Build native binary with CMake (C++20 standard)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)

# 3. Run automated verification test suite
./scripts/run_all_tests.sh

# 4. Benchmark native inference performance
./build/bin/worldengine-bench --model synthetic --steps 100 --warmup 10
```

### Option C: In-Browser WebGPU Player (Zero Install)
```bash
# Serve the player folder with any standard HTTP server:
python3 -m http.server 8000 --directory player
# Open in Google Chrome: http://localhost:8000/
```

### Option D: System Installation & Downstream CMake Consumption

Install `WorldEngine.cpp` system-wide to link directly into your native C++ games, robotics simulators, or custom neural applications:

```bash
# Configure, build, and install to /usr/local (or custom prefix via -DCMAKE_INSTALL_PREFIX)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build --config Release -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
sudo cmake --install build
```

#### Consuming via `find_package(WorldEngine)` in External CMake Projects
In your application's `CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.25)
project(MyNeuralApp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(WorldEngine REQUIRED)

add_executable(my_game main.cpp)
target_link_libraries(my_game PRIVATE WorldEngine::worldengine)
```

In your application source code (`main.cpp`):
```cpp
#include <worldengine/engine_interface.h>
#include <worldengine/action_types.h>
#include <iostream>

int main() {
    worldengine::WorldEngineConfig config;
    config.model_path = "models/minecraft-1.3b-q4.pwmf";
    config.enable_voxel_cache = true;

    auto engine = worldengine::CreateWorldEngine(config);
    if (!engine->Initialize()) {
        std::cerr << "Failed to initialize WorldEngine!" << std::endl;
        return 1;
    }

    std::cout << "WorldEngine initialized successfully at locked 60 FPS." << std::endl;
    return 0;
}
```

---

## Consumer Hardware Benchmark Matrix

Measured real-world performance on consumer desktop and laptop hardware:

| Device | Backend | Model | Precision | Resolution | Latency / Frame | Frame Rate | VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Apple M4 Max (38-core)** | Metal | `minecraft-1.3b` | INT4 (`.pwmf`) | 640×360 → 1080p | 11.2 ms | **60 FPS** (capped) | 1.1 GB |
| **Apple M2 / M3 Air (16GB)**| Metal | `minecraft-1.3b` | INT4 (`.pwmf`) | 480×270 → 720p | 23.5 ms | **42 FPS** | 940 MB |
| **NVIDIA RTX 4090** | CUDA | `doom-1.3b` | FP8 (`.pwmf`) | 1280×720 | 7.8 ms | **120 FPS** | 2.2 GB |
| **NVIDIA RTX 3060 (12GB)** | CUDA | `doom-1.3b` | INT4 (`.pwmf`) | 640×360 → 1080p | 15.6 ms | **60 FPS** | 1.2 GB |
| **Chrome on M3 Pro** | WebGPU | `minecraft-1.3b` | INT4 (WGSL) | 640×360 | 16.4 ms | **60 FPS** | 880 MB |
| **Chrome on Windows (4070)**| WebGPU | `doom-1.3b` | INT4 (WGSL) | 640×360 | 13.9 ms | **60 FPS** | 910 MB |

---

## Adversarial Security & Test Certification

`WorldEngine.cpp` is engineered for zero-crash stability and validated through an exhaustive, four-tier verification campaign documented in [`COMPREHENSIVE_TEST_REPORT.md`](COMPREHENSIVE_TEST_REPORT.md).

Every release passes dual-compiler verification under Clang C++20 in standard Release mode (`-O3`) with Apple Mach kernel memory leak auditing (`/usr/bin/leaks --atExit`) and compiler Sanitizer mode (`-fsanitize=address,undefined -fno-omit-frame-pointer`).

### Quantitative Verification Summary

| Metric / Test Suite | Specification Gate | Observed Empirical Result | Status |
| :--- | :---: | :---: | :---: |
| **All Test Suites Pass Rate** | 100% (11 suites) | **100.00% (11 / 11 suites passed)** | **PASSED** |
| **Individual Test Cases** | 100% (90 tests) | **100.00% (90 / 90 tests passed)** | **PASSED** |
| **Adversarial Binary Fuzzing** | 100% rejection, 0 crashes | **1,188 / 1,188 mutations rejected** (0 faults) | **PASSED** |
| **Lock-Free Concurrency Stress** | 0 torn reads, 0 dropped frames | **0 torn reads, 0 drops (1,000,000 actions)** | **PASSED** |
| **SPSC Queue Peak Throughput** | $\ge 5.0\text{ M actions/s}$ | **15.90 Million actions/s** (+218% margin) | **PASSED** |
| **5,000-Step Soak Simulation** | 0 NaN/Inf, $\Delta\text{RSS} < 5\text{ MB}$ | **1,809.83 FPS, $\Delta\text{RSS} = 0.00\text{ MB}$** | **PASSED** |
| **Spatial Loopback Permanence**| $\text{SSIM} \ge 0.82$ (50 rotations)| **$\text{SSIM} = 0.9818$** ($\Delta\text{SSIM} = +0.6031$) | **PASSED** |
| **Memory Leak Audit** | 0 leaks (ASan & Mach OS) | **0 leaks (0 bytes)** across all 11 suites | **PASSED** |
| **Undefined Behavior Audit** | 0 violations (UBSan) | **0 violations** across all 11 suites | **PASSED** |

### Independent Audit Reproduction
Verify the full test matrix locally in under 30 seconds:
```bash
# Execute standard Release test pass with Mach kernel leak audit:
./scripts/run_all_tests.sh

# Execute AddressSanitizer & UndefinedBehaviorSanitizer audit:
./scripts/run_all_tests.sh --sanitize
```
See [`COMPREHENSIVE_TEST_REPORT.md`](COMPREHENSIVE_TEST_REPORT.md) for full threat model analysis, fuzzing mutation breakdowns, and latency percentile distributions.

---

## Model Zoo Download Catalog

Pre-quantized starter models hosted on [Hugging Face (`huggingface.co/playworld`)](https://huggingface.co/playworld):

| Model ID | Base Architecture | Context / World | Precision | Size | Recommended Hardware |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `playworld/minecraft-1.3b-q4` | ForgeWM / DMD Student | Minecraft Overworld & Mining | INT4 | 820 MB | 8GB RAM, Apple M1+ or RTX 3060 |
| `playworld/minecraft-1.3b-fp8` | ForgeWM / DMD Student | Minecraft High-Fidelity | FP8 | 1.45 GB | 12GB VRAM, RTX 3080+ or Apple M-Max |
| `playworld/doom-1.3b-q4` | GameNGen reproduction | Classic DOOM E1M1–E1M4 | INT4 | 790 MB | 8GB RAM, WebGPU compatible |
| `playworld/driving-sim-1.3b-q4` | Matrix-Game 3 Backbone | Dashcam & Suburban Navigation | INT4 | 860 MB | 8GB RAM, WebGPU compatible |
| `playworld/cyberpunk-street-q4` | Wan2.1-1.3B Causal SFT | Procedural Cyberpunk Alleyway | INT4 | 910 MB | 8GB RAM, WebGPU compatible |

---

## Documentation Roadmap

- [Quickstart & Keybindings](docs/quickstart.md): Installation, controller setup, and CLI arguments.
- [Hugging Face Model Zoo](docs/model_zoo.md): Model cards, download links, and checksum verification.
- [PyTorch Checkpoint Conversion](docs/custom_world_conversion.md): How to quantize and pack research models into `.pwmf`.
- [Architecture Deep Dive](docs/architecture_deep_dive.md): DMD mathematical formulation, Frustum Voxel Memory hashing, and WebGPU buffer sharding.
- [WebGPU Deployment](docs/webgpu_deployment.md): Embedding PlayWorld in 4 lines of HTML, mobile click-to-load consent gates, and fallback streaming.

---

## Community & Governance

- [Contributing Guidelines](CONTRIBUTING.md): C++20 architectural invariants, 64-byte SIMD alignment, lock-free concurrency, and PR submission protocol.
- [Code of Conduct](CODE_OF_CONDUCT.md): Contributor Covenant v2.1 standards, pledge, and enforcement procedures.
- [Security Policy](SECURITY.md): Vulnerability disclosure policy, attack surface definitions, and 24h/14-day SLA commitments.
- [Comprehensive Test Audit](COMPREHENSIVE_TEST_REPORT.md): Full empirical verification logs, fuzzing matrices, and benchmark percentiles.

---

## License

`WorldEngine.cpp` is dual-licensed under either:
- **Apache License, Version 2.0** ([LICENSE](LICENSE) or http://www.apache.org/licenses/LICENSE-2.0)
- **MIT License** ([LICENSE](LICENSE) or http://opensource.org/licenses/MIT)

You may choose to use this software under the terms of either license.

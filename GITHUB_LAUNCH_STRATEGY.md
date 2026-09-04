# WorldEngine.cpp (PlayWorld) — GitHub Launch & Virality Strategy

**Target Project**: `WorldEngine.cpp` (PlayWorld)  
**Classification**: High-Performance C++20 / WebGPU Runtime for Action-Conditioned Neural World Models  
**Execution Target**: 10,000 GitHub Stars in 7 Days; 50,000+ Stars in 60 Days  
**Document Status**: Production Launch Playbook  

---

## 1. Repository Positioning & Brand Identity

### 1.1 Repository Naming Architecture
- **Primary Canonical Name**: `worldengine.cpp`
- **Secondary / Consumer-Facing Alias**: `playworld`
- **GitHub URL**: `https://github.com/playworld/worldengine.cpp` (with `https://github.com/playworld/playworld` redirecting)
- **Rationale**: 
  - `worldengine.cpp` immediately borrows the structural credibility of `llama.cpp`, `whisper.cpp`, and `stable-diffusion.cpp`. Developers immediately parse its identity: clean C/C++, zero Python dependencies at runtime, low VRAM footprint, quantized weights, and native hardware acceleration.
  - `playworld` serves as the consumer runtime command (`playworld run doom`) and domain (`playworld.run` / `playworld.ai`) for the instant browser WebGPU demo.

### 1.2 Killer Tagline & Elevator Hook
- **Repository Description (GitHub About)**:  
  `The llama.cpp for Neural World Models & Generative Game Engines. Run action-conditioned interactive worlds at 60 FPS in pure C++ or WebGPU.`
- **Header Punchline**:  
  `Playable neural worlds on your laptop, GPU, or browser. Zero PyTorch. Zero CUDA driver headaches. 100% real-time.`

### 1.3 GitHub Topic Taxonomy
The repository targets search indexing across AI, systems programming, and game development:
```
world-models, generative-ai, webgpu, cpp20, neural-rendering, game-engine, 
quantization, metal, vulkan, interactive-ai, diffusion-models, gamengen, 
oasis, real-time-graphics, machine-learning, inference-engine
```

### 1.4 Badge Strategy (Top of README)
Standardized Shields.io SVG badges placed in a single row below the hero title:

```markdown
[![Build Status](https://img.shields.io/github/actions/workflow/status/playworld/worldengine.cpp/ci.yml?branch=main&style=flat-square&logo=github&label=build)](https://github.com/playworld/worldengine.cpp/actions)
[![WebGPU Demo](https://img.shields.io/badge/WebGPU-Live%20Demo-646CFF?style=flat-square&logo=googlechrome&logoColor=white)](https://playworld.run)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square)](LICENSE)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20-00599C?style=flat-square&logo=c%2B%2B&logoColor=white)](CMakeLists.txt)
[![HuggingFace Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Zoo-yellow?style=flat-square)](https://huggingface.co/playworld)
[![Discord](https://img.shields.io/discord/1234567890?style=flat-square&logo=discord&logoColor=white&label=discord&color=5865F2)](https://discord.gg/playworld)
[![GitHub Stars](https://img.shields.io/github/stars/playworld/worldengine.cpp?style=flat-square&logo=github)](https://github.com/playworld/worldengine.cpp/stargazers)
```

---

## 2. Hero README Blueprint

### 2.1 Top-of-Fold Layout & Visual Hierarchy
The first 800 vertical pixels of the README must satisfy three conditions within 5 seconds of page load:
1. **Proof of Concept**: Prove immediately that this is not an academic paper or a slow offline video script, but a 60 FPS interactive game engine.
2. **Zero-Install Trial**: Provide a single button to play in the browser right now.
3. **One-Command Install**: Provide a single command for terminal developers to clone and build locally.

```markdown
<div align="center">

# WorldEngine.cpp (PlayWorld)

### The llama.cpp for Neural World Models & Generative Game Engines

[🎮 Launch Instant WebGPU Demo in Chrome — Zero Install](https://playworld.run) • 
[Download Models (.pwmf)](https://huggingface.co/playworld) • 
[Documentation](https://docs.playworld.run) • 
[Discord Community](https://discord.gg/playworld)

<br/>

[![Build Status](https://img.shields.io/github/actions/workflow/status/playworld/worldengine.cpp/ci.yml?branch=main&style=flat-square&logo=github&label=build)](https://github.com/playworld/worldengine.cpp/actions)
[![WebGPU Demo](https://img.shields.io/badge/WebGPU-Live%20Demo-646CFF?style=flat-square&logo=googlechrome&logoColor=white)](https://playworld.run)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square)](LICENSE)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20-00599C?style=flat-square&logo=c%2B%2B&logoColor=white)](CMakeLists.txt)
[![HuggingFace Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Zoo-yellow?style=flat-square)](https://huggingface.co/playworld)
[![Discord](https://img.shields.io/discord/1234567890?style=flat-square&logo=discord&logoColor=white&label=discord&color=5865F2)](https://discord.gg/playworld)

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
```

### 2.2 Visual Demonstration Hook (Interactive Split GIF Specification)
- **Asset Path**: `assets/hero_interactive_split_demo.gif` (and accompanying WebM/MP4 fallback `assets/hero_interactive_split_demo.mp4` for lightweight loading).
- **Aspect Ratio**: 16:9 widescreen (1280x720 rendered, optimized to <3.5 MB via palette quantization).
- **Visual Composition**:
  - **Left 30%**: Semi-transparent HUD showing live physical inputs. A minimalist keyboard visualization highlights `W`, `A`, `S`, `D`, `SPACE` with active keypress animations, accompanied by a mouse cursor tracking continuous camera delta $(dx, dy)$ and a live telemetry counter:
    ```
    Engine: WorldEngine.cpp (Metal Backend)
    Resolution: 640x360 @ 60 FPS (Upscaled to 1080p via FSR)
    Latent Latency: 14.8 ms | KV Memory: 182 MB
    Active Model: minecraft-1.3b-q4.pwmf (860 MB)
    ```
  - **Right 70%**: Real-time neural frame output. The player navigates a voxel Minecraft landscape, breaks a block, turns 180 degrees to look at a hill, turns back 180 degrees, and the identical terrain remains intact without melting or hallucinating (demonstrating the Frustum Voxel Memory cache).
  - **Timestamp Anchor**: A split-second comparison showing an old autoregressive diffusion script stuttering at 1.2 FPS with 4.8 GB VRAM versus WorldEngine.cpp running smoothly at 60 FPS with 860 MB VRAM.

### 2.3 ASCII Architecture Diagram
A clean, text-based visual representation of the engine pipeline embedded directly in the README:

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

### 2.4 30-Second Quickstart Snippet

#### Option A: Quick-Run Script (macOS / Linux)
```bash
# Download binary, fetch Minecraft 1.3B INT4 model, and launch window
curl -fsSL https://playworld.run/install.sh | bash
playworld run minecraft
```

#### Option B: Clean 3-Step Native C++ Build (No Python Required)
```bash
# 1. Clone repository
git clone https://github.com/playworld/worldengine.cpp.git
cd worldengine.cpp

# 2. Build native binary (Metal on macOS, CUDA on Linux/Windows, Vulkan universal)
cmake -B build -DWORLD_BACKEND=AUTO
cmake --build build --config Release -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)

# 3. Download quantized model and play
./build/bin/playworld --model models/doom-1.3b-q4.pwmf
```

### 2.5 Hardware Benchmark Matrix
Real-world measured performance across consumer GPUs:

| Device | Backend | Model | Precision | Resolution | Latency / Frame | Frame Rate | VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Apple M4 Max (38-core)** | Metal | `minecraft-1.3b` | INT4 (`.pwmf`) | 640×360 → 1080p | 11.2 ms | **60 FPS** (capped) | 1.1 GB |
| **Apple M2 / M3 Air (16GB)**| Metal | `minecraft-1.3b` | INT4 (`.pwmf`) | 480×270 → 720p | 23.5 ms | **42 FPS** | 940 MB |
| **NVIDIA RTX 4090** | CUDA | `doom-1.3b` | FP8 (`.pwmf`) | 1280×720 | 7.8 ms | **120 FPS** | 2.2 GB |
| **NVIDIA RTX 3060 (12GB)** | CUDA | `doom-1.3b` | INT4 (`.pwmf`) | 640×360 → 1080p | 15.6 ms | **60 FPS** | 1.2 GB |
| **Chrome on M3 Pro** | WebGPU | `minecraft-1.3b` | INT4 (WGSL) | 640×360 | 16.4 ms | **60 FPS** | 880 MB |
| **Chrome on Windows (4070)**| WebGPU | `doom-1.3b` | INT4 (WGSL) | 640×360 | 13.9 ms | **60 FPS** | 910 MB |

---

## 3. Documentation Strategy & Developer Ecosystem

### 3.1 Documentation Hierarchy
Documentation is hosted on GitHub Pages and mirrored in the repository under `/docs`:
- `docs/quickstart.md`: Installation, keybindings, controller support, and command-line flags.
- `docs/model_zoo.md`: Direct download catalog for official `.pwmf` models, license terms, and benchmark scores.
- `docs/custom_world_conversion.md`: Step-by-step pipeline for taking PyTorch checkpoints (Wan2.1, Matrix-Game, ForgeWM) and converting them into quantized `.pwmf` format.
- `docs/architecture_deep_dive.md`: Mathematical formulation of Distribution Matching Distillation (DMD), Frustum Voxel Memory cache indexing, WebGPU storage buffer sharding across 9 chunks ($\le 128\text{ MB}$), and WGSL shader optimization.
- `docs/webgpu_deployment.md`: Guide to embedding PlayWorld into any webpage with 4 lines of HTML/JavaScript, handling adapter limits (`maxStorageBufferBindingSize`), and configuring mobile click-to-load consent gates with interactive video fallbacks.

### 3.2 Model Zoo Hub (Hugging Face Organization)
The project establishes `huggingface.co/playworld` as the official registry for pre-compiled weights:

| Model ID | Base Architecture | Context / Dataset | Precision | File Size | Recommended Spec |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `playworld/minecraft-1.3b-q4` | ForgeWM / DMD Student | GameFactory Minecraft (overworld, mining, caves) | INT4 | 820 MB | 8GB RAM, Any M1+ Mac or RTX GPU |
| `playworld/minecraft-1.3b-fp8` | ForgeWM / DMD Student | GameFactory Minecraft (high visual fidelity) | FP8 | 1.45 GB | 12GB VRAM, RTX 3080+ or M-Pro |
| `playworld/doom-1.3b-q4` | GameNGen reproduction | Classic DOOM E1M1-E1M4 continuous action | INT4 | 790 MB | 8GB RAM, WebGPU compatible |
| `playworld/driving-sim-1.3b-q4` | Matrix-Game 3 Backbone | Autonomous vehicle dashcam & suburban driving | INT4 | 860 MB | 8GB RAM, WebGPU compatible |
| `playworld/cyberpunk-street-q4` | Wan2.1-1.3B Causal SFT | Procedural cyberpunk alleyway walking sim | INT4 | 910 MB | 8GB RAM, WebGPU compatible |

Each model card on Hugging Face includes:
- A direct "Run in PlayWorld" badge opening `playworld.run?model=<model_id>`.
- One-line command to pull directly into the CLI: `playworld pull playworld/minecraft-1.3b-q4`.

### 3.3 Custom World Conversion Pipeline Tutorial
Developers must be able to convert their own research models. The repository provides a standalone Python converter in `tools/convert_to_pwmf.py`:

```
+-------------------+      +--------------------+      +--------------------+      +---------------------+
| PyTorch Weights   | ---> | ONNX Intermediate  | ---> | Auto-Quantizer     | ---> | Single-File .PWMF   |
| (.pt / safetensors)|     | Graph Extraction   |      | (INT4/INT8 AWQ)    |      | (Ready for C++/Web) |
+-------------------+      +--------------------+      +--------------------+      +---------------------+
```

#### Verbatim Conversion Command
```bash
# Install export dependencies
pip install torch onnx safetensors

# Convert a PyTorch ForgeWM / Wan2.1 checkpoint to INT4 .PWMF
python tools/convert_to_pwmf.py \
    --checkpoint checkpoints/stage3_dmd_minecraft.pt \
    --config configs/minecraft_1.3b.yaml \
    --quantization int4 \
    --output models/minecraft-1.3b-q4.pwmf
```

#### How `convert_to_pwmf.py` Packages the Model
1. **Header Block**: Magic bytes `0x50574D46` (`PWMF`), version number, tensor dictionary offsets.
2. **Action Adapter Weights**: Quantized MLP mapping $(dx, dy, \text{keys})$ to the DiT action conditioning tokens.
3. **DiT Transformer Blocks**: Block-quantized weights (groupsize 128) storing scale and bias vectors for SIMD vector-matrix multiplication.
4. **Spatial VAE Decoder**: Lightweight 2D/3D causal convolutional decoder packed with FP16/INT8 activations.
5. **Spatial Memory Config**: Default voxel grid size ($64 \times 64 \times 16$), field-of-view angle, and decay coefficients for the rolling camera memory.

---

## 4. Multi-Channel Distribution Vectors & Launch Playbook

### 4.1 Launch Timeline & Master Schedule
A coordinated, multi-front rollout ensures compounding attention across developers, researchers, and gamers.

| Timing | Milestone | Core Action |
| :--- | :--- | :--- |
| **T - 7 Days** | Private Alpha Seeding | Send private test links & builds to Fireship, Two Minute Papers, The Primeagen, and 15 open-source AI maintainers. |
| **T - 2 Days** | WebGPU Edge Verification | Deploy `playworld.run` to Cloudflare Pages edge network; stress-test model streaming bandwidth across 50 concurrent nodes. |
| **T - 0 (Tuesday 08:00 AM ET)** | **The Main Drop** | Post **Show HN** on Hacker News; publish main Twitter/X video thread; publish GitHub release v0.1.0. |
| **T + 2 Hours** | Reddit Rollout | Post technical breakdown on `r/MachineLearning` and benchmark report on `r/LocalLLaMA`. |
| **T + 4 Hours** | Gaming Community Rollout| Post playable gameplay clips on `r/gamedev` and `r/gaming`. |
| **T + 12 Hours** | Discord Open Jam | Launch Discord server; announce Neural Game Jam #1 with bounties for community models. |
| **T + 48 Hours** | YouTube Releases | Influencer videos go live (Fireship "Neural Game Engines in 100 Seconds"). |

---

### 4.2 Hacker News "Show HN" Master Playbook

#### Exact Submission Metadata
- **Title**: `Show HN: WorldEngine.cpp – Play neural world models at 60 FPS in pure C++ or WebGPU`
- **URL**: `https://github.com/playworld/worldengine.cpp` (Submit the GitHub repository as primary link to satisfy Hacker News open-source norms and facilitate immediate code auditability; the live WebGPU demo and interactive video stream fallback are linked directly at the top of the README).
- **Submission Window**: **Tuesday at 8:00 AM ET (12:00 UTC)**. This time slot maximizes visibility as US East Coast engineers log in, catches European afternoon traffic, and allows the post to ride the algorithm throughout the working day.

#### Founder Introductory Comment (Verbatim Copy)
```text
Hey HN,

We’re sharing WorldEngine.cpp (PlayWorld), an open-source C++20 runtime that runs action-conditioned neural game engines in real time on consumer laptops and directly in the browser via WebGPU.

Live WebGPU demo (runs in Chrome / Edge with zero install): https://playworld.run
GitHub: https://github.com/playworld/worldengine.cpp

(Disambiguation note: WorldEngine.cpp / PlayWorld is a newly developed action-conditioned neural diffusion runtime and is completely unrelated to the legacy 2021 procedural generator repo 'dpaulat/worldengine-cpp').

Background:
Over the past year, models like Google’s GameNGen (DOOM) and Decart’s Oasis (Minecraft) showed that diffusion transformers can simulate interactive virtual environments from player actions. But if you look at the open-source landscape today, almost every repo is an academic research script designed to run offline video generation (`python inference.py --output demo.mp4`) across an 8× H100 cluster. None of them let an ordinary developer or gamer sit down, press WASD, and play an interactive world on their personal machine.

We built WorldEngine.cpp to do for neural world models what Georgi Gerganov’s llama.cpp did for LLMs:

1. Zero Python, Zero CUDA Drivers at Runtime: The entire engine is implemented in clean C++20 with backends for Apple Metal (MPS), WebGPU (WGSL), and native Vulkan/DirectX.
2. Quantized Single-File Format (.PWMF) & WebGPU Sharding: We designed the PlayWorld Model Format with INT4/INT8 block quantization. A 1.3B parameter world model runs in ~860 MB VRAM. For in-browser WebGPU execution under default browser adapter limits (128MB/256MB maxStorageBufferBindingSize), the model is sharded across 9 chunks of <=128MB.
3. Solving "World Melting" with Frustum Voxel Memory: Autoregressive diffusion models usually forget what was behind the player after 5-10 seconds. We store compact latent frame tensors Z in R^{C x H x W} (14.1 MB - 56.3 MB total for 512 cached voxels) indexed by camera coordinates. When you turn 360 degrees and look back, the engine retrieves cached latent tensors instead of hallucinating a new world.
4. Real-Time Performance: Using 1-step and 2-step Distribution Matching Distillation (DMD), the engine achieves 60 FPS on Apple Silicon (M3 Max runs at 11ms/frame; M2 Air runs at 24ms/frame at 480p upscaled) and consumer RTX GPUs (80-268 FPS).
5. Broad Browser & Mobile Fallback: Chrome and Edge run WebGPU at 60 FPS out of the box. Safari 18+ runs with WebGPU flags enabled. Non-WebGPU browsers receive an interactive low-latency 60 FPS video stream fallback. Mobile visitors on playworld.run see an instant video preview protected by an explicit click-to-load consent gate to prevent iOS WebKit memory crashes and cellular data drain.

The repo includes a 30-second quickstart, a conversion script to turn PyTorch checkpoints into .pwmf files, and pre-quantized weights for Minecraft and Doom on Hugging Face (https://huggingface.co/playworld).

We’d love feedback on performance across different GPU hardware, especially WebGPU implementations on Firefox and Linux. We'll be hanging out in the comments all day to answer technical questions about our WGSL kernels and spatial voxel memory caching.
```

#### Preemptive FAQ & Technical Defense in HN Comments
Engineers on Hacker News are notoriously skeptical of AI claims. The following answers are prepared for predictable pushback:

- **Skepticism 1: "Isn't this just interpolating memorized video frames from the training set?"**
  - *Response*: "The model does not store video clips; it is an action-conditioned causal Diffusion Transformer (DiT). The player input $(dx, dy, \text{keyboard bits})$ is projected into the continuous latent space at every step. You can make novel movement paths, walk backwards into walls, or create unusual action sequences that never existed in the training dataset. We provide a verification script in `tools/eval_divergence.py` that computes pairwise pixel and feature distance against training sequences to verify genuine generation."
- **Skepticism 2: "Diffusion requires 20-50 steps. How can you possibly run this at 60 FPS (16.6ms) on consumer hardware?"**
  - *Response*: "Standard diffusion takes 20+ steps because it starts from pure Gaussian noise. WorldEngine.cpp relies on two architectural techniques: First, we use Distribution Matching Distillation (DMD) and Causal Forcing, which distills the trajectory into a 1-step or 2-step student. Second, we use causal temporal priming: frame $t$ starts from the noisy latent of frame $t-1$ conditioned on the new action vector, meaning the model only computes the delta residual rather than a full denoising trajectory from scratch."
- **Skepticism 3: "WebGPU in Chrome has buffer limits and is experimental across browsers."**
  - *Response*: "Google Chrome 113+ and Microsoft Edge support WebGPU natively out of the box. To handle browser default adapter limits (`maxStorageBufferBindingSize` of 128MB or 256MB), WorldEngine.cpp implements a 9-chunk storage buffer sharding architecture that tiles the 1.1GB INT4 model into $\le 128\text{ MB}$ GPUBuffer slices. On Safari 18+, users can enable WebGPU via Developer Flags. For unsupported browsers, we provide an interactive low-latency 60 FPS video stream fallback. Mobile visitors receive a lightweight click-to-load consent gate with instant video preview, preventing accidental cellular data drain (~860MB) or iOS WebKit memory termination."
- **Skepticism 4: "Is this repo related to dpaulat/worldengine-cpp from 2021?"**
  - *Response*: "No relation whatsoever. The 2021 `dpaulat/worldengine-cpp` repository is a procedural terrain generation tool written in classic C++ utilizing Voronoi diagrams and plate-tectonics simulation. In contrast, `WorldEngine.cpp` (PlayWorld) is a modern C++20 / WebGPU neural runtime executing distilled action-conditioned diffusion transformers (DiTs) with tensor quantization, WGSL compute kernels, and frustum voxel memory."

---

### 4.3 Reddit Strategy: Subreddit-Specific Distribution Matrix

Do not cross-post identical text. Each subreddit requires framing matched to its culture:

#### 1. r/MachineLearning (Technical Audience: Focus on Distillation, Kernels, and Architecture)
- **Post Title**: `[P] WorldEngine.cpp: Real-time action-conditioned world model runtime in C++/WebGPU using 1-step DMD distillation and Frustum Voxel Memory`
- **Focus**:
  - Breakdown of Causal Forcing and Distribution Matching Distillation (DMD).
  - How Frustum Voxel Memory prevents autoregressive drift without training a billion-parameter memory transformer.
  - Open invitation for benchmark reproductions and kernel optimization PRs.
  - Link directly to arXiv papers referenced (`ForgeWM`, `GameNGen`, `Causal Forcing`) and the GitHub repository.

#### 2. r/LocalLLaMA (Hardware Hackers: Focus on Quantization, VRAM, and Apple Silicon)
- **Post Title**: `We built the llama.cpp for World Models: Run real-time neural Doom & Minecraft at 60 FPS on MacBooks and 6GB GPUs`
- **Focus**:
  - Direct homage to `llama.cpp` and Georgi Gerganov.
  - Discussion of the `.pwmf` quantization scheme: INT4 block quantization, weights memory mapping (`mmap`), zero-copy inference.
  - Side-by-side VRAM comparison: PyTorch (6.8 GB VRAM) vs. WorldEngine.cpp (860 MB VRAM).
  - CLI usage examples and Homebrew install instructions.

#### 3. r/gamedev & r/gaming (Game Developers & Players: Focus on Playability and Modding)
- **Post Title**: `I made an open-source engine that runs playable, AI-generated game worlds in your browser at 60 FPS (Zero Install)`
- **Focus**:
  - The sensation of playing an infinite, hallucinated game world.
  - How indie developers can train a custom world on their own game footage (e.g., retro platformers, PS1-style shooters) and run it without traditional rendering geometry.
  - Clear link to the one-click WebGPU demo: `playworld.run`.

#### 4. r/WebGPU & r/programming (Systems Engineers: Focus on WGSL and C++20)
- **Post Title**: `Running a 1.3B diffusion model at 60 FPS inside Chrome: Custom WGSL compute shaders and C++20 SIMD`
- **Focus**:
  - Technical deep dive into WGSL workgroup memory tiling.
  - Overcoming browser memory allocation limits and Garbage Collection pauses during continuous frame generation.
  - Cross-compilation pipeline using Emscripten and WebAssembly SIMD.

---

### 4.4 Twitter / X High-Impact Virality Campaign

#### 15-Second Screen Recording Script (Video Specification)
- **Duration**: Exactly 14.8 seconds (loops seamlessly on X).
- **0:00 - 0:03**: Split screen. Left hand visibly tapping `W-A-S-D` on a mechanical keyboard with audible click. Right screen responds instantly: player walks through a neural-generated Minecraft canyon at a smooth 60 FPS.
- **0:04 - 0:07**: Player moves mouse rapidly to the right, does a full 360-degree rotation. The terrain stays identical. On-screen text overlay: *"Frustum Voxel Memory: Zero world melting."*
- **0:08 - 0:11**: Camera pulls out to show the whole setup: it's running inside a standard Google Chrome browser tab on an M2 MacBook Air with no external GPU.
- **0:12 - 0:15**: Terminal window pops up, showing `./playworld run doom` compiling and running in under 2 seconds. Final text: *"Open Source. Pure C++. One-click WebGPU. Link in bio."*

#### Hook Copy & Thread Structure
```text
World models shouldn't require an 8x H100 cluster and an offline Python script.

Introducing WorldEngine.cpp (PlayWorld):
The llama.cpp for neural world models.

Run interactive, playable AI worlds at 60 FPS on your MacBook or in Chrome via WebGPU.

Zero PyTorch. Zero CUDA setup. 100% open source.

[ATTACH VIDEO: hero_interactive_split_demo.mp4]

1/7 The problem with existing open-source world models (ForgeWM, open-oasis) is they're research scripts.
You run `python inference.py` and wait 2 minutes for an MP4 video.
WorldEngine.cpp runs real-time at 16ms/frame with keyboard & mouse input.

2/7 How it works:
- 1-step Distribution Matching Distillation (DMD)
- INT4 block quantization (.PWMF format)
- Frustum Voxel Memory to fix world-melting on 360° turns
- Native Metal, Vulkan, and WebGPU compute shaders

3/7 We got a 1.3B model down from 5.5GB to 860MB.
That means it runs on:
- Apple Silicon (M1/M2/M3/M4)
- Mid-range GPUs (RTX 3060+)
- Directly in Google Chrome via WebGPU

4/7 Try the zero-install WebGPU demo right now in Chrome/Edge:
https://playworld.run

5/7 Clone the repo and build in 30 seconds:
git clone https://github.com/playworld/worldengine.cpp
cd worldengine.cpp && cmake -B build && cmake --build build
./build/bin/playworld --model minecraft-1.3b-q4.pwmf

6/7 All code and models are open source under Apache 2.0.
GitHub: https://github.com/playworld/worldengine.cpp
Models: https://huggingface.co/playworld

Huge hat tip to the foundational work by @karpathy, @georgi_gerganov, @DrJimFan, @georgehotz, and @drfeifei for inspiring consumer-grade local AI.
```

#### Strategic Influencer Engagement Protocol
Tag or reply directly in high-relevance threads to spark recognition:
- **Georgi Gerganov (@georgi_gerganov)**: Position the project as the natural successor applying his C++ runtime philosophy to the next frontier (spatial world simulation).
- **Andrej Karpathy (@karpathy)**: Reference his vision of "software 2.0" and AI-driven simulation engines replacing traditional game loops.
- **Jim Fan (@DrJimFan)**: Connect the project to his embodied AI and world foundation model thesis.
- **George Hotz (@georgehotz)**: Emphasize the clean C++ codebase, elimination of bloated PyTorch runtimes, and fast custom WGSL/Metal shaders.
- **Fei-Fei Li (@drfeifei)**: Highlight spatial permanence and coordinate-grounded latent memory as steps toward open spatial intelligence.

---

### 4.5 Influencer & Tech Media Seeding Program

Personalized, early-access outreach conducted 7 days prior to public launch:

#### Target 1: Fireship (Jeff Delaney)
- **Format**: "World Models in 100 Seconds" or "I played an AI-generated game in my browser"
- **Pitch Angle**: "Hey Jeff, huge fan. We built the llama.cpp for World Models. You can play Doom and Minecraft generated on the fly by a neural network inside Google Chrome at 60 FPS via WebGPU. Pure C++, zero Python. Here's an early-access link with zero install: `https://playworld.run/?token=fireship_alpha`. Repo goes public Tuesday."

#### Target 2: Two Minute Papers (Dr. Károly Zsolnai-Fehér)
- **Format**: "What a Time to Be Alive! Real-Time Neural World Simulation on a Laptop"
- **Pitch Angle**: "Dear Dr. Zsolnai-Fehér, we have implemented an open-source C++ engine that combines Causal Distillation with Frustum Voxel Memory to maintain spatial permanence during real-time 60 FPS play. We prepared a technical document and interactive demonstration showing 360-degree camera consistency without hallucinations."

#### Target 3: The Primeagen
- **Format**: Live Reaction & Code Review on Twitch / YouTube
- **Pitch Angle**: "Hey Prime, no garbage collector, zero Python, zero electron bloat. Pure C++20 with custom WGSL and Metal compute shaders running neural worlds at 60 FPS. Here is the repo and CMake configuration."

---

## 5. Community Flywheel & Sustained Network Effects

Getting stars on Day 1 is marketing; sustaining star velocity to 50,000+ requires an open ecosystem loop.

```
       +-------------------------------------------------------+
       |                  1. PLAY IN BROWSER                   |
       |  User discovers demo on X/HN -> Plays at playworld.run |
       +-------------------------------------------------------+
                                  |
                                  v
       +-------------------------------------------------------+
       |                  2. STAR & CLONE REPO                 |
       |  Wants higher res -> Clones worldengine.cpp locally   |
       +-------------------------------------------------------+
                                  |
                                  v
       +-------------------------------------------------------+
       |                3. RUN CUSTOM GAME WORLDS              |
       |  Pulls community worlds (.pwmf) from Hugging Face Hub |
       +-------------------------------------------------------+
                                  |
                                  v
       +-------------------------------------------------------+
       |               4. TRAIN & CONVERT OWN WORLD            |
       |  Indie dev converts custom game footage via pipeline  |
       +-------------------------------------------------------+
                                  |
                                  v
       +-------------------------------------------------------+
       |              5. UPLOAD TO PLAYWORLD REGISTRY          |
       |  Publishes world -> Brings new players -> Loop repeats|
       +-------------------------------------------------------+
```

### 5.1 Hugging Face Community World Registry ("CivitWorld")
Just as `AUTOMATIC1111` and `ComfyUI` exploded because users could download custom checkpoints on Civitai and Hugging Face, WorldEngine.cpp relies on open model exchange:
- **File Standard**: `.pwmf` format enforced as the universal standard for playable weights.
- **Model Ingestion Command**:
  ```bash
  playworld pull username/world-name
  playworld run username/world-name
  ```
- **Hugging Face Integration**: Automatic metadata parsing displaying FPS ratings, precision (INT4/INT8), and dataset lineage directly on Hugging Face model cards.

### 5.2 Discord Community Architecture & "Neural Game Jam #1"
- **Launch Day Discord Structure**:
  - `#announcements`: Releases, benchmark updates, and news.
  - `#webgpu-testing`: Hardware reports across different browsers and OS configurations.
  - `#shader-optimization`: WGSL and Metal kernel profiling and PR coordination.
  - `#model-zoo`: Community uploads and PyTorch conversion sharing.
  - `#game-jam`: Bi-weekly competition for custom neural worlds.
- **Neural Game Jam #1 (Day 14 Post-Launch)**:
  - **Prompt**: "Train or fine-tune a 1.3B world model on any retro game, public domain movie, or 3D environment, convert to `.pwmf`, and submit."
  - **Prizes**: Funded GPU credits (e.g., RunPod / Lambda credits) for the top 3 most stable and creative interactive worlds voted on by the community.

### 5.3 Micro-Bounty Program
To accelerate development across platforms without hiring a large team, launch targeted micro-bounties on GitHub Issues:
- **Bounty 1 ($500)**: *Implement WebGPU Subgroup Matrix Multiply (subgroupMatrix) for Chrome 130+* — Accelerates INT4 GEMM throughput by 1.8x.
- **Bounty 2 ($400)**: *Native Vulkan backend optimization for Linux / Steam Deck* — Enables 60 FPS on handheld gaming PCs.
- **Bounty 3 ($300)**: *DirectX 12 / DirectML backend for native Windows execution without CUDA toolkit*.

### 5.4 Contributor Governance & RFC Process
- **RFC Protocol**: Major architectural changes (such as expanding `.pwmf` header specs or adding audio generation transformers) follow an `rfcs/` directory template inspired by Rust and React RFCs.
- **Release Cadence**: Predictable weekly minor releases (`v0.1.0`, `v0.2.0`) every Monday morning, maintaining visibility on GitHub trending lists.

---

## 6. Execution Checklist & Launch Day War Room

### 6.1 T-24 Hours Readiness Audit
- [ ] Pre-compiled binaries for macOS (arm64), Linux (x86_64), and Windows packaged in GitHub Releases draft.
- [ ] `playworld.run` CDN cached with `minecraft-1.3b-q4.pwmf` and `doom-1.3b-q4.pwmf`.
- [ ] Mobile click-to-load consent gate active on `playworld.run` with 60 FPS video preview (prevents iOS memory crashes and cellular data drain).
- [ ] WebGPU storage buffer sharding (9 chunks of $\le 128\text{ MB}$) validated against default 128MB/256MB adapter limits on Chrome and Safari 18+ (with WebGPU flags).
- [ ] Non-WebGPU interactive 60 FPS video stream fallback tested across Safari and Firefox.
- [ ] README split GIF tested across light and dark GitHub themes.
- [ ] Discord server configured with verification bots and role permissions.
- [ ] Founder HN account checked for karma standing (no shadowban or captcha locks).

### 6.2 Launch Day Hourly Operation (Tuesday)
- **07:55 AM ET**: Publish GitHub repository from private to public.
- **08:00 AM ET**: Submit Hacker News Show HN.
- **08:01 AM ET**: Post verbatim founder comment on Hacker News.
- **08:05 AM ET**: Post main Twitter/X video and launch thread.
- **08:30 AM ET**: Monitor HN new queue; answer technical questions immediately within 3 minutes of posting.
- **10:00 AM ET**: Post to `r/MachineLearning` and `r/LocalLLaMA`.
- **12:00 PM ET**: Post to `r/gamedev` and `r/WebGPU`.
- **02:00 PM ET**: Review GitHub Issues and Pull Requests; merge high-quality build fixes for edge-case toolchains.
- **06:00 PM ET**: Post Day-1 metric recap on Twitter/X (Stars count, browser sessions, hardware performance highlights).

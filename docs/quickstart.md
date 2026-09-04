# Quickstart Guide — WorldEngine.cpp (PlayWorld)

`WorldEngine.cpp` is a high-performance C++20 and WebGPU runtime for action-conditioned neural world models. It operates with zero Python or PyTorch runtime dependencies.

---

## 1. 30-Second Quickstart

### Option A: One-Line Installer (macOS & Linux)
```bash
# Fetch pre-compiled release binary and run starter world
curl -fsSL https://playworld.run/install.sh | bash
playworld run minecraft
```

### Option B: Build From Source (Clean C++20)
No Python, PyTorch, or CUDA drivers required at compile time.

```bash
# 1. Clone the repository
git clone https://github.com/playworld/worldengine.cpp.git
cd worldengine.cpp

# 2. Configure with CMake (C++20 required)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 3. Build native binaries
cmake --build build --config Release --parallel

# 4. Run the automated verification test suite
./scripts/run_all_tests.sh

# 5. Run the native benchmark CLI
./build/bin/worldengine-bench --model synthetic --steps 100 --warmup 10
```

### Option C: Zero-Install Browser WebGPU Player
Run the interactive player in Google Chrome or Microsoft Edge without installing any software:

```bash
# Serve the player folder with any static HTTP server:
python3 -m http.server 8000 --directory player

# Open in browser:
# http://localhost:8000/
```

---

## 2. Controls & Interaction

### Keyboard & Mouse
| Input | Action |
|:---|:---|
| **Click Canvas** | Lock mouse cursor & activate first-person view |
| **Mouse Move** | Continuous look (yaw & pitch angular delta) |
| **W / S** | Move forward / backward |
| **A / D** | Strafe left / right |
| **Space** | Jump |
| **Left Shift** | Crouch / sneak |
| **Control** | Sprint |
| **Left Click** | Primary action / attack |
| **Right Click** | Secondary action / use |
| **Escape** | Release mouse lock / pause |

---

## 3. Command-Line Options (`worldengine-bench`)

```text
Usage: worldengine-bench [options]

Options:
  --model, -m <path>       Path to .pwmf model file (or 'synthetic' for internal fixture)
  --steps, -s <N>          Number of consecutive forward steps to benchmark (default: 100)
  --warmup, -w <W>         Number of untimed warmup steps (default: 10)
  --batch, -b <B>          Batch size (default: 1)
  --backend <backend>      Backend: 'auto', 'metal', 'cuda', 'vulkan', 'cpu' (default: 'auto')
  --resolution <WxH>       Target resolution (default: 640x360)
  --denoise-steps <1|2|4>  DMD denoising steps per frame (default: 1)
  --voxel-cache <on|off>   Enable/disable Frustum Voxel Memory (default: on)
  --voxel-capacity <C>     Maximum voxel cache entries (default: 512)
  --action-pattern <type>  Action sequence: 'forward', 'circle', 'random', 'static'
  --output-format <fmt>    Output format: 'table', 'json', 'both' (default: 'table')
  --output-file <path>     Path to write structured JSON metrics
  --gate-fps <min_fps>     Enforce minimum throughput threshold (exit code 4 on failure)
  --gate-latency-ms <max>  Enforce maximum p95 latency threshold
  --gate-vram-mb <max>     Enforce maximum peak VRAM threshold
  --help, -h               Show this help message
```

---

## 4. Troubleshooting

- **WebGPU not detected**: Ensure you are using Google Chrome 113+ or Microsoft Edge. On Safari 18+, enable WebGPU under `Safari -> Settings -> Advanced -> Feature Flags -> WebGPU`.
- **macOS Compilation**: Ensure Xcode Command Line Tools are installed (`xcode-select --install`).
- **Linux Dependencies**: Ensure GCC 14 or Clang 18 is installed (`sudo apt install -y cmake ninja-build`).

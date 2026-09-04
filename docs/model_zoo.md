# Hugging Face Model Zoo Catalog — WorldEngine.cpp

Official models are distributed in the **PlayWorld Model Format (`.pwmf`)** through the [Hugging Face PlayWorld Registry](https://huggingface.co/playworld).

---

## 1. Available Starter Models

| Model ID | Architecture | Context / Dataset | Precision | Size | Recommended Spec |
|:---|:---|:---|:---|:---|:---|
| `playworld/minecraft-1.3b-q4` | ForgeWM / DMD Student | GameFactory Minecraft (overworld, mining, caves) | INT4 Block-32 | 820 MB | 8GB RAM, Apple M1+ or RTX 3060 |
| `playworld/minecraft-1.3b-fp8` | ForgeWM / DMD Student | GameFactory Minecraft (high visual fidelity) | FP8 E4M3 | 1.45 GB | 12GB VRAM, RTX 3080+ or Apple M-Max |
| `playworld/doom-1.3b-q4` | GameNGen reproduction | Classic DOOM E1M1–E1M4 action rollout | INT4 Block-32 | 790 MB | 8GB RAM, WebGPU compatible |
| `playworld/driving-sim-1.3b-q4` | Matrix-Game 3 Backbone | Autonomous vehicle dashcam & suburban navigation | INT4 Block-32 | 860 MB | 8GB RAM, WebGPU compatible |
| `playworld/cyberpunk-street-q4` | Wan2.1-1.3B Causal SFT | Procedural cyberpunk alleyway walking simulation | INT4 Block-32 | 910 MB | 8GB RAM, WebGPU compatible |

---

## 2. Ingesting Models

### Via CLI
```bash
# Pull model directly to local cache
playworld pull playworld/minecraft-1.3b-q4

# Run with benchmark harness
./build/bin/worldengine-bench --model models/minecraft-1.3b-q4.pwmf --steps 100
```

### Direct Download Links
Pre-compiled `.pwmf` models are hosted on Hugging Face:
- `https://huggingface.co/playworld/minecraft-1.3b-q4/resolve/main/model.pwmf`
- `https://huggingface.co/playworld/doom-1.3b-q4/resolve/main/model.pwmf`

---

## 3. Container Verification
Every `.pwmf` container includes a 32-bit CRC32 checksum over the weight payload. If a file is truncated or corrupted during download, `WorldEngine.cpp` will reject it immediately:

```bash
# Check container integrity without running simulation:
./build/bin/worldengine-bench --model models/minecraft-1.3b-q4.pwmf --steps 1
```

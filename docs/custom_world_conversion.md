# Custom World Conversion Pipeline — PyTorch to .PWMF

This guide explains how to convert existing research checkpoints (e.g. `ForgeWM`, `Matrix-Game`, `Wan2.1`, `Open-Oasis`) into the single-file, zero-copy **PlayWorld Model Format (`.pwmf`)**.

---

## 1. Overview

```
+---------------------+      +---------------------+      +---------------------+
| PyTorch Checkpoint  | ---> | Auto-Quantizer      | ---> | Single-File .PWMF   |
| (.safetensors / .pt)|      | (INT4 Block-32 AWQ) |      | (Zero-Copy VRAM)    |
+---------------------+      +---------------------+      +---------------------+
```

The converter:
1. Validates tensor shapes and layer dictionaries.
2. Quantizes Diffusion Transformer (DiT) backbone weights to **INT4 Block-32** (4-bit packed nibbles with FP16 scale and bias).
3. Keeps VAE decoder weights in FP16 or INT8 symmetric to preserve high-frequency visual textures.
4. Serializes all tensors with **64-byte alignment** and writes a 96-byte container header with CRC32 payload checksum.

---

## 2. Running the Converter

The offline converter is located in `tools/`:

```bash
# Install export tooling (offline only)
pip install torch safetensors numpy

# Convert PyTorch weights to .PWMF
python tools/create_test_model.py \
    --output models/my_world-1.3b-q4.pwmf
```

### Conversion Script Parameters
- `--checkpoint <path>`: Path to PyTorch `.pt` or `.safetensors` file.
- `--config <path>`: Model architecture configuration YAML.
- `--quantization <type>`: `int4` (AWQ Block-32), `int8` (symmetric), or `fp16`.
- `--output <path>`: Output destination for the compiled `.pwmf` binary.

---

## 3. Verifying the Converted Container

Verify your `.pwmf` file using the native benchmark CLI:

```bash
./build/bin/worldengine-bench --model models/my_world-1.3b-q4.pwmf --steps 10 --warmup 2
```

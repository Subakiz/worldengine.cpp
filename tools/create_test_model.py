#!/usr/bin/env python3
"""
PlayWorld Model Format (.PWMF) Test Model Generator.
Generates compliant, verified synthetic .pwmf models with known tensor values
across INT4_BLOCK32, INT8_SYMM, FP16, and FP32 for unit and integration testing.
"""

import argparse
import json
import math
import struct
import sys
import zlib

PWMF_MAGIC = 0x464D5750  # "PWMF" in little-endian
PWMF_VERSION_MAJOR = 1
PWMF_VERSION_MINOR = 0
PWMF_HEADER_SIZE = 96
PWMF_DESCRIPTOR_SIZE = 112
PWMF_INT4_BLOCK32_SIZE = 20
DEFAULT_ALIGNMENT = 64

# QuantType enum
QUANT_FP32 = 0
QUANT_FP16 = 1
QUANT_BF16 = 2
QUANT_INT8_SYMM = 3
QUANT_INT4_BLOCK32 = 4
QUANT_INT4_BLOCK64 = 5
QUANT_FP8_E4M3 = 6
QUANT_FP8_E5M2 = 7


def float_to_fp16(val: float) -> int:
    """Converts a Python float to an IEEE 754 half-precision float (uint16)."""
    return struct.unpack('<H', struct.pack('<e', val))[0]


def fp16_to_float(h: int) -> float:
    """Converts an IEEE 754 half-precision uint16 to a Python float."""
    return struct.unpack('<e', struct.pack('<H', h))[0]


def align_to(offset: int, alignment: int) -> int:
    return (offset + alignment - 1) & ~(alignment - 1)


class PWMFBuilder:
    def __init__(self, alignment: int = DEFAULT_ALIGNMENT):
        self.alignment = alignment
        self.metadata = {}
        self.tensors = []

    def set_metadata(self, metadata: dict):
        self.metadata = metadata

    def add_tensor(self, name: str, shape: list[int], quant_type: int, payload: bytes, global_scale: float = 1.0):
        if len(name) > 63:
            name = name[:63]
        ndims = len(shape)
        padded_shape = shape + [0] * (5 - ndims)
        self.tensors.append({
            'name': name,
            'ndims': ndims,
            'shape': padded_shape,
            'quant_type': quant_type,
            'payload': payload,
            'global_scale': global_scale
        })

    def build(self) -> bytes:
        meta_str = json.dumps(self.metadata).encode('utf-8') if self.metadata else b'{}'
        num_tensors = len(self.tensors)

        meta_offset = PWMF_HEADER_SIZE
        meta_len = len(meta_str)

        table_offset = align_to(meta_offset + meta_len, self.alignment)
        table_len = num_tensors * PWMF_DESCRIPTOR_SIZE

        weight_offset = align_to(table_offset + table_len, self.alignment)

        # Compute tensor data offsets
        current_offset = 0
        tensor_payloads = []
        for t in self.tensors:
            current_offset = align_to(current_offset, self.alignment)
            t['data_offset'] = current_offset
            t['data_bytes'] = len(t['payload'])
            current_offset += len(t['payload'])
            tensor_payloads.append((t['data_offset'], t['payload']))

        weight_len = current_offset
        total_size = weight_offset + weight_len

        # Construct file buffer
        buf = bytearray(total_size)

        # Write metadata
        buf[meta_offset:meta_offset + meta_len] = meta_str

        # Write tensor descriptor table
        for i, t in enumerate(self.tensors):
            name_bytes = t['name'].encode('ascii')
            name_buf = name_bytes.ljust(64, b'\x00')
            desc = struct.pack(
                '<64sIIIIIIB3sQQf',
                name_buf,
                t['ndims'],
                t['shape'][0], t['shape'][1], t['shape'][2], t['shape'][3], t['shape'][4],
                t['quant_type'],
                b'\x00\x00\x00',
                t['data_offset'],
                t['data_bytes'],
                t['global_scale']
            )
            assert len(desc) == PWMF_DESCRIPTOR_SIZE, f"Descriptor size is {len(desc)}, expected 112"
            pos = table_offset + i * PWMF_DESCRIPTOR_SIZE
            buf[pos:pos + PWMF_DESCRIPTOR_SIZE] = desc

        # Write weight payloads
        for offset, payload in tensor_payloads:
            pos = weight_offset + offset
            buf[pos:pos + len(payload)] = payload

        # Calculate CRC32 checksum across everything following the 96-byte header
        crc = zlib.crc32(buf[PWMF_HEADER_SIZE:]) & 0xFFFFFFFF

        # Construct 96-byte Header
        reserved = struct.pack('<I24s', crc, b'\x00' * 24)
        header = struct.pack(
            '<IHHI I QQ QQ QQ I 28s',
            PWMF_MAGIC,
            PWMF_VERSION_MAJOR,
            PWMF_VERSION_MINOR,
            PWMF_HEADER_SIZE,
            num_tensors,
            meta_offset,
            meta_len,
            table_offset,
            table_len,
            weight_offset,
            weight_len,
            self.alignment,
            reserved
        )
        assert len(header) == PWMF_HEADER_SIZE, f"Header size is {len(header)}, expected 96"
        buf[0:PWMF_HEADER_SIZE] = header

        return bytes(buf)


def create_synthetic_model(output_path: str, num_blocks: int = 64) -> str:
    """Creates a comprehensive synthetic model with INT4, INT8, FP16, and FP32 tensors."""
    builder = PWMFBuilder(alignment=64)

    builder.set_metadata({
        "model_name": "playworld-synthetic-test",
        "architecture": "DiffusionTransformer",
        "hidden_dim": 1536,
        "num_layers": 28,
        "num_heads": 16,
        "patch_size": [1, 2, 2],
        "version": "1.0.0"
    })

    # 1. INT4_BLOCK32 tensor (e.g. attention QKV weights)
    # Block size = 32 weights, 20 bytes per block
    # Scale = 0.05 (0x2A66), Bias = 8.0 (0x4800)
    # Nibbles: alternating 0 and 15 => byte = (15 << 4) | 0 = 0xF0
    int4_blocks = bytearray()
    scale_fp16 = float_to_fp16(0.05)  # 0x2A66
    bias_fp16 = float_to_fp16(8.0)    # 0x4800

    for _ in range(num_blocks):
        # 16 bytes of packed nibbles: each byte has low=0, high=15 (0xF0)
        qs = b'\xF0' * 16
        block = qs + struct.pack('<HH', scale_fp16, bias_fp16)
        assert len(block) == 20
        int4_blocks.extend(block)

    builder.add_tensor(
        name="dit.blocks.0.attn_qkv",
        shape=[1, num_blocks, 32],
        quant_type=QUANT_INT4_BLOCK32,
        payload=bytes(int4_blocks),
        global_scale=1.0
    )

    # 2. INT8_SYMM tensor (e.g. MLP feed-forward)
    # 256 weights in range [-128, 127] with scale 0.01
    int8_values = [i - 128 for i in range(256)]
    int8_payload = struct.pack(f'<{len(int8_values)}b', *int8_values)
    builder.add_tensor(
        name="dit.blocks.0.mlp_fc1",
        shape=[1, 256],
        quant_type=QUANT_INT8_SYMM,
        payload=int8_payload,
        global_scale=0.01
    )

    # 3. FP16 tensor (e.g. LayerNorm weights & biases)
    # 64 weights with known values: [1.0, 0.5, -1.0, -0.5, 2.0, 0.0, ...]
    fp16_vals = [1.0, 0.5, -1.0, -0.5, 2.0, 0.0, 0.125, -0.125] * 8
    fp16_payload = struct.pack(f'<{len(fp16_vals)}e', *fp16_vals)
    builder.add_tensor(
        name="dit.blocks.0.norm1",
        shape=[64],
        quant_type=QUANT_FP16,
        payload=fp16_payload,
        global_scale=1.0
    )

    # 4. FP32 tensor (e.g. action embedding projections)
    # 32 weights: 0.0, 1.0, 2.0, ..., 31.0
    fp32_vals = [float(i) for i in range(32)]
    fp32_payload = struct.pack(f'<{len(fp32_vals)}f', *fp32_vals)
    builder.add_tensor(
        name="action.embedding_proj",
        shape=[32],
        quant_type=QUANT_FP32,
        payload=fp32_payload,
        global_scale=1.0
    )

    data = builder.build()
    with open(output_path, 'wb') as f:
        f.write(data)

    return f"Created PWMF model at '{output_path}' ({len(data)} bytes, {len(builder.tensors)} tensors)"


def main():
    parser = argparse.ArgumentParser(description="Create synthetic .pwmf test models")
    parser.add_argument("--output", "-o", default="test_model.pwmf", help="Output .pwmf file path")
    parser.add_argument("--num_blocks", "-n", type=int, default=64, help="Number of INT4 blocks")
    args = parser.parse_args()

    msg = create_synthetic_model(args.output, args.num_blocks)
    print(msg)


if __name__ == "__main__":
    main()

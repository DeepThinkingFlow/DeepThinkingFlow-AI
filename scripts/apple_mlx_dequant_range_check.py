#!/usr/bin/env python3
"""Dequantize one expert projection from the DeepThinkingFlow checkpoint and report value ranges."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


MXFP4_LUT = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Range-check one quantized expert projection by unpacking packed FP4 blocks."
    )
    parser.add_argument("--weights", default="original/model.safetensors", help="Path to safetensors weights.")
    parser.add_argument("--layer-index", type=int, default=0, help="MoE block index.")
    parser.add_argument(
        "--projection",
        choices=("mlp1", "mlp2"),
        default="mlp1",
        help="Which expert projection to dequantize.",
    )
    parser.add_argument("--expert-index", type=int, default=0, help="Expert index to inspect.")
    parser.add_argument("--group-size", type=int, default=32, help="Logical values represented by each scale group.")
    return parser.parse_args()


def unpack_fp4(blocks: np.ndarray) -> np.ndarray:
    lo = np.bitwise_and(blocks, 0x0F)
    hi = np.right_shift(blocks, 4)
    stacked = np.stack([lo, hi], axis=-1)
    return stacked.reshape(*blocks.shape[:-1], blocks.shape[-1] * 2).astype(np.float32)


def decode_mxfp4(blocks: np.ndarray) -> np.ndarray:
    lo = np.bitwise_and(blocks, 0x0F)
    hi = np.bitwise_and(np.right_shift(blocks, 4), 0x0F)
    stacked = np.stack([MXFP4_LUT[lo], MXFP4_LUT[hi]], axis=-1)
    return stacked.reshape(*blocks.shape[:-1], blocks.shape[-1] * 2)


def decode_ue8(scales: np.ndarray) -> np.ndarray:
    return np.power(2.0, scales.astype(np.float32) - 127.0)


def expand_scales(scales: np.ndarray, group_size: int) -> np.ndarray:
    return np.repeat(scales[..., None], group_size, axis=-1)


def load_header_and_offsets(weights_path: Path) -> tuple[dict[str, object], int]:
    with weights_path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    return header, 8 + header_len


def dtype_for_safetensors(dtype_name: str) -> np.dtype:
    mapping = {
        "U8": np.uint8,
        "BF16": np.uint16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype for local range check: {dtype_name}")
    return mapping[dtype_name]


def load_tensor_np(weights_path: Path, name: str) -> np.ndarray:
    header, data_start = load_header_and_offsets(weights_path)
    spec = header[name]
    begin, end = spec["data_offsets"]
    dtype = dtype_for_safetensors(spec["dtype"])
    shape = tuple(spec["shape"])
    with weights_path.open("rb") as handle:
        handle.seek(data_start + begin)
        raw = handle.read(end - begin)
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def main() -> int:
    args = parse_args()
    weights_path = Path(args.weights).resolve()
    prefix = f"block.{args.layer_index}.mlp.{args.projection}_weight"
    blocks_name = f"{prefix}.blocks"
    scales_name = f"{prefix}.scales"

    blocks = load_tensor_np(weights_path, blocks_name)
    scales = load_tensor_np(weights_path, scales_name)

    expert_blocks = blocks[args.expert_index]
    expert_scales = scales[args.expert_index]
    unpacked = unpack_fp4(expert_blocks)
    decoded_values = decode_mxfp4(expert_blocks)
    decoded_scales = decode_ue8(expert_scales)
    expanded_scales = expand_scales(decoded_scales, args.group_size)
    dequant = decoded_values * expanded_scales
    unpacked_flat = unpacked.reshape(unpacked.shape[0], -1)
    decoded_values_flat = decoded_values.reshape(decoded_values.shape[0], -1)
    decoded_scales_flat = decoded_scales.reshape(decoded_scales.shape[0], -1)
    expanded_scales_flat = expanded_scales.reshape(expanded_scales.shape[0], -1)
    dequant_flat = dequant.reshape(dequant.shape[0], -1)

    payload = {
        "weights": str(weights_path),
        "layer_index": args.layer_index,
        "projection": args.projection,
        "expert_index": args.expert_index,
        "blocks_shape": list(expert_blocks.shape),
        "scales_shape": list(expert_scales.shape),
        "unpacked_shape": list(unpacked.shape),
        "decoded_values_shape": list(decoded_values.shape),
        "decoded_scales_shape": list(decoded_scales.shape),
        "expanded_scales_shape": list(expanded_scales.shape),
        "unpacked_flat_shape": list(unpacked_flat.shape),
        "decoded_values_flat_shape": list(decoded_values_flat.shape),
        "decoded_scales_flat_shape": list(decoded_scales_flat.shape),
        "expanded_scales_flat_shape": list(expanded_scales_flat.shape),
        "dequant_flat_shape": list(dequant_flat.shape),
        "group_size": args.group_size,
        "ranges": {
            "unpacked": {
                "min": float(unpacked.min()),
                "max": float(unpacked.max()),
                "mean": float(unpacked.mean()),
            },
            "decoded_values": {
                "min": float(decoded_values.min()),
                "max": float(decoded_values.max()),
                "mean": float(decoded_values.mean()),
            },
            "decoded_scales": {
                "min": float(decoded_scales.min()),
                "max": float(decoded_scales.max()),
                "mean": float(decoded_scales.mean()),
            },
            "scales": {
                "min": float(expanded_scales.min()),
                "max": float(expanded_scales.max()),
                "mean": float(expanded_scales.mean()),
            },
            "dequant": {
                "min": float(dequant.min()),
                "max": float(dequant.max()),
                "mean": float(dequant.mean()),
                "abs_max": float(np.abs(dequant).max()),
            },
        },
        "claim_boundary": {
            "provisional_decode_only": True,
            "not_full_moe_forward": True,
            "mxfp4_lut_assumed": True,
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

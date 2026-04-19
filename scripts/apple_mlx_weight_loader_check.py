#!/usr/bin/env python3
"""Load DeepThinkingFlow weights with MLX and verify the first-layer tensor shapes."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from deepthinkingflow_apple.mlx_adapter import MLXUnavailable, load_deepthinkingflow_weight_shapes_with_mlx, verify_deepthinkingflow_first_block_shapes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load DeepThinkingFlow safetensors with MLX and verify key first-block shapes."
    )
    parser.add_argument(
        "--weights",
        default="original/model.safetensors",
        help="Path to the safetensors weight file.",
    )
    parser.add_argument(
        "--config",
        default="runtime/transformers/DeepThinkingFlow/config.json",
        help="Path to the model config.json.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_header_shapes(path: Path) -> dict[str, list[int]]:
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    return {
        name: spec.get("shape", [])
        for name, spec in header.items()
        if isinstance(spec, dict) and "shape" in spec
    }


def main() -> int:
    args = parse_args()
    weights_path = Path(args.weights).resolve()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    payload = {
        "weights": str(weights_path),
        "config": str(config_path),
        "architecture": config.get("architectures", []),
        "model_type": config.get("model_type", ""),
    }
    try:
        mlx_shapes = load_deepthinkingflow_weight_shapes_with_mlx(weights_path)
        payload["mlx_available"] = True
        payload["verified_shapes"] = verify_deepthinkingflow_first_block_shapes(mlx_shapes, config)
        payload["sample_shapes"] = {
            key: mlx_shapes.get(key)
            for key in (
                "block.0.attn.qkv.weight",
                "block.0.attn.out.weight",
                "block.0.mlp.gate.weight",
                "block.0.mlp.mlp1_weight",
                "block.0.mlp.mlp2_weight",
            )
        }
    except MLXUnavailable as exc:
        payload["mlx_available"] = False
        payload["mlx_error"] = str(exc)
        header_shapes = load_header_shapes(weights_path)
        payload["verified_shapes"] = verify_deepthinkingflow_first_block_shapes(header_shapes, config)
        payload["sample_shapes"] = {
            key: header_shapes.get(key)
            for key in (
                "block.0.attn.qkv.weight",
                "block.0.attn.out.weight",
                "block.0.mlp.gate.weight",
                "block.0.mlp.mlp1_weight",
                "block.0.mlp.mlp2_weight",
            )
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

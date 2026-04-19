#!/usr/bin/env python3
"""Inspect DeepThinkingFlow quantized MoE/FFN metadata from a safetensors header."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from deepthinkingflow_apple.mlx_adapter import inspect_deepthinkingflow_moe_ffn_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect quantized MoE/FFN metadata for one DeepThinkingFlow block."
    )
    parser.add_argument("--weights", default="original/model.safetensors", help="Path to safetensors weights.")
    parser.add_argument(
        "--config",
        default="runtime/transformers/DeepThinkingFlow/config.json",
        help="Path to the transformers config.json file.",
    )
    parser.add_argument("--layer-index", type=int, default=0, help="Block index to inspect.")
    return parser.parse_args()


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
    weight_shapes = load_header_shapes(weights_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    payload = {
        "weights": str(weights_path),
        "config": str(config_path),
        "inspection": inspect_deepthinkingflow_moe_ffn_metadata(weight_shapes, config, layer_index=args.layer_index),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

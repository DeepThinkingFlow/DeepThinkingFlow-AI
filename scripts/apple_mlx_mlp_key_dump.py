#!/usr/bin/env python3
"""Dump the real block.N.mlp.* tensor names from the DeepThinkingFlow safetensors header."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from deepthinkingflow_apple.mlx_adapter import list_block_mlp_keys_from_shapes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump the real DeepThinkingFlow MLP tensor names for one block from safetensors."
    )
    parser.add_argument("--weights", default="original/model.safetensors", help="Path to safetensors weights.")
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
    shapes = load_header_shapes(Path(args.weights).resolve())
    payload = {
        "weights": str(Path(args.weights).resolve()),
        "layer_index": args.layer_index,
        "mlp_keys": list_block_mlp_keys_from_shapes(shapes, args.layer_index),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

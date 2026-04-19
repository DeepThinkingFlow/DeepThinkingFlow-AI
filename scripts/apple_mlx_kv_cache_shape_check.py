#!/usr/bin/env python3
"""Dry-run alternating attention and KV-cache shapes for DeepThinkingFlow layers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from deepthinkingflow_apple.mlx_adapter import dry_run_deepthinkingflow_block_attention_shapes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run alternating attention and KV-cache shapes for one DeepThinkingFlow block."
    )
    parser.add_argument(
        "--config",
        default="runtime/transformers/DeepThinkingFlow/config.json",
        help="Path to config.json.",
    )
    parser.add_argument("--layer-index", type=int, default=0, help="Layer index to inspect.")
    parser.add_argument("--seq-len", type=int, default=1, help="Number of new query tokens.")
    parser.add_argument("--cached-seq-len", type=int, default=0, help="Existing cache length before append.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    payload = {
        "config": str(config_path),
        "layer_index": args.layer_index,
        "seq_len": args.seq_len,
        "cached_seq_len": args.cached_seq_len,
        "attention_shapes": dry_run_deepthinkingflow_block_attention_shapes(
            args.seq_len,
            config,
            layer_index=args.layer_index,
            cached_seq_len=args.cached_seq_len,
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

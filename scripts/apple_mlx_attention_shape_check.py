#!/usr/bin/env python3
"""Dry-run shape check for RMSNorm + fused QKV split + GQA repeat_kv on DeepThinkingFlow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from deepthinkingflow_apple.mlx_adapter import dry_run_deepthinkingflow_attention_shapes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run shape check for the first DeepThinkingFlow attention path before MoE implementation."
    )
    parser.add_argument(
        "--config",
        default="runtime/transformers/DeepThinkingFlow/config.json",
        help="Path to config.json.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=16,
        help="Sequence length to use for the dry-run shape calculation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(Path(args.config).resolve().read_text(encoding="utf-8"))
    payload = {
        "config": str(Path(args.config).resolve()),
        "seq_len": args.seq_len,
        "attention_shapes": dry_run_deepthinkingflow_attention_shapes(args.seq_len, config),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

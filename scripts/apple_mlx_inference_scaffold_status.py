#!/usr/bin/env python3
"""Report tokenizer and inference-loop scaffold readiness for the DeepThinkingFlow Apple path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from deepthinkingflow_apple.inference import inference_scaffold_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report tokenizer and inference-loop scaffold readiness for the Apple/MLX path."
    )
    parser.add_argument(
        "--model-dir",
        default="runtime/transformers/DeepThinkingFlow",
        help="Path to the local Transformers model directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = inference_scaffold_status(args.model_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

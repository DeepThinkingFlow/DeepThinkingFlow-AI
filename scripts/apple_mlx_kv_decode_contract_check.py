#!/usr/bin/env python3
"""Verify KV-cache decode expectations for DeepThinkingFlow Apple-path layer types."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from deepthinkingflow_apple.inference import kv_cache_decode_status, load_config
from deepthinkingflow_apple.mlx_adapter import KVCache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify KV-cache decode contract for Apple/MLX path.")
    parser.add_argument("--model-dir", default="runtime/transformers/DeepThinkingFlow", help="Local model directory.")
    parser.add_argument("--sliding-cache-len", type=int, default=128, help="Synthetic cache length for sliding layers.")
    parser.add_argument("--full-cache-len", type=int, default=257, help="Synthetic cache length for full-attention layers.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.model_dir)
    cache = KVCache()
    layer_types = list(config.get("layer_types") or [])

    class _FakeTensor:
        def __init__(self, seq_len: int) -> None:
            self.shape = (seq_len, int(config["num_key_value_heads"]), int(config["head_dim"]))

    for layer_idx in range(int(config["num_hidden_layers"])):
        layer_type = layer_types[layer_idx] if layer_idx < len(layer_types) else "full_attention"
        target_len = args.sliding_cache_len if layer_type == "sliding_attention" else args.full_cache_len
        while len(cache.k) <= layer_idx:
            cache.k.append(None)
            cache.v.append(None)
        cache.k[layer_idx] = _FakeTensor(target_len)
        cache.v[layer_idx] = _FakeTensor(target_len)

    payload = kv_cache_decode_status(config, cache)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

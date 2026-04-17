#!/usr/bin/env python3
"""Create a tiny local GptOss model directory for smoke-training the LoRA pipeline."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from transformers import GptOssConfig, GptOssForCausalLM

ROOT_DIR = Path(__file__).resolve().parents[1]
BASE_MODEL_DIR = ROOT_DIR / "runtime" / "transformers" / "DeepThinkingFlow"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a tiny local GptOss model dir for CPU smoke training."
    )
    parser.add_argument(
        "--output-dir",
        default="runtime/transformers/DeepThinkingFlow-tiny-smoke",
        help="Output directory for the tiny model.",
    )
    return parser.parse_args()


def copy_required_runtime_files(output_dir: Path) -> None:
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "generation_config.json",
    ]:
        shutil.copy2(BASE_MODEL_DIR / name, output_dir / name)


def main() -> int:
    args = parse_args()
    output_dir = (ROOT_DIR / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = GptOssConfig(
        vocab_size=201088,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        num_local_experts=4,
        num_experts_per_tok=2,
        experts_per_token=2,
        max_position_embeddings=4096,
        initial_context_length=256,
        sliding_window=64,
        layer_types=["sliding_attention", "full_attention"],
        rope_theta=150000,
        attention_bias=True,
        use_cache=False,
    )
    model = GptOssForCausalLM(config)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    copy_required_runtime_files(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

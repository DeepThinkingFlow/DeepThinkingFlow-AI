#!/usr/bin/env python3
"""Verify DeepThinkingFlow Apple-path prompt and sampling contract without requiring MLX weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from deepthinkingflow_apple.inference import (
    GenerationConfig,
    load_config,
    validate_generation_config,
    verify_generation_contract,
)
from deepthinkingflow_apple.tokenizer import GPTOssTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify prompt packaging and sampling contract for Apple/MLX path.")
    parser.add_argument("--model-dir", default="runtime/transformers/DeepThinkingFlow", help="Local model directory.")
    parser.add_argument("--prompt", default="Xin chao", help="Prompt to verify.")
    parser.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.model_dir)
    sampling_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        reasoning_effort=args.reasoning_effort,
    )
    try:
        validate_generation_config(sampling_config)
    except ValueError as exc:
        print(
            json.dumps(
                {
                    "model_dir": str(Path(args.model_dir).resolve()),
                    "dependency_ready": False,
                    "configuration_ready": False,
                    "configuration_error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    payload = {
        "model_dir": str(Path(args.model_dir).resolve()),
        "dependency_ready": True,
        "configuration_ready": True,
    }
    try:
        tokenizer = GPTOssTokenizer(args.model_dir)
        payload.update(
            verify_generation_contract(
                tokenizer=tokenizer,
                config=config,
                prompt=args.prompt,
                sampling=sampling_config,
            )
        )
    except RuntimeError as exc:
        payload.update(
            {
                "dependency_ready": False,
                "dependency_error": str(exc),
                "prompt": args.prompt,
                "sampling_contract": {
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "min_p": args.min_p,
                    "repetition_penalty": args.repetition_penalty,
                    "reasoning_effort": args.reasoning_effort,
                },
                "prompt_package_ready": False,
                "rendered_prompt_non_empty": False,
                "input_token_count": 0,
                "stop_token_ids": [],
                "layer_types_count": len(list(config.get("layer_types") or [])),
                "num_hidden_layers": int(config["num_hidden_layers"]),
            }
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

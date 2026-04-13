#!/usr/bin/env python3
"""Render a DeepThinkingFlow chat-template prompt with the local behavior bundle injected."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepthinkingflow_runtime import (
    DEFAULT_BUNDLE_DIR,
    DEFAULT_MODEL_DIR,
    ensure_file,
    load_system_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a DeepThinkingFlow prompt using the Transformers chat template."
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Transformers-ready local model directory.",
    )
    parser.add_argument(
        "--bundle",
        default=DEFAULT_BUNDLE_DIR,
        help="Behavior bundle directory.",
    )
    parser.add_argument(
        "--user",
        required=True,
        help="User message to render.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default="high",
        help="Reasoning effort passed into the DeepThinkingFlow chat template.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print messages and rendered prompt as JSON.",
    )
    parser.add_argument(
        "--reasoning-in-system",
        action="store_true",
        help="Also append 'Reasoning: <level>' to the system prompt text.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise SystemExit(
            "transformers is required for prompt rendering. "
            "Install it first, for example: pip install -U transformers tokenizers"
        ) from exc

    bundle_dir = Path(args.bundle).resolve()
    model_dir = Path(args.model_dir).resolve()
    ensure_file(model_dir / "tokenizer.json", "tokenizer.json")
    ensure_file(model_dir / "chat_template.jinja", "chat_template.jinja")
    system_prompt = load_system_prompt(
        bundle_dir,
        reasoning_effort=args.reasoning_effort,
        reasoning_in_system=args.reasoning_in_system,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": args.user},
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort=args.reasoning_effort,
    )

    if args.json:
        payload = {
            "model_dir": str(model_dir),
            "bundle": str(bundle_dir),
            "reasoning_effort": args.reasoning_effort,
            "messages": messages,
            "rendered_prompt": rendered,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

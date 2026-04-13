#!/usr/bin/env python3
"""Run DeepThinkingFlow with Transformers while injecting the local behavior bundle."""

from __future__ import annotations

import argparse
import json
import sys

from deepthinkingflow_runtime import (
    DEFAULT_BUNDLE_DIR,
    DEFAULT_MODEL_DIR,
    build_low_memory_warning_payload,
    generate_response,
    load_model_and_tokenizer,
    load_system_prompt,
    render_prompt,
    resolve_bundle_dir,
    resolve_model_ref,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate with DeepThinkingFlow through Transformers using a local behavior bundle."
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Transformers-ready local model directory or HF model id.",
    )
    parser.add_argument(
        "--bundle",
        default=DEFAULT_BUNDLE_DIR,
        help="Behavior bundle directory.",
    )
    parser.add_argument(
        "--user",
        required=True,
        help="User message.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default="high",
        help="Reasoning effort passed into the DeepThinkingFlow chat template.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map argument.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        help="Transformers torch_dtype argument.",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional attention implementation override.",
    )
    parser.add_argument(
        "--print-rendered-prompt",
        action="store_true",
        help="Print the rendered DeepThinkingFlow harmony prompt before generation.",
    )
    parser.add_argument(
        "--include-raw-completion",
        action="store_true",
        help="Include the raw decoded completion, which may contain analysis/channel tokens.",
    )
    parser.add_argument(
        "--include-analysis",
        action="store_true",
        help="Include extracted analysis text if present. Keep this off for end-user output.",
    )
    parser.add_argument(
        "--reasoning-in-system",
        action="store_true",
        help="Also append 'Reasoning: <level>' to the system prompt text.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle_dir = resolve_bundle_dir(args.bundle)
    system_prompt = load_system_prompt(
        bundle_dir,
        reasoning_effort=args.reasoning_effort,
        reasoning_in_system=args.reasoning_in_system,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": args.user},
    ]

    model_ref, model_path = resolve_model_ref(args.model_dir)
    warning = build_low_memory_warning_payload(model_path)
    if warning:
        print(json.dumps(warning, ensure_ascii=False), file=sys.stderr)

    tokenizer, model = load_model_and_tokenizer(
        model_ref,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
    )
    rendered = render_prompt(tokenizer, messages, args.reasoning_effort)
    if args.print_rendered_prompt:
        print("===== RENDERED PROMPT =====")
        print(rendered)
        print("===== END PROMPT =====")

    response = generate_response(
        model,
        tokenizer,
        messages=messages,
        reasoning_effort=args.reasoning_effort,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    result = {
        "model": model_ref,
        "bundle": str(bundle_dir),
        "reasoning_effort": args.reasoning_effort,
        "final_text": response["final_text"],
    }
    if args.include_analysis:
        result["analysis_text"] = response["analysis_text"]
    if args.include_raw_completion:
        result["decoded_completion"] = response["decoded_completion"]
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

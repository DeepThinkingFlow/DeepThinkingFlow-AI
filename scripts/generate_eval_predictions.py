#!/usr/bin/env python3
"""Generate eval-case predictions from a base model or a base+adapter pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from deepthinkingflow_runtime import (
    generate_response,
    load_model_and_tokenizer,
    load_system_prompt,
    resolve_bundle_dir,
    resolve_model_ref,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate predictions JSONL for DeepThinkingFlow eval cases."
    )
    parser.add_argument("--model-dir", required=True, help="Base model dir or HF model id.")
    parser.add_argument("--bundle", default="behavior/DeepThinkingFlow", help="Behavior bundle dir.")
    parser.add_argument("--eval-cases", required=True, help="Eval cases JSONL.")
    parser.add_argument("--output-jsonl", required=True, help="Predictions JSONL output.")
    parser.add_argument("--adapter-dir", default="", help="Optional PEFT adapter dir.")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="high")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of eval cases.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def attach_adapter_if_needed(model: Any, adapter_dir: str) -> Any:
    if not adapter_dir.strip():
        return model
    try:
        from peft import PeftModel
    except Exception as exc:
        raise SystemExit("peft is required to load adapter predictions.") from exc
    return PeftModel.from_pretrained(model, adapter_dir.strip())


def main() -> int:
    args = parse_args()
    bundle_dir = resolve_bundle_dir(args.bundle)
    system_prompt = load_system_prompt(bundle_dir, args.reasoning_effort, reasoning_in_system=False)
    model_ref, _model_path = resolve_model_ref(args.model_dir)
    tokenizer, model = load_model_and_tokenizer(
        model_ref,
        device_map=args.device_map,
        dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
    )
    model = attach_adapter_if_needed(model, args.adapter_dir)

    cases = load_jsonl(Path(args.eval_cases).resolve())
    if args.limit > 0:
        cases = cases[: min(args.limit, len(cases))]

    output_path = Path(args.output_jsonl).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": case["user"]},
            ]
            response = generate_response(
                model,
                tokenizer,
                messages=messages,
                reasoning_effort=args.reasoning_effort,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            payload = {
                "id": case["id"],
                "final_text": response["final_text"],
                "analysis_text": response["analysis_text"],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "model_dir": model_ref,
                "adapter_dir": args.adapter_dir.strip(),
                "eval_cases": str(Path(args.eval_cases).resolve()),
                "output_jsonl": str(output_path),
                "cases_generated": len(cases),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

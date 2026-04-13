#!/usr/bin/env python3
"""Validate, deduplicate, and split a harmony-format SFT dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a harmony-format SFT dataset for DeepThinkingFlow."
    )
    parser.add_argument("--input", required=True, help="Input JSONL dataset.")
    parser.add_argument("--train-out", required=True, help="Output train JSONL path.")
    parser.add_argument("--eval-out", required=True, help="Output eval JSONL path.")
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.2,
        help="Evaluation split ratio in [0, 1).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Keep duplicates instead of deduplicating by canonical message content.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Input dataset is empty.")
    for idx, row in enumerate(rows, start=1):
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Row {idx} missing non-empty messages list.")
        if messages[-1].get("role") != "assistant":
            raise ValueError(f"Row {idx} must end with assistant.")
        if "content" not in messages[-1]:
            raise ValueError(f"Row {idx} assistant must include content.")


def canonical_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(row["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dedupe_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    dropped = 0
    for row in rows:
        key = canonical_hash(row)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        deduped.append(row)
    return deduped, dropped


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not (0 <= args.eval_ratio < 1):
        raise SystemExit("--eval-ratio must be in [0, 1).")

    input_path = Path(args.input).resolve()
    train_out = Path(args.train_out).resolve()
    eval_out = Path(args.eval_out).resolve()

    rows = load_jsonl(input_path)
    validate_rows(rows)

    dropped_duplicates = 0
    if not args.no_dedupe:
        rows, dropped_duplicates = dedupe_rows(rows)

    shuffled = list(rows)
    random.Random(args.seed).shuffle(shuffled)
    eval_count = max(1, int(round(len(shuffled) * args.eval_ratio))) if args.eval_ratio > 0 else 0
    eval_rows = shuffled[:eval_count]
    train_rows = shuffled[eval_count:]
    if not train_rows:
        raise SystemExit("Evaluation split consumed the whole dataset. Reduce --eval-ratio.")

    write_jsonl(train_out, train_rows)
    write_jsonl(eval_out, eval_rows)

    summary = {
        "input": str(input_path),
        "train_out": str(train_out),
        "eval_out": str(eval_out),
        "input_examples": len(rows) + dropped_duplicates,
        "deduplicated_examples": len(rows),
        "dropped_duplicates": dropped_duplicates,
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "seed": args.seed,
        "eval_ratio": args.eval_ratio,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

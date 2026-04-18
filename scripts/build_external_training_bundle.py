#!/usr/bin/env python3
"""Build train/eval JSONL assets from prepared external chat datasets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/eval JSONL files from prepared external chat dataset exports."
    )
    parser.add_argument(
        "--input-jsonl",
        action="append",
        dest="input_jsonl",
        required=True,
        help="Input JSONL path produced by export_prepared_chat_jsonl.py. Repeat for multiple sources.",
    )
    parser.add_argument(
        "--train-output",
        required=True,
        help="Output JSONL path for the train split.",
    )
    parser.add_argument(
        "--eval-output",
        required=True,
        help="Output JSONL path for the eval split.",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.1,
        help="Fraction of rows reserved for eval.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic shuffle seed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on the total number of rows after concatenation.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def validate_rows(rows: list[dict[str, Any]], label: str) -> None:
    if not rows:
        raise SystemExit(f"{label} is empty.")
    for index, row in enumerate(rows, start=1):
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise SystemExit(f"{label} row {index} is missing messages.")
        if messages[-1].get("role") != "assistant":
            raise SystemExit(f"{label} row {index} must end with assistant.")
        if not isinstance(messages[-1].get("content"), str) or not messages[-1]["content"].strip():
            raise SystemExit(f"{label} row {index} assistant content is invalid.")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if not (0.0 < args.eval_ratio < 1.0):
        raise SystemExit("--eval-ratio must be between 0 and 1.")

    combined_rows: list[dict[str, Any]] = []
    for raw_path in args.input_jsonl:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise SystemExit(f"Missing input JSONL: {path}")
        rows = read_jsonl(path)
        validate_rows(rows, str(path))
        combined_rows.extend(rows)

    if args.limit > 0:
        combined_rows = combined_rows[: min(args.limit, len(combined_rows))]
    validate_rows(combined_rows, "combined external dataset")

    shuffled = list(combined_rows)
    random.Random(args.seed).shuffle(shuffled)
    eval_count = max(1, int(round(len(shuffled) * args.eval_ratio)))
    if eval_count >= len(shuffled):
        raise SystemExit("Eval split consumed the full dataset. Reduce --eval-ratio or add more rows.")

    eval_rows = shuffled[:eval_count]
    train_rows = shuffled[eval_count:]
    write_jsonl(Path(args.train_output).resolve(), train_rows)
    write_jsonl(Path(args.eval_output).resolve(), eval_rows)

    print(
        json.dumps(
            {
                "train_output": str(Path(args.train_output).resolve()),
                "eval_output": str(Path(args.eval_output).resolve()),
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "sources": [str(Path(path).resolve()) for path in args.input_jsonl],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export a prepared HF dataset directory into harmony-style JSONL rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a prepared dataset directory with `messages` into JSONL."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Dataset directory previously written by save_to_disk().",
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Target JSONL path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional row limit for the exported JSONL.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from datasets import load_from_disk
    except Exception as exc:
        raise SystemExit("datasets is required to export a prepared HF dataset directory.") from exc
    dataset = load_from_disk(args.input_dir)
    output_path = Path(args.output_jsonl).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(dataset) if args.limit <= 0 else min(args.limit, len(dataset))
    with output_path.open("w", encoding="utf-8") as handle:
        for index in range(total):
            row = dataset[index]
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise SystemExit(f"Row {index} is missing a valid messages field.")
            payload = {
                "messages": messages,
                "source_dataset": row.get("source_dataset", ""),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "input_dir": str(Path(args.input_dir).resolve()),
                "output_jsonl": str(output_path),
                "rows_exported": total,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

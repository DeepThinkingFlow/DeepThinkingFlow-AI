#!/usr/bin/env python3
"""Build deterministic DeepThinkingFlow train/eval assets from base and skill-compliance datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REQUIRED_SKILL_CATEGORIES = (
    "reject-false-weight-claim",
    "runtime-vs-learned",
    "short-analysis-no-cot",
    "deep-style-without-fake-internals",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic DeepThinkingFlow train/eval assets from base and skill-compliance datasets."
    )
    parser.add_argument(
        "--bundle",
        default="behavior/DeepThinkingFlow",
        help="Behavior bundle directory.",
    )
    parser.add_argument(
        "--skill-eval-per-category",
        type=int,
        default=1,
        help="How many skill-compliance examples per category should be reserved for eval.",
    )
    return parser.parse_args()


def ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"Missing {label}: {path}")


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def canonical_messages_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(row["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_messages(rows: list[dict[str, Any]], *, label: str) -> None:
    if not rows:
        raise SystemExit(f"{label} dataset is empty.")
    for idx, row in enumerate(rows, start=1):
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise SystemExit(f"{label} row {idx} missing messages.")
        assistant = messages[-1]
        if assistant.get("role") != "assistant":
            raise SystemExit(f"{label} row {idx} must end with assistant.")
        if "content" not in assistant:
            raise SystemExit(f"{label} row {idx} assistant must include content.")


def validate_skill_rows(rows: list[dict[str, Any]]) -> Counter[str]:
    validate_messages(rows, label="skill compliance")
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    for idx, row in enumerate(rows, start=1):
        category = row.get("category")
        if category not in REQUIRED_SKILL_CATEGORIES:
            raise SystemExit(f"skill compliance row {idx} has unsupported category: {category}")
        digest = canonical_messages_hash(row)
        if digest in seen:
            raise SystemExit(f"Duplicate skill compliance example detected at row {idx}")
        seen.add(digest)
        counts[category] += 1
    missing = sorted(set(REQUIRED_SKILL_CATEGORIES) - set(counts))
    if missing:
        raise SystemExit(f"Skill compliance dataset missing categories: {missing}")
    return counts


def split_skill_rows(
    rows: list[dict[str, Any]],
    *,
    eval_per_category: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, int]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    stats: dict[str, dict[str, int]] = {}

    for category in REQUIRED_SKILL_CATEGORIES:
        ordered = sorted(grouped[category], key=canonical_messages_hash)
        if len(ordered) <= eval_per_category:
            raise SystemExit(
                f"Category '{category}' does not have enough rows for eval split: "
                f"count={len(ordered)}, eval_per_category={eval_per_category}"
            )
        eval_part = ordered[-eval_per_category:]
        train_part = ordered[:-eval_per_category]
        train_rows.extend(train_part)
        eval_rows.extend(eval_part)
        stats[category] = {
            "total": len(ordered),
            "train": len(train_part),
            "eval": len(eval_part),
        }

    return train_rows, eval_rows, stats


def ensure_disjoint(left: list[dict[str, Any]], right: list[dict[str, Any]], *, label: str) -> None:
    left_hashes = {canonical_messages_hash(row) for row in left}
    right_hashes = {canonical_messages_hash(row) for row in right}
    overlap = left_hashes & right_hashes
    if overlap:
        raise SystemExit(f"{label} datasets overlap on {len(overlap)} examples.")


def main() -> int:
    args = parse_args()
    if args.skill_eval_per_category < 1:
        raise SystemExit("--skill-eval-per-category must be >= 1")

    bundle_dir = Path(args.bundle).resolve()
    training_dir = bundle_dir / "training"
    ensure_file(training_dir / "harmony_sft_vi.jsonl", "base harmony dataset")
    ensure_file(training_dir / "harmony_sft_vi.train.jsonl", "base harmony train split")
    ensure_file(training_dir / "harmony_sft_vi.eval.jsonl", "base harmony eval split")
    ensure_file(training_dir / "harmony_sft_skill_compliance_vi.jsonl", "skill compliance dataset")

    base_all = read_jsonl(training_dir / "harmony_sft_vi.jsonl")
    base_train = read_jsonl(training_dir / "harmony_sft_vi.train.jsonl")
    base_eval = read_jsonl(training_dir / "harmony_sft_vi.eval.jsonl")
    skill_all = read_jsonl(training_dir / "harmony_sft_skill_compliance_vi.jsonl")

    validate_messages(base_all, label="base harmony all")
    validate_messages(base_train, label="base harmony train")
    validate_messages(base_eval, label="base harmony eval")
    ensure_disjoint(base_train, base_eval, label="base harmony train/eval")
    validate_skill_rows(skill_all)

    skill_train, skill_eval, skill_stats = split_skill_rows(
        skill_all,
        eval_per_category=args.skill_eval_per_category,
    )
    ensure_disjoint(skill_train, skill_eval, label="skill compliance train/eval")

    combined_all = [*base_all, *skill_all]
    combined_train = [*base_train, *skill_train]
    combined_eval = [*base_eval, *skill_eval]
    ensure_disjoint(combined_train, combined_eval, label="combined train/eval")

    outputs = {
        training_dir / "harmony_sft_skill_compliance_vi.train.jsonl": skill_train,
        training_dir / "harmony_sft_skill_compliance_vi.eval.jsonl": skill_eval,
        training_dir / "harmony_sft_plus_skill_compliance_vi.jsonl": combined_all,
        training_dir / "harmony_sft_plus_skill_compliance_vi.train.jsonl": combined_train,
        training_dir / "harmony_sft_plus_skill_compliance_vi.eval.jsonl": combined_eval,
    }
    for path, rows in outputs.items():
        write_jsonl(path, rows)

    summary = {
        "bundle": str(bundle_dir),
        "skill_eval_per_category": args.skill_eval_per_category,
        "base": {
            "all": len(base_all),
            "train": len(base_train),
            "eval": len(base_eval),
        },
        "skill_compliance": {
            "all": len(skill_all),
            "train": len(skill_train),
            "eval": len(skill_eval),
            "per_category": skill_stats,
        },
        "combined": {
            "all": len(combined_all),
            "train": len(combined_train),
            "eval": len(combined_eval),
        },
        "outputs": {str(path.relative_to(bundle_dir)): len(rows) for path, rows in outputs.items()},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

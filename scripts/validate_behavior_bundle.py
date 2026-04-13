#!/usr/bin/env python3
"""Validate a behavior bundle for runtime steering and SFT seed data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a behavior bundle.")
    parser.add_argument("bundle", help="Path to the behavior bundle directory.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_messages_hash(messages: list[dict]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_bundle(bundle_dir: Path) -> dict:
    profile_path = bundle_dir / "profile.json"
    system_prompt_path = bundle_dir / "system_prompt.txt"
    ensure(profile_path.is_file(), f"Missing {profile_path}")
    ensure(system_prompt_path.is_file(), f"Missing {system_prompt_path}")

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    for key in ("name", "target_model", "files"):
        ensure(key in profile, f"profile.json missing key: {key}")
    quality_gates = profile.get("quality_gates", {})
    prepared_datasets = profile.get("prepared_datasets", {})

    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
    ensure(system_prompt, "system_prompt.txt is empty")
    ensure("<hard_rules>" in system_prompt, "system_prompt.txt missing <hard_rules> block")

    sft_path = bundle_dir / profile["files"]["sft_dataset"]
    evals_path = bundle_dir / profile["files"]["eval_cases"]
    ensure(sft_path.is_file(), f"Missing {sft_path}")
    ensure(evals_path.is_file(), f"Missing {evals_path}")
    harmony_sft_path = None
    if profile["files"].get("harmony_sft_dataset"):
        harmony_sft_path = bundle_dir / profile["files"]["harmony_sft_dataset"]
        ensure(harmony_sft_path.is_file(), f"Missing {harmony_sft_path}")

    sft_rows = read_jsonl(sft_path)
    eval_rows = read_jsonl(evals_path)
    harmony_rows = read_jsonl(harmony_sft_path) if harmony_sft_path else []
    ensure(sft_rows, "SFT dataset is empty")
    ensure(eval_rows, "Eval dataset is empty")
    if harmony_sft_path:
        ensure(harmony_rows, "Harmony SFT dataset is empty")

    for idx, row in enumerate(sft_rows, start=1):
        ensure("messages" in row, f"SFT row {idx} missing messages")
        ensure(isinstance(row["messages"], list) and row["messages"], f"SFT row {idx} has invalid messages")
        roles = [msg.get("role") for msg in row["messages"]]
        ensure(roles[0] == "system", f"SFT row {idx} should start with a system message")
        ensure(roles[-1] == "assistant", f"SFT row {idx} should end with an assistant message")

    eval_ids: set[str] = set()
    for idx, row in enumerate(eval_rows, start=1):
        for key in ("id", "user", "expected_traits"):
            ensure(key in row, f"Eval row {idx} missing key: {key}")
        ensure(isinstance(row["expected_traits"], list) and row["expected_traits"], f"Eval row {idx} has invalid expected_traits")
        ensure(isinstance(row["id"], str) and row["id"].strip(), f"Eval row {idx} has invalid id")
        ensure(isinstance(row["user"], str) and row["user"].strip(), f"Eval row {idx} has invalid user")
        if quality_gates.get("require_unique_eval_ids"):
            ensure(row["id"] not in eval_ids, f"Duplicate eval id: {row['id']}")
            eval_ids.add(row["id"])
        if "required_keywords" in row:
            ensure(
                isinstance(row["required_keywords"], list) and all(isinstance(item, str) and item for item in row["required_keywords"]),
                f"Eval row {idx} has invalid required_keywords",
            )
        if "required_keyword_groups" in row:
            ensure(
                isinstance(row["required_keyword_groups"], list)
                and all(
                    isinstance(group, list) and group and all(isinstance(item, str) and item for item in group)
                    for group in row["required_keyword_groups"]
                ),
                f"Eval row {idx} has invalid required_keyword_groups",
            )
        if "forbidden_keywords" in row:
            ensure(
                isinstance(row["forbidden_keywords"], list) and all(isinstance(item, str) and item for item in row["forbidden_keywords"]),
                f"Eval row {idx} has invalid forbidden_keywords",
            )
        if "must_start_with_one_of" in row:
            ensure(
                isinstance(row["must_start_with_one_of"], list)
                and all(isinstance(item, str) and item for item in row["must_start_with_one_of"]),
                f"Eval row {idx} has invalid must_start_with_one_of",
            )
        if "max_chars" in row:
            ensure(isinstance(row["max_chars"], int) and row["max_chars"] > 0, f"Eval row {idx} has invalid max_chars")
        if "analysis_max_chars" in row:
            ensure(
                isinstance(row["analysis_max_chars"], int) and row["analysis_max_chars"] >= 0,
                f"Eval row {idx} has invalid analysis_max_chars",
            )
        if "min_numbered_steps" in row:
            ensure(
                isinstance(row["min_numbered_steps"], int) and row["min_numbered_steps"] >= 0,
                f"Eval row {idx} has invalid min_numbered_steps",
            )

    harmony_hashes: set[str] = set()
    for idx, row in enumerate(harmony_rows, start=1):
        ensure("messages" in row, f"Harmony SFT row {idx} missing messages")
        messages = row["messages"]
        ensure(isinstance(messages, list) and messages, f"Harmony SFT row {idx} has invalid messages")
        assistant = messages[-1]
        ensure(assistant.get("role") == "assistant", f"Harmony SFT row {idx} must end with assistant")
        ensure("content" in assistant, f"Harmony SFT row {idx} assistant must include content")
        if "thinking" in assistant:
            ensure(isinstance(assistant["thinking"], str), f"Harmony SFT row {idx} thinking must be a string")
        if quality_gates.get("require_unique_harmony_examples"):
            digest = canonical_messages_hash(messages)
            ensure(digest not in harmony_hashes, f"Duplicate harmony example detected at row {idx}")
            harmony_hashes.add(digest)

    if prepared_datasets.get("train_dataset"):
        ensure((bundle_dir / prepared_datasets["train_dataset"]).is_file(), "Missing prepared train dataset")
    if prepared_datasets.get("eval_dataset"):
        ensure((bundle_dir / prepared_datasets["eval_dataset"]).is_file(), "Missing prepared eval dataset")

    if "min_sft_examples" in quality_gates:
        ensure(len(sft_rows) >= int(quality_gates["min_sft_examples"]), "SFT dataset below minimum size gate")
    if "min_harmony_sft_examples" in quality_gates:
        ensure(
            len(harmony_rows) >= int(quality_gates["min_harmony_sft_examples"]),
            "Harmony SFT dataset below minimum size gate",
        )
    if "min_eval_cases" in quality_gates:
        ensure(len(eval_rows) >= int(quality_gates["min_eval_cases"]), "Eval dataset below minimum size gate")

    return {
        "bundle": profile["name"],
        "target_model": profile["target_model"],
        "system_prompt_chars": len(system_prompt),
        "sft_examples": len(sft_rows),
        "harmony_sft_examples": len(harmony_rows),
        "eval_cases": len(eval_rows),
        "prepared_train_dataset": prepared_datasets.get("train_dataset", ""),
        "prepared_eval_dataset": prepared_datasets.get("eval_dataset", ""),
    }


def main() -> int:
    args = parse_args()
    summary = validate_bundle(Path(args.bundle).resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

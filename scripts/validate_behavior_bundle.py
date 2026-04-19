#!/usr/bin/env python3
"""Validate a behavior bundle for runtime steering and SFT seed data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from deepthinkingflow_json_io import load_jsonl_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a behavior bundle.")
    parser.add_argument("bundle", help="Path to the behavior bundle directory.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return load_jsonl_file(path, "jsonl file")


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_messages_hash(messages: list[dict]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_jsonl_if_exists(path: Path | None) -> list[dict]:
    if path is None or not path.is_file():
        return []
    return read_jsonl(path)


def validate_harmony_rows(
    rows: list[dict],
    *,
    label: str,
    require_unique: bool,
    require_category: bool = False,
    allowed_categories: set[str] | None = None,
) -> Counter[str]:
    hashes: set[str] = set()
    categories: Counter[str] = Counter()
    for idx, row in enumerate(rows, start=1):
        ensure("messages" in row, f"{label} row {idx} missing messages")
        messages = row["messages"]
        ensure(isinstance(messages, list) and messages, f"{label} row {idx} has invalid messages")
        assistant = messages[-1]
        ensure(assistant.get("role") == "assistant", f"{label} row {idx} must end with assistant")
        ensure("content" in assistant, f"{label} row {idx} assistant must include content")
        if require_category:
            category = row.get("category")
            ensure(isinstance(category, str) and category.strip(), f"{label} row {idx} missing category")
            category = category.strip()
            if allowed_categories is not None:
                ensure(category in allowed_categories, f"{label} row {idx} uses unsupported category: {category}")
            categories[category] += 1
        if "thinking" in assistant:
            ensure(isinstance(assistant["thinking"], str), f"{label} row {idx} thinking must be a string")
        if require_unique:
            digest = canonical_messages_hash(messages)
            ensure(digest not in hashes, f"Duplicate {label.lower()} example detected at row {idx}")
            hashes.add(digest)
    return categories


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
    required_skill_categories = set(quality_gates.get("required_skill_compliance_categories", []))

    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
    ensure(system_prompt, "system_prompt.txt is empty")
    ensure("<hard_rules>" in system_prompt, "system_prompt.txt missing <hard_rules> block")

    sft_path = bundle_dir / profile["files"]["sft_dataset"]
    evals_path = bundle_dir / profile["files"]["eval_cases"]
    promotion_policy_path = None
    skill_eval_cases_path = None
    ensure(sft_path.is_file(), f"Missing {sft_path}")
    ensure(evals_path.is_file(), f"Missing {evals_path}")
    if profile["files"].get("promotion_policy"):
        promotion_policy_path = bundle_dir / profile["files"]["promotion_policy"]
        ensure(promotion_policy_path.is_file(), f"Missing {promotion_policy_path}")
    if profile["files"].get("skill_compliance_eval_cases"):
        skill_eval_cases_path = bundle_dir / profile["files"]["skill_compliance_eval_cases"]
        ensure(skill_eval_cases_path.is_file(), f"Missing {skill_eval_cases_path}")
    harmony_sft_path = None
    skill_compliance_path = None
    if profile["files"].get("harmony_sft_dataset"):
        harmony_sft_path = bundle_dir / profile["files"]["harmony_sft_dataset"]
        ensure(harmony_sft_path.is_file(), f"Missing {harmony_sft_path}")
    if profile["files"].get("skill_compliance_dataset"):
        skill_compliance_path = bundle_dir / profile["files"]["skill_compliance_dataset"]
        ensure(skill_compliance_path.is_file(), f"Missing {skill_compliance_path}")

    sft_rows = read_jsonl(sft_path)
    eval_rows = read_jsonl(evals_path)
    skill_eval_rows = read_jsonl_if_exists(skill_eval_cases_path)
    promotion_policy = json.loads(promotion_policy_path.read_text(encoding="utf-8")) if promotion_policy_path else {}
    harmony_rows = read_jsonl_if_exists(harmony_sft_path)
    skill_compliance_rows = read_jsonl_if_exists(skill_compliance_path)
    ensure(sft_rows, "SFT dataset is empty")
    ensure(eval_rows, "Eval dataset is empty")
    if harmony_sft_path:
        ensure(harmony_rows, "Harmony SFT dataset is empty")
    if quality_gates.get("require_skill_compliance_examples"):
        ensure(skill_compliance_path is not None, "profile.json requires skill_compliance_dataset")
        ensure(skill_compliance_rows, "Skill compliance dataset is empty")
    if skill_eval_cases_path:
        ensure(skill_eval_rows, "Skill compliance eval dataset is empty")

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

    skill_eval_ids: set[str] = set()
    for idx, row in enumerate(skill_eval_rows, start=1):
        for key in ("id", "user", "expected_traits"):
            ensure(key in row, f"Skill eval row {idx} missing key: {key}")
        ensure(isinstance(row["expected_traits"], list) and row["expected_traits"], f"Skill eval row {idx} has invalid expected_traits")
        ensure(isinstance(row["id"], str) and row["id"].strip(), f"Skill eval row {idx} has invalid id")
        ensure(isinstance(row["user"], str) and row["user"].strip(), f"Skill eval row {idx} has invalid user")
        if quality_gates.get("require_unique_skill_compliance_eval_ids"):
            ensure(row["id"] not in skill_eval_ids, f"Duplicate skill eval id: {row['id']}")
            skill_eval_ids.add(row["id"])
        if "required_keywords" in row:
            ensure(
                isinstance(row["required_keywords"], list) and all(isinstance(item, str) and item for item in row["required_keywords"]),
                f"Skill eval row {idx} has invalid required_keywords",
            )
        if "required_keyword_groups" in row:
            ensure(
                isinstance(row["required_keyword_groups"], list)
                and all(
                    isinstance(group, list) and group and all(isinstance(item, str) and item for item in group)
                    for group in row["required_keyword_groups"]
                ),
                f"Skill eval row {idx} has invalid required_keyword_groups",
            )
        if "forbidden_keywords" in row:
            ensure(
                isinstance(row["forbidden_keywords"], list) and all(isinstance(item, str) and item for item in row["forbidden_keywords"]),
                f"Skill eval row {idx} has invalid forbidden_keywords",
            )
        if "max_chars" in row:
            ensure(isinstance(row["max_chars"], int) and row["max_chars"] > 0, f"Skill eval row {idx} has invalid max_chars")
        if "analysis_max_chars" in row:
            ensure(
                isinstance(row["analysis_max_chars"], int) and row["analysis_max_chars"] >= 0,
                f"Skill eval row {idx} has invalid analysis_max_chars",
            )

    validate_harmony_rows(
        harmony_rows,
        label="Harmony SFT",
        require_unique=bool(quality_gates.get("require_unique_harmony_examples")),
    )
    skill_category_counts = validate_harmony_rows(
        skill_compliance_rows,
        label="Skill compliance",
        require_unique=bool(quality_gates.get("require_unique_skill_compliance_examples")),
        require_category=True,
        allowed_categories=required_skill_categories or None,
    )
    if required_skill_categories:
        missing_categories = sorted(required_skill_categories - set(skill_category_counts))
        ensure(not missing_categories, f"Skill compliance dataset missing categories: {missing_categories}")
    min_examples_per_category = quality_gates.get("min_examples_per_skill_compliance_category")
    if min_examples_per_category:
        for category in sorted(skill_category_counts):
            ensure(
                skill_category_counts[category] >= int(min_examples_per_category),
                f"Skill compliance category '{category}' is below minimum size gate",
            )

    prepared_rows: dict[str, list[dict]] = {}
    for key, rel_path in prepared_datasets.items():
        path = bundle_dir / rel_path
        ensure(path.is_file(), f"Missing prepared dataset '{key}'")
        prepared_rows[key] = read_jsonl(path)

    if {"skill_compliance_train_dataset", "skill_compliance_eval_dataset"} <= set(prepared_rows):
        skill_train_hashes = {
            canonical_messages_hash(row["messages"]) for row in prepared_rows["skill_compliance_train_dataset"]
        }
        skill_eval_hashes = {
            canonical_messages_hash(row["messages"]) for row in prepared_rows["skill_compliance_eval_dataset"]
        }
        ensure(
            skill_train_hashes.isdisjoint(skill_eval_hashes),
            "Prepared skill-compliance train/eval datasets overlap",
        )
        ensure(
            len(prepared_rows["skill_compliance_train_dataset"]) + len(prepared_rows["skill_compliance_eval_dataset"])
            == len(skill_compliance_rows),
            "Prepared skill-compliance train/eval counts do not match source dataset",
        )

    if {"base_train_dataset", "base_eval_dataset"} <= set(prepared_rows):
        base_train_hashes = {canonical_messages_hash(row["messages"]) for row in prepared_rows["base_train_dataset"]}
        base_eval_hashes = {canonical_messages_hash(row["messages"]) for row in prepared_rows["base_eval_dataset"]}
        ensure(
            base_train_hashes.isdisjoint(base_eval_hashes),
            "Prepared base train/eval datasets overlap",
        )
        ensure(
            len(prepared_rows["base_train_dataset"]) + len(prepared_rows["base_eval_dataset"]) == len(harmony_rows),
            "Prepared base train/eval counts do not match source harmony dataset",
        )

    if {"train_dataset", "eval_dataset"} <= set(prepared_rows):
        prepared_train_hashes = {canonical_messages_hash(row["messages"]) for row in prepared_rows["train_dataset"]}
        prepared_eval_hashes = {canonical_messages_hash(row["messages"]) for row in prepared_rows["eval_dataset"]}
        ensure(
            prepared_train_hashes.isdisjoint(prepared_eval_hashes),
            "Prepared combined train/eval datasets overlap",
        )
        if {"base_train_dataset", "skill_compliance_train_dataset"} <= set(prepared_rows):
            ensure(
                len(prepared_rows["train_dataset"])
                == len(prepared_rows["base_train_dataset"]) + len(prepared_rows["skill_compliance_train_dataset"]),
                "Prepared combined train dataset count does not match base+skill train datasets",
            )
        if {"base_eval_dataset", "skill_compliance_eval_dataset"} <= set(prepared_rows):
            ensure(
                len(prepared_rows["eval_dataset"])
                == len(prepared_rows["base_eval_dataset"]) + len(prepared_rows["skill_compliance_eval_dataset"]),
                "Prepared combined eval dataset count does not match base+skill eval datasets",
            )
        ensure(
            len(prepared_rows["train_dataset"]) + len(prepared_rows["eval_dataset"])
            == len(harmony_rows) + len(skill_compliance_rows),
            "Prepared combined train/eval totals do not match source harmony+skill datasets",
        )

    if "min_sft_examples" in quality_gates:
        ensure(len(sft_rows) >= int(quality_gates["min_sft_examples"]), "SFT dataset below minimum size gate")
    if "min_harmony_sft_examples" in quality_gates:
        ensure(
            len(harmony_rows) >= int(quality_gates["min_harmony_sft_examples"]),
            "Harmony SFT dataset below minimum size gate",
        )
    if "min_skill_compliance_examples" in quality_gates:
        ensure(
            len(skill_compliance_rows) >= int(quality_gates["min_skill_compliance_examples"]),
            "Skill compliance dataset below minimum size gate",
        )
    if "min_eval_cases" in quality_gates:
        ensure(len(eval_rows) >= int(quality_gates["min_eval_cases"]), "Eval dataset below minimum size gate")
    if "min_skill_compliance_eval_cases" in quality_gates:
        ensure(
            len(skill_eval_rows) >= int(quality_gates["min_skill_compliance_eval_cases"]),
            "Skill compliance eval dataset below minimum size gate",
        )
    if "min_balanced_train_examples" in quality_gates and "balanced_train_dataset" in prepared_rows:
        ensure(
            len(prepared_rows["balanced_train_dataset"]) >= int(quality_gates["min_balanced_train_examples"]),
            "Balanced train dataset below minimum size gate",
        )
    if "min_balanced_eval_examples" in quality_gates and "balanced_eval_dataset" in prepared_rows:
        ensure(
            len(prepared_rows["balanced_eval_dataset"]) >= int(quality_gates["min_balanced_eval_examples"]),
            "Balanced eval dataset below minimum size gate",
        )
    if promotion_policy:
        ensure(
            promotion_policy.get("schema_version") == "dtf-promotion-policy/v1",
            "promotion_policy.json has unsupported schema_version",
        )
        claim_levels = promotion_policy.get("claim_levels", {})
        ensure(isinstance(claim_levels, dict) and claim_levels, "promotion_policy.json missing claim_levels")
        for level in ("runtime-only", "training-ready", "learned-only-after-training", "weight-level-verified"):
            ensure(level in claim_levels, f"promotion_policy.json missing claim level: {level}")
            requires = claim_levels[level].get("requires", [])
            ensure(
                isinstance(requires, list) and requires and all(isinstance(item, str) and item for item in requires),
                f"promotion_policy.json claim level '{level}' has invalid requires",
            )

    return {
        "bundle": profile["name"],
        "target_model": profile["target_model"],
        "system_prompt_chars": len(system_prompt),
        "sft_examples": len(sft_rows),
        "harmony_sft_examples": len(harmony_rows),
        "skill_compliance_examples": len(skill_compliance_rows),
        "skill_compliance_categories": dict(skill_category_counts),
        "eval_cases": len(eval_rows),
        "skill_compliance_eval_cases": len(skill_eval_rows),
        "promotion_policy_path": profile["files"].get("promotion_policy", ""),
        "prepared_train_dataset": prepared_datasets.get("train_dataset", ""),
        "prepared_eval_dataset": prepared_datasets.get("eval_dataset", ""),
        "prepared_balanced_train_dataset": prepared_datasets.get("balanced_train_dataset", ""),
        "prepared_balanced_eval_dataset": prepared_datasets.get("balanced_eval_dataset", ""),
    }


def main() -> int:
    args = parse_args()
    summary = validate_bundle(Path(args.bundle).resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

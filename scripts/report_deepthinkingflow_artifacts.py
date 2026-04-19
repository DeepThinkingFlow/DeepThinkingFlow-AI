#!/usr/bin/env python3
"""Hash DeepThinkingFlow artifacts and classify the strongest supportable claim level."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deepthinkingflow_json_io import load_json_file

SCHEMA_VERSION = "dtf-artifact-report/v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report hashes for base weights, adapter artifacts, eval outputs, and claim level."
    )
    parser.add_argument("--base-weights", default="", help="Base weight file or directory.")
    parser.add_argument("--training-config", default="", help="Training config JSON used to produce the adapter.")
    parser.add_argument("--train-dataset", default="", help="Train dataset artifact used for the run.")
    parser.add_argument("--eval-dataset", default="", help="Eval dataset artifact used for the run.")
    parser.add_argument("--behavior-bundle", default="", help="Behavior bundle directory used for runtime/training alignment.")
    parser.add_argument("--adapter-dir", default="", help="LoRA or QLoRA adapter directory.")
    parser.add_argument("--eval-output", default="", help="Eval output JSON/JSONL file.")
    parser.add_argument("--compare-report", default="", help="Optional before/after compare report.")
    parser.add_argument("--output", default="", help="Optional path to write the JSON report.")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def collect_path_report(path: Path) -> dict[str, Any]:
    if path.is_file():
        stat = path.stat()
        return {
            "type": "file",
            "path": str(path),
            "sha256": file_sha256(path),
            "size_bytes": stat.st_size,
        }
    if path.is_dir():
        files = sorted(file_path for file_path in path.rglob("*") if file_path.is_file())
        digest = hashlib.sha256()
        file_reports = []
        for file_path in files:
            rel_path = str(file_path.relative_to(path))
            sha = file_sha256(file_path)
            stat = file_path.stat()
            digest.update(rel_path.encode("utf-8"))
            digest.update(sha.encode("utf-8"))
            file_reports.append(
                {
                    "path": rel_path,
                    "sha256": sha,
                    "size_bytes": stat.st_size,
                }
            )
        return {
            "type": "directory",
            "path": str(path),
            "sha256": digest.hexdigest(),
            "file_count": len(file_reports),
            "files": file_reports,
        }
    raise ValueError(f"Artifact path does not exist: {path}")


def detect_claim_level(base_weights: dict[str, Any] | None, adapter_dir: dict[str, Any] | None, eval_output: dict[str, Any] | None, compare_report: dict[str, Any] | None) -> str:
    if base_weights and adapter_dir and eval_output and compare_report:
        return "learned-only-after-training"
    if base_weights and eval_output and compare_report and not adapter_dir:
        return "weight-level-verified"
    if adapter_dir or eval_output:
        return "training-ready"
    return "runtime-only"


def load_training_config(path: Path) -> dict[str, Any]:
    return load_json_file(path, "training config")


def load_optional_artifact_json(raw_path: str) -> dict[str, Any] | None:
    if not raw_path:
        return None
    path = Path(raw_path).resolve()
    if not path.is_file():
        return None
    return load_json_file(path, "artifact json")


def build_lineage_status(
    *,
    training_config_payload: dict[str, Any] | None,
    train_dataset: dict[str, Any] | None,
    eval_dataset: dict[str, Any] | None,
    behavior_bundle: dict[str, Any] | None,
    base_weights: dict[str, Any] | None,
    adapter_dir: dict[str, Any] | None,
    eval_output: dict[str, Any] | None,
) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "training_config_present": training_config_payload is not None,
        "train_dataset_present": train_dataset is not None,
        "eval_dataset_present": eval_dataset is not None,
        "behavior_bundle_present": behavior_bundle is not None,
        "base_weights_present": base_weights is not None,
        "adapter_dir_present": adapter_dir is not None,
        "eval_output_present": eval_output is not None,
        "config_dataset_match": False,
        "config_eval_dataset_match": False,
        "config_bundle_match": False,
    }
    if training_config_payload is not None:
        configured_dataset = str(training_config_payload.get("dataset_path", "")).strip()
        configured_eval_dataset = str(training_config_payload.get("eval_dataset_path", "")).strip()
        configured_bundle = str(training_config_payload.get("behavior_bundle_dir", "")).strip()
        if train_dataset is not None:
            checks["config_dataset_match"] = Path(configured_dataset).resolve() == Path(train_dataset["path"]).resolve()
        if eval_dataset is not None and configured_eval_dataset:
            checks["config_eval_dataset_match"] = Path(configured_eval_dataset).resolve() == Path(eval_dataset["path"]).resolve()
        if behavior_bundle is not None and configured_bundle:
            checks["config_bundle_match"] = Path(configured_bundle).resolve() == Path(behavior_bundle["path"]).resolve()

    checks["lineage_complete_for_training_claim"] = all(
        [
            checks["training_config_present"],
            checks["train_dataset_present"],
            checks["behavior_bundle_present"],
            checks["base_weights_present"],
            checks["config_dataset_match"],
            checks["config_bundle_match"],
        ]
    )
    checks["lineage_complete_for_learned_claim"] = all(
        [
            checks["lineage_complete_for_training_claim"],
            checks["eval_dataset_present"],
            checks["eval_output_present"],
            checks["adapter_dir_present"],
            checks["config_eval_dataset_match"],
        ]
    )
    return checks


def build_claim_evidence(
    *,
    base_weights: dict[str, Any] | None,
    adapter_dir: dict[str, Any] | None,
    eval_output: dict[str, Any] | None,
    compare_report: dict[str, Any] | None,
) -> dict[str, bool]:
    return {
        "has_base_weights": base_weights is not None,
        "has_adapter_dir": adapter_dir is not None,
        "has_eval_output": eval_output is not None,
        "has_compare_report": compare_report is not None,
        "base_checkpoint_alone_cannot_prove_skill_following": True,
    }


def build_quality_signals(
    *,
    eval_output_payload: dict[str, Any] | None,
    compare_report_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    trait_pass_rate = None if eval_output_payload is None else eval_output_payload.get("trait_pass_rate")
    rubric_pass_rate = None if eval_output_payload is None else eval_output_payload.get("rubric_pass_rate")
    compare_available = compare_report_payload is not None
    not_worse_trait = None if compare_report_payload is None else compare_report_payload.get(
        "candidate_is_not_worse_on_trait_pass_rate"
    )
    not_worse_rubric = None if compare_report_payload is None else compare_report_payload.get(
        "candidate_is_not_worse_on_rubric_pass_rate"
    )
    not_worse_every_case_trait = None if compare_report_payload is None else compare_report_payload.get(
        "candidate_is_not_worse_on_every_shared_case_trait_count"
    )
    not_worse_every_case_rubric = None if compare_report_payload is None else compare_report_payload.get(
        "candidate_is_not_worse_on_every_shared_case_rubric_count"
    )
    candidate_quality_is_non_regressing = bool(not_worse_trait) and bool(not_worse_rubric)
    return {
        "eval_output_parsed": eval_output_payload is not None,
        "compare_report_parsed": compare_available,
        "trait_pass_rate": trait_pass_rate,
        "rubric_pass_rate": rubric_pass_rate,
        "candidate_is_not_worse_on_trait_pass_rate": not_worse_trait,
        "candidate_is_not_worse_on_rubric_pass_rate": not_worse_rubric,
        "candidate_is_not_worse_on_every_shared_case_trait_count": not_worse_every_case_trait,
        "candidate_is_not_worse_on_every_shared_case_rubric_count": not_worse_every_case_rubric,
        "candidate_quality_is_non_regressing": candidate_quality_is_non_regressing if compare_available else None,
        "learned_claim_has_quality_regression_risk": compare_available and not candidate_quality_is_non_regressing,
        "semantic_skill_compliance_still_unproven": compare_available and not candidate_quality_is_non_regressing,
    }


def claim_notes(claim_level: str) -> list[str]:
    notes = {
        "runtime-only": [
            "No training artifact evidence was supplied.",
            "Do not claim learned behavior at the weight level from this report.",
        ],
        "training-ready": [
            "Some training-side evidence exists, but the report is still below learned-only-after-training.",
            "Do not claim base checkpoint adherence from this report alone.",
        ],
        "learned-only-after-training": [
            "Adapter-backed learned behavior is supportable when eval and compare artifacts are present.",
            "This still does not prove the untouched base checkpoint changed.",
        ],
        "weight-level-verified": [
            "This is the strongest claim level currently supported by the supplied artifacts.",
            "Use only when the report truly corresponds to a merged or replacement checkpoint plus eval evidence.",
        ],
    }
    return notes[claim_level]


def maybe_collect(raw_path: str) -> dict[str, Any] | None:
    if not raw_path:
        return None
    return collect_path_report(Path(raw_path).resolve())


def main() -> int:
    args = parse_args()
    base_weights = maybe_collect(args.base_weights)
    training_config = maybe_collect(args.training_config)
    train_dataset = maybe_collect(args.train_dataset)
    eval_dataset = maybe_collect(args.eval_dataset)
    behavior_bundle = maybe_collect(args.behavior_bundle)
    adapter_dir = maybe_collect(args.adapter_dir)
    eval_output = maybe_collect(args.eval_output)
    compare_report = maybe_collect(args.compare_report)
    training_config_payload = load_training_config(Path(args.training_config).resolve()) if args.training_config else None
    eval_output_payload = load_optional_artifact_json(args.eval_output)
    compare_report_payload = load_optional_artifact_json(args.compare_report)
    claim_level = detect_claim_level(base_weights, adapter_dir, eval_output, compare_report)
    lineage_status = build_lineage_status(
        training_config_payload=training_config_payload,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        behavior_bundle=behavior_bundle,
        base_weights=base_weights,
        adapter_dir=adapter_dir,
        eval_output=eval_output,
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_weights": base_weights,
        "training_config": training_config,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "behavior_bundle": behavior_bundle,
        "adapter_dir": adapter_dir,
        "eval_output": eval_output,
        "compare_report": compare_report,
        "lineage_status": lineage_status,
        "claim_level": claim_level,
        "claim_evidence": build_claim_evidence(
            base_weights=base_weights,
            adapter_dir=adapter_dir,
            eval_output=eval_output,
            compare_report=compare_report,
        ),
        "quality_signals": build_quality_signals(
            eval_output_payload=eval_output_payload,
            compare_report_payload=compare_report_payload,
        ),
        "claim_notes": claim_notes(claim_level),
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

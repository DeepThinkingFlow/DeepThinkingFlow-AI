#!/usr/bin/env python3
"""Check whether supplied DeepThinkingFlow evidence satisfies the promotion policy for a claim level."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from deepthinkingflow_json_io import load_json_file, now_utc_iso, write_json_file

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT_DIR / "behavior" / "DeepThinkingFlow"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check claim-level promotion readiness against the DeepThinkingFlow promotion policy."
    )
    parser.add_argument(
        "--bundle",
        default=str(DEFAULT_BUNDLE),
        help="Behavior bundle directory containing profile.json and promotion_policy.json.",
    )
    parser.add_argument(
        "--claim-level",
        required=True,
        choices=["runtime-only", "training-ready", "learned-only-after-training", "weight-level-verified"],
        help="Claim level to validate.",
    )
    parser.add_argument("--verify-report", default="", help="Optional verify report JSON path.")
    parser.add_argument("--artifact-report", default="", help="Optional artifact report JSON path.")
    parser.add_argument("--release-manifest", default="", help="Optional release manifest JSON path.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    return parser.parse_args()


def load_optional_json(path: str, label: str) -> dict[str, Any] | None:
    raw = path.strip()
    if not raw:
        return None
    return load_json_file(Path(raw).resolve(), label)


def resolve_bundle(bundle: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    profile = load_json_file(bundle / "profile.json", "profile.json")
    promotion_policy_rel = profile.get("files", {}).get("promotion_policy", "")
    if not promotion_policy_rel:
        raise SystemExit("profile.json does not define files.promotion_policy")
    promotion_policy = load_json_file(bundle / promotion_policy_rel, "promotion_policy.json")
    return profile, promotion_policy


def build_evidence(
    *,
    verify_payload: dict[str, Any] | None,
    artifact_payload: dict[str, Any] | None,
    release_manifest_payload: dict[str, Any] | None,
) -> dict[str, bool]:
    verify_state = {} if verify_payload is None else verify_payload.get("verified", {})
    bundle_valid = bool(verify_state.get("bundle_valid", False))
    verify_gate = {} if verify_payload is None else verify_payload.get("claim_gate", {})
    artifact_lineage = {} if artifact_payload is None else artifact_payload.get("lineage_status", {})
    quality = {} if artifact_payload is None else artifact_payload.get("quality_signals", {})
    release_state = {} if release_manifest_payload is None else release_manifest_payload.get("release_state", {})
    return {
        "valid_behavior_bundle": bundle_valid,
        "verified_runtime_boundary": bool(verify_gate.get("passed", False)),
        "prepared_train_dataset": bool(artifact_lineage.get("config_dataset_match", False)),
        "prepared_eval_dataset": bool(artifact_lineage.get("config_eval_dataset_match", False)),
        "training_config": bool(artifact_lineage.get("training_config_present", False)),
        "base_weight_hash": bool(artifact_payload and artifact_payload.get("base_weights", {}).get("sha256")),
        "adapter_hash": bool(artifact_payload and artifact_payload.get("adapter_dir", {}).get("sha256")),
        "merged_or_replacement_weight_hash": bool(artifact_payload and artifact_payload.get("base_weights", {}).get("sha256")),
        "eval_output": bool(artifact_payload and artifact_payload.get("eval_output", {}).get("sha256")),
        "compare_report": bool(artifact_payload and artifact_payload.get("compare_report", {}).get("sha256")),
        "lineage_complete_for_training_claim": bool(artifact_lineage.get("lineage_complete_for_training_claim", False)),
        "lineage_complete_for_learned_claim": bool(artifact_lineage.get("lineage_complete_for_learned_claim", False)),
        "candidate_quality_is_non_regressing": bool(quality.get("candidate_quality_is_non_regressing", False)),
        "golden_release_gate_passed": bool(release_state.get("golden_release_gate_passed", False)),
    }


def evaluate_readiness(claim_level: str, promotion_policy: dict[str, Any], evidence: dict[str, bool]) -> dict[str, Any]:
    claim_levels = promotion_policy.get("claim_levels", {})
    policy = claim_levels.get(claim_level)
    if not isinstance(policy, dict):
        raise SystemExit(f"Promotion policy missing claim level: {claim_level}")
    required_checks = list(policy.get("requires", []))
    missing = [check for check in required_checks if not evidence.get(check, False)]
    hard_failures: list[str] = []
    if claim_level in {"learned-only-after-training", "weight-level-verified"}:
        if not evidence.get("compare_report", False):
            hard_failures.append("missing_compare_report_for_learned_claim")
        if not evidence.get("eval_output", False):
            hard_failures.append("missing_eval_output_for_learned_claim")
        if not evidence.get("candidate_quality_is_non_regressing", False):
            hard_failures.append("quality_regression_on_learned_claim")
        if not evidence.get("lineage_complete_for_learned_claim", False):
            hard_failures.append("incomplete_lineage_for_claim")
    return {
        "claim_level": claim_level,
        "required_checks": required_checks,
        "missing_requirements": missing,
        "hard_failures": hard_failures,
        "ready": not missing and not hard_failures,
    }


def main() -> int:
    args = parse_args()
    bundle = Path(args.bundle).resolve()
    profile, promotion_policy = resolve_bundle(bundle)
    verify_payload = load_optional_json(args.verify_report, "verify report")
    artifact_payload = load_optional_json(args.artifact_report, "artifact report")
    release_manifest_payload = load_optional_json(args.release_manifest, "release manifest")
    evidence = build_evidence(
        verify_payload=verify_payload,
        artifact_payload=artifact_payload,
        release_manifest_payload=release_manifest_payload,
    )
    readiness = evaluate_readiness(args.claim_level, promotion_policy, evidence)
    payload = {
        "schema_version": "dtf-promotion-readiness/v1",
        "generated_at_utc": now_utc_iso(),
        "bundle": str(bundle),
        "policy_name": promotion_policy.get("policy_name", ""),
        "target_model": profile.get("target_model", ""),
        "evidence": evidence,
        "readiness": readiness,
    }
    if args.output:
        write_json_file(Path(args.output).resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

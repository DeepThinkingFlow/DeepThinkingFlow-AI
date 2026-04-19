#!/usr/bin/env python3
"""Production-style health report for the full DeepThinkingFlow project."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from deepthinkingflow_exit_codes import OK, PRECONDITION_FAILED
from deepthinkingflow_json_io import load_json_file, now_utc_iso, run_json_command

ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a high-signal DeepThinkingFlow health check for runtime, training, and release readiness."
    )
    parser.add_argument(
        "--bundle",
        default="behavior/DeepThinkingFlow",
        help="Behavior bundle directory.",
    )
    parser.add_argument(
        "--model-dir",
        default="runtime/transformers/DeepThinkingFlow",
        help="Primary runtime model directory.",
    )
    parser.add_argument(
        "--training-config",
        default="training/DeepThinkingFlow-lora/config.example.json",
        help="Training config used for project diagnostics.",
    )
    parser.add_argument(
        "--verify-report",
        default="",
        help="Optional existing verify report. When omitted, doctor will generate one with --skip-tests.",
    )
    parser.add_argument(
        "--artifact-report",
        default="",
        help="Optional artifact report used to assess claim readiness.",
    )
    return parser.parse_args()


def summarize_doctor(
    *,
    verify_payload: dict[str, Any],
    artifact_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    ready = verify_payload.get("project_preflight", {}).get("ready", {})
    claim_gate = verify_payload.get("claim_gate", {})
    lineage = {} if artifact_payload is None else artifact_payload.get("lineage_status", {})
    issues: list[str] = []

    if not verify_payload.get("verified", {}).get("bundle_valid", False):
        issues.append("Behavior bundle is not valid.")
    if not verify_payload.get("verified", {}).get("claim_gate_passed", False):
        issues.append("Claim gate is failing.")
    if not ready.get("inference_soft_gate_clear", False):
        issues.append("Inference soft gate is not clear on the current host.")
    if not ready.get("training_soft_gate_clear", False):
        issues.append("Training soft gate is not clear on the current host.")
    if not ready.get("training_locally_feasible", False):
        issues.append("Configured local 20B training is not feasible on the current host.")
    if artifact_payload is not None and not lineage.get("lineage_complete_for_training_claim", False):
        issues.append("Artifact lineage is incomplete for training-side claims.")
    quality = {} if artifact_payload is None else artifact_payload.get("quality_signals", {})
    if artifact_payload is not None and quality.get("candidate_quality_is_non_regressing") is False:
        issues.append("Artifact compare report shows quality regression risk.")

    return {
        "generated_at_utc": now_utc_iso(),
        "project_ready_for_runtime_only_release": bool(
            verify_payload.get("verified", {}).get("bundle_valid", False)
            and verify_payload.get("verified", {}).get("claim_gate_passed", False)
        ),
        "project_ready_for_local_host_training": bool(ready.get("training_locally_feasible", False)),
        "artifact_lineage_complete_for_training_claim": bool(lineage.get("lineage_complete_for_training_claim", False)),
        "artifact_lineage_complete_for_learned_claim": bool(lineage.get("lineage_complete_for_learned_claim", False)),
        "claim_level": "runtime-only" if artifact_payload is None else artifact_payload.get("claim_level", "runtime-only"),
        "candidate_quality_is_non_regressing": quality.get("candidate_quality_is_non_regressing"),
        "semantic_skill_compliance_still_unproven": quality.get("semantic_skill_compliance_still_unproven"),
        "claim_gate": claim_gate,
        "issues": issues,
    }


def main() -> int:
    args = parse_args()
    verify_payload = (
        load_json_file(Path(args.verify_report).resolve(), "verify report")
        if args.verify_report
        else run_json_command(
            [
                sys.executable,
                str((ROOT_DIR / "scripts" / "verify_deepthinkingflow_project.py").resolve()),
                "--bundle",
                str(Path(args.bundle).resolve()),
                "--model-dir",
                str(Path(args.model_dir).resolve()),
                "--training-config",
                str(Path(args.training_config).resolve()),
                "--skip-tests",
                "--require-claim-level",
                "runtime-only",
            ],
            cwd=ROOT_DIR,
            label="verify health check",
        )
    )
    artifact_payload = load_json_file(Path(args.artifact_report).resolve(), "artifact report") if args.artifact_report else None
    summary = {
        "schema_version": "dtf-doctor-report/v1",
        "root_dir": str(ROOT_DIR),
        "verify_report": verify_payload,
        "artifact_report": artifact_payload,
        "doctor": summarize_doctor(
            verify_payload=verify_payload,
            artifact_payload=artifact_payload,
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["doctor"]["issues"]:
        return PRECONDITION_FAILED
    return OK


if __name__ == "__main__":
    raise SystemExit(main())

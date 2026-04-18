#!/usr/bin/env python3
"""Production-style health report for the full DeepThinkingFlow project."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deepthinkingflow_exit_codes import OK, PRECONDITION_FAILED

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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_json(command: list[str], label: str) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=str(ROOT_DIR),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(f"{label} failed with code {completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}")
    return json.loads(completed.stdout)


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

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_ready_for_runtime_only_release": bool(
            verify_payload.get("verified", {}).get("bundle_valid", False)
            and verify_payload.get("verified", {}).get("claim_gate_passed", False)
        ),
        "project_ready_for_local_host_training": bool(ready.get("training_locally_feasible", False)),
        "artifact_lineage_complete_for_training_claim": bool(lineage.get("lineage_complete_for_training_claim", False)),
        "artifact_lineage_complete_for_learned_claim": bool(lineage.get("lineage_complete_for_learned_claim", False)),
        "claim_level": "runtime-only" if artifact_payload is None else artifact_payload.get("claim_level", "runtime-only"),
        "claim_gate": claim_gate,
        "issues": issues,
    }


def main() -> int:
    args = parse_args()
    verify_payload = (
        load_json(Path(args.verify_report).resolve())
        if args.verify_report
        else run_json(
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
            "verify health check",
        )
    )
    artifact_payload = load_json(Path(args.artifact_report).resolve()) if args.artifact_report else None
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

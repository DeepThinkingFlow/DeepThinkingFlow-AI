#!/usr/bin/env python3
"""Run a consolidated verification suite for DeepThinkingFlow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import preflight_deepthinkingflow_project as project_preflight
import validate_behavior_bundle as bundle_validator
from deepthinkingflow_exit_codes import INVALID_ARTIFACT, OK

ROOT_DIR = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "dtf-verify-report/v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bundle validation, project preflight, and smoke tests in one command."
    )
    parser.add_argument(
        "--bundle",
        default="behavior/DeepThinkingFlow",
        help="Behavior bundle directory to validate.",
    )
    parser.add_argument(
        "--model-dir",
        default="runtime/transformers/DeepThinkingFlow",
        help="Model directory used for preflight checks.",
    )
    parser.add_argument(
        "--training-config",
        default="training/DeepThinkingFlow-lora/config.example.json",
        help="Training config used for project preflight.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip the Python smoke suite.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to write the verification JSON report.",
    )
    parser.add_argument(
        "--artifact-report",
        default="",
        help="Optional artifact report JSON used to enforce claim-quality gates.",
    )
    parser.add_argument(
        "--require-claim-level",
        default="runtime-only",
        choices=["runtime-only", "training-ready", "learned-only-after-training", "weight-level-verified"],
        help="Minimum claim level the supplied artifacts must support.",
    )
    return parser.parse_args()


def run_subprocess(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=str(ROOT_DIR),
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def summarize_environment() -> dict[str, str]:
    return {
        "python_executable": sys.executable,
        "root_dir": str(ROOT_DIR),
    }


CLAIM_LEVEL_ORDER = {
    "runtime-only": 0,
    "training-ready": 1,
    "learned-only-after-training": 2,
    "weight-level-verified": 3,
}


def load_optional_json(path: str) -> dict[str, Any] | None:
    raw = path.strip()
    if not raw:
        return None
    resolved = Path(raw).resolve()
    if not resolved.is_file():
        raise SystemExit(f"Missing artifact report: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def evaluate_claim_gate(
    *,
    required_claim_level: str,
    artifact_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    actual_claim_level = "runtime-only" if artifact_payload is None else str(artifact_payload.get("claim_level", "runtime-only"))
    lineage = {} if artifact_payload is None else artifact_payload.get("lineage_status", {})
    gate = {
        "required_claim_level": required_claim_level,
        "actual_claim_level": actual_claim_level,
        "artifact_report_available": artifact_payload is not None,
        "artifact_lineage_complete_for_training_claim": bool(lineage.get("lineage_complete_for_training_claim", False)),
        "artifact_lineage_complete_for_learned_claim": bool(lineage.get("lineage_complete_for_learned_claim", False)),
        "passed": False,
        "reasons": [],
    }
    if CLAIM_LEVEL_ORDER[actual_claim_level] < CLAIM_LEVEL_ORDER[required_claim_level]:
        gate["reasons"].append(
            f"Actual claim level {actual_claim_level} is below required {required_claim_level}."
        )
    if required_claim_level in {"training-ready", "learned-only-after-training", "weight-level-verified"} and not gate["artifact_lineage_complete_for_training_claim"]:
        gate["reasons"].append("Artifact lineage is incomplete for a training-side claim.")
    if required_claim_level in {"learned-only-after-training", "weight-level-verified"} and not gate["artifact_lineage_complete_for_learned_claim"]:
        gate["reasons"].append("Artifact lineage is incomplete for a learned-behavior claim.")
    gate["passed"] = not gate["reasons"]
    return gate


def main() -> int:
    args = parse_args()
    bundle_dir = Path(args.bundle).resolve()

    bundle_summary = bundle_validator.validate_bundle(bundle_dir)

    preflight_command = [
        sys.executable,
        str((ROOT_DIR / "scripts" / "preflight_deepthinkingflow_project.py").resolve()),
        "--bundle",
        str(bundle_dir),
        "--model-dir",
        str(Path(args.model_dir).resolve()),
        "--training-config",
        str(Path(args.training_config).resolve()),
    ]
    preflight_run = run_subprocess(preflight_command)
    if preflight_run["returncode"] != 0:
        raise SystemExit("preflight-all failed during verification.")
    preflight_payload = json.loads(preflight_run["stdout"])

    tests_run = None
    if not args.skip_tests:
        tests_command = [
            sys.executable,
            "-m",
            "unittest",
            "tests.test_deepthinkingflow_smoke",
        ]
        tests_run = run_subprocess(tests_command)
        if tests_run["returncode"] != 0:
            raise SystemExit("Smoke test suite failed during verification.")

    artifact_payload = load_optional_json(args.artifact_report)
    claim_gate = evaluate_claim_gate(
        required_claim_level=args.require_claim_level,
        artifact_payload=artifact_payload,
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": summarize_environment(),
        "bundle_validation": bundle_summary,
        "commands": {
            "preflight": preflight_run["command"],
            "tests": tests_run["command"] if tests_run is not None else None,
        },
        "project_preflight": preflight_payload,
        "artifact_report": artifact_payload,
        "claim_gate": claim_gate,
        "tests": tests_run,
        "verified": {
            "bundle_valid": True,
            "preflight_ran": True,
            "tests_passed": tests_run is None or tests_run["returncode"] == 0,
            "claim_gate_passed": claim_gate["passed"],
        },
    }

    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not claim_gate["passed"]:
        return INVALID_ARTIFACT
    return OK


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a release-oriented manifest summarizing bundle, runtime, artifact, and verification state."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deepthinkingflow_exit_codes import INVALID_ARTIFACT, OK, VERIFICATION_FAILED

ROOT_DIR = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "dtf-release-manifest/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a release manifest from verify and artifact reports."
    )
    parser.add_argument("--verify-report", default="", help="Optional existing verify report JSON path.")
    parser.add_argument("--artifact-report", default="", help="Optional existing artifact report JSON path.")
    parser.add_argument("--output", required=True, help="Target JSON path for the release manifest.")
    parser.add_argument("--release-id", default="", help="Optional release id. Auto-generated when omitted.")
    return parser.parse_args()


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Missing {label}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_json_command(command: list[str], label: str) -> dict[str, Any]:
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


def default_release_id() -> str:
    return datetime.now(timezone.utc).strftime("dtf-release-%Y%m%dT%H%M%SZ")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_report_ref(path: Path | None, payload: dict[str, Any], label: str) -> dict[str, Any]:
    reference = {
        "label": label,
        "schema_version": payload.get("schema_version", ""),
        "generated_at_utc": payload.get("generated_at_utc", ""),
    }
    if path is not None:
        reference["path"] = str(path)
        reference["sha256"] = file_sha256(path)
        reference["size_bytes"] = path.stat().st_size
    else:
        reference["path"] = ""
        reference["sha256"] = ""
        reference["size_bytes"] = 0
    return reference


def main() -> int:
    args = parse_args()
    verify_path = Path(args.verify_report).resolve() if args.verify_report else None
    artifact_path = Path(args.artifact_report).resolve() if args.artifact_report else None
    verify_payload = (
        load_json(verify_path, "verify report")
        if verify_path is not None
        else run_json_command(
            [sys.executable, str((ROOT_DIR / "scripts" / "verify_deepthinkingflow_project.py").resolve()), "--skip-tests"],
            "verify report generation",
        )
    )
    artifact_payload = (
        load_json(artifact_path, "artifact report")
        if artifact_path is not None
        else None
    )
    claim_gate = verify_payload.get("claim_gate", {})
    if not bool(claim_gate.get("passed", True)):
        raise SystemExit(
            "verify report claim gate did not pass. Refusing to build a higher-confidence release manifest."
        )

    preflight_ready = verify_payload.get("project_preflight", {}).get("ready", {})
    release_candidate = bool(
        verify_payload["verified"]["bundle_valid"]
        and verify_payload["verified"]["preflight_ran"]
        and verify_payload["verified"]["tests_passed"]
    )
    local_host_ready = bool(
        preflight_ready.get("bundle_valid", False)
        and preflight_ready.get("inference_soft_gate_clear", False)
        and preflight_ready.get("training_soft_gate_clear", False)
        and preflight_ready.get("training_locally_feasible", False)
    )

    release_manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_id": args.release_id.strip() or default_release_id(),
        "root_dir": str(ROOT_DIR),
        "verify_report_ref": build_report_ref(verify_path, verify_payload, "verify-report"),
        "artifact_report_ref": build_report_ref(artifact_path, artifact_payload, "artifact-report") if artifact_payload else None,
        "verify_report": verify_payload,
        "artifact_report": artifact_payload,
        "release_state": {
            "bundle_valid": verify_payload["verified"]["bundle_valid"],
            "preflight_ran": verify_payload["verified"]["preflight_ran"],
            "tests_passed": verify_payload["verified"]["tests_passed"],
            "claim_gate_passed": bool(claim_gate.get("passed", True)),
            "claim_level": artifact_payload["claim_level"] if artifact_payload else "runtime-only",
            "release_candidate": release_candidate,
            "local_host_ready": local_host_ready,
        },
        "compatibility": {
            "runtime_only_boundary_enforced": True,
            "weight_level_adherence_requires_training_artifacts": True,
            "base_checkpoint_alone_is_not_release_evidence": True,
        },
        "evidence_summary": {
            "verify_report_available": True,
            "artifact_report_available": artifact_payload is not None,
            "base_checkpoint_alone_cannot_prove_skill_following": True,
            "artifact_lineage_complete_for_training_claim": (
                artifact_payload.get("lineage_status", {}).get("lineage_complete_for_training_claim", False)
                if artifact_payload
                else False
            ),
            "artifact_lineage_complete_for_learned_claim": (
                artifact_payload.get("lineage_status", {}).get("lineage_complete_for_learned_claim", False)
                if artifact_payload
                else False
            ),
        },
    }

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(release_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(release_manifest, ensure_ascii=False, indent=2))
    return OK


if __name__ == "__main__":
    raise SystemExit(main())

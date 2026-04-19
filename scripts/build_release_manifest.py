#!/usr/bin/env python3
"""Build a release-oriented manifest summarizing bundle, runtime, artifact, and verification state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from deepthinkingflow_exit_codes import OK
from deepthinkingflow_json_io import file_sha256, load_json_file, now_utc_iso, run_json_command, write_json_file

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


def default_release_id() -> str:
    return now_utc_iso().replace("-", "").replace(":", "").replace("+00:00", "Z")


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
        load_json_file(verify_path, "verify report")
        if verify_path is not None
        else run_json_command(
            [sys.executable, str((ROOT_DIR / "scripts" / "verify_deepthinkingflow_project.py").resolve()), "--skip-tests"],
            cwd=ROOT_DIR,
            label="verify report generation",
        )
    )
    artifact_payload = (
        load_json_file(artifact_path, "artifact report")
        if artifact_path is not None
        else None
    )
    claim_gate = verify_payload.get("claim_gate", {})
    if not bool(claim_gate.get("passed", True)):
        raise SystemExit(
            "verify report claim gate did not pass. Refusing to build a higher-confidence release manifest."
        )
    golden_release_gate = {
        "passed": True,
        "reasons": [],
        "requires_tests_passed": True,
        "requires_bundle_valid": True,
        "requires_preflight": True,
        "requires_claim_gate": True,
        "requires_non_regressing_quality_for_learned_release": True,
    }

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
    if artifact_payload is not None and artifact_payload.get("claim_level") in {"learned-only-after-training", "weight-level-verified"}:
        quality = artifact_payload.get("quality_signals", {})
        if quality.get("candidate_quality_is_non_regressing") is not True:
            golden_release_gate["passed"] = False
            golden_release_gate["reasons"].append(
                "Artifact quality is regressing or missing compare-based non-regression evidence for a learned release."
            )
    if not release_candidate:
        golden_release_gate["passed"] = False
        golden_release_gate["reasons"].append("Core verification state is not release-candidate ready.")

    release_manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": now_utc_iso(),
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
            "golden_release_gate_passed": golden_release_gate["passed"],
        },
        "golden_release_gate": golden_release_gate,
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
    write_json_file(output_path, release_manifest)
    print(json.dumps(release_manifest, ensure_ascii=False, indent=2))
    return OK


if __name__ == "__main__":
    raise SystemExit(main())

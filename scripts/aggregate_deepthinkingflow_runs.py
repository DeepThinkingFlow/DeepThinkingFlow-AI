#!/usr/bin/env python3
"""Aggregate DeepThinkingFlow run artifacts into one lineage-oriented report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from deepthinkingflow_json_io import file_sha256, load_json_file, now_utc_iso, write_json_file

ROOT_DIR = Path(__file__).resolve().parents[1]
SUPPORTED_NAMES = {
    "artifact-report": "dtf-artifact-report",
    "verify-report": "dtf-verify-report",
    "release-manifest": "dtf-release-manifest",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate artifact, verify, and release reports into a lineage view."
    )
    parser.add_argument(
        "--search-root",
        default="out",
        help="Directory to scan recursively for DeepThinkingFlow JSON reports.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSON path.",
    )
    return parser.parse_args()


def detect_schema_family(payload: dict[str, Any]) -> str:
    schema_version = str(payload.get("schema_version", ""))
    for label, prefix in SUPPORTED_NAMES.items():
        if schema_version.startswith(prefix):
            return label
    return ""


def collect_reports(search_root: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    append_report = reports.append
    for path in sorted(search_root.rglob("*.json")):
        try:
            payload = load_json_file(path, "report")
        except Exception:
            continue
        family = detect_schema_family(payload)
        if not family:
            continue
        report = {
            "family": family,
            "path": str(path),
            "sha256": file_sha256(path),
            "generated_at_utc": payload.get("generated_at_utc", ""),
            "schema_version": payload.get("schema_version", ""),
        }
        if family == "artifact-report":
            report["claim_level"] = payload.get("claim_level", "runtime-only")
            report["lineage_complete_for_training_claim"] = bool(
                payload.get("lineage_status", {}).get("lineage_complete_for_training_claim", False)
            )
            report["lineage_complete_for_learned_claim"] = bool(
                payload.get("lineage_status", {}).get("lineage_complete_for_learned_claim", False)
            )
            report["candidate_quality_is_non_regressing"] = payload.get("quality_signals", {}).get(
                "candidate_quality_is_non_regressing"
            )
        elif family == "verify-report":
            report["claim_gate_passed"] = bool(payload.get("claim_gate", {}).get("passed", False))
            report["tests_passed"] = bool(payload.get("verified", {}).get("tests_passed", False))
        elif family == "release-manifest":
            report["release_candidate"] = bool(payload.get("release_state", {}).get("release_candidate", False))
            report["local_host_ready"] = bool(payload.get("release_state", {}).get("local_host_ready", False))
        append_report(report)
    return reports


def summarize_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    claim_levels = [report.get("claim_level", "") for report in reports if report["family"] == "artifact-report"]
    return {
        "report_count": len(reports),
        "artifact_report_count": sum(1 for report in reports if report["family"] == "artifact-report"),
        "verify_report_count": sum(1 for report in reports if report["family"] == "verify-report"),
        "release_manifest_count": sum(1 for report in reports if report["family"] == "release-manifest"),
        "claim_levels_seen": sorted(level for level in set(claim_levels) if level),
        "non_regressing_artifact_reports": sum(
            1
            for report in reports
            if report["family"] == "artifact-report" and report.get("candidate_quality_is_non_regressing") is True
        ),
        "release_candidates": sum(
            1 for report in reports if report["family"] == "release-manifest" and report.get("release_candidate")
        ),
    }


def main() -> int:
    args = parse_args()
    search_root = (ROOT_DIR / args.search_root).resolve()
    reports = collect_reports(search_root) if search_root.exists() else []
    payload = {
        "schema_version": "dtf-lineage-summary/v1",
        "generated_at_utc": now_utc_iso(),
        "root_dir": str(ROOT_DIR),
        "search_root": str(search_root),
        "summary": summarize_reports(reports),
        "reports": reports,
    }
    if args.output:
        write_json_file(Path(args.output).resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

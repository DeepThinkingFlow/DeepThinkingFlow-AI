#!/usr/bin/env python3
"""Run a tiny-smoke release lane with real adapter training and artifact production."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from deepthinkingflow_exit_codes import OK, VERIFICATION_FAILED

ROOT_DIR = Path(__file__).resolve().parents[1]


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (ROOT_DIR / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tiny DeepThinkingFlow release lane with real smoke training, artifact hashing, verify, and release manifest generation."
    )
    parser.add_argument(
        "--tiny-config",
        default="training/DeepThinkingFlow-lora/config.tiny-smoke.json",
        help="Tiny smoke training config.",
    )
    parser.add_argument(
        "--artifact-report",
        default="out/tiny-smoke-artifact-report.json",
        help="Artifact report output path.",
    )
    parser.add_argument(
        "--verify-report",
        default="out/tiny-smoke-verify-report.json",
        help="Verify report output path.",
    )
    parser.add_argument(
        "--release-manifest",
        default="out/tiny-smoke-release-manifest.json",
        help="Release manifest output path.",
    )
    return parser.parse_args()


def run_command(command: list[str], label: str) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=str(ROOT_DIR),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"{label} failed with code {completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    return {
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    tiny_config_path = Path(args.tiny_config).resolve()
    tiny_config = load_json(tiny_config_path)
    model_dir = resolve_repo_path(str(tiny_config["model_name_or_path"]))
    output_dir = resolve_repo_path(str(tiny_config["output_dir"]))
    artifact_report_path = Path(args.artifact_report).resolve()
    verify_report_path = Path(args.verify_report).resolve()
    release_manifest_path = Path(args.release_manifest).resolve()

    if not model_dir.exists():
        run_command(
            [
                sys.executable,
                str((ROOT_DIR / "scripts" / "create_tiny_gpt_oss_smoke_model.py").resolve()),
                "--output-dir",
                str(model_dir.relative_to(ROOT_DIR)),
            ],
            "tiny model creation",
        )

    run_command(
        [
            sys.executable,
            str((ROOT_DIR / "scripts" / "train_transformers_deepthinkingflow_lora.py").resolve()),
            "--config",
            str(tiny_config_path),
        ],
        "tiny smoke LoRA training",
    )

    run_command(
        [
            sys.executable,
            str((ROOT_DIR / "scripts" / "report_deepthinkingflow_artifacts.py").resolve()),
            "--base-weights",
            str(model_dir),
            "--training-config",
            str(tiny_config_path),
            "--train-dataset",
            str(resolve_repo_path(str(tiny_config["dataset_path"]))),
            "--eval-dataset",
            str(resolve_repo_path(str(tiny_config["eval_dataset_path"]))),
            "--behavior-bundle",
            str(resolve_repo_path(str(tiny_config["behavior_bundle_dir"]))),
            "--adapter-dir",
            str(output_dir),
            "--output",
            str(artifact_report_path),
        ],
        "artifact report generation",
    )

    run_command(
        [
            sys.executable,
            str((ROOT_DIR / "scripts" / "verify_deepthinkingflow_project.py").resolve()),
            "--training-config",
            str(tiny_config_path),
            "--artifact-report",
            str(artifact_report_path),
            "--require-claim-level",
            "training-ready",
            "--skip-tests",
            "--output",
            str(verify_report_path),
        ],
        "verify report generation",
    )

    run_command(
        [
            sys.executable,
            str((ROOT_DIR / "scripts" / "build_release_manifest.py").resolve()),
            "--verify-report",
            str(verify_report_path),
            "--artifact-report",
            str(artifact_report_path),
            "--output",
            str(release_manifest_path),
        ],
        "release manifest generation",
    )

    summary = {
        "schema_version": "dtf-tiny-smoke-release-lane/v1",
        "tiny_config": str(tiny_config_path),
        "artifact_report": str(artifact_report_path),
        "verify_report": str(verify_report_path),
        "release_manifest": str(release_manifest_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return OK


if __name__ == "__main__":
    raise SystemExit(main())

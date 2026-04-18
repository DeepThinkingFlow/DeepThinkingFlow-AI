#!/usr/bin/env python3
"""Run staged DeepThinkingFlow LoRA training with checkpoint resume."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT_DIR / "scripts" / "train_transformers_deepthinkingflow_lora.py"
DEFAULT_STAGE_CONFIGS = [
    "training/DeepThinkingFlow-lora/config.local-safe.stage1.json",
    "training/DeepThinkingFlow-lora/config.local-safe.stage2.json",
    "training/DeepThinkingFlow-lora/config.local-safe.stage3.json",
]
EXTERNAL_STAGE_CONFIGS = [
    "training/DeepThinkingFlow-lora/config.external-local-safe.stage1.json",
    "training/DeepThinkingFlow-lora/config.external-local-safe.stage2.json",
    "training/DeepThinkingFlow-lora/config.external-local-safe.stage3.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run staged local-safe DeepThinkingFlow LoRA training."
    )
    parser.add_argument(
        "--stage-config",
        action="append",
        dest="stage_configs",
        help="Stage config path. Repeat to override the default stage sequence.",
    )
    parser.add_argument(
        "--external",
        action="store_true",
        help="Use the external-dataset staged config sequence.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate each stage config without launching real training.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_stage_inputs_exist(config: dict, stage_path: Path) -> None:
    missing: list[str] = []
    dataset_path = Path(str(config.get("dataset_path", ""))).resolve()
    eval_dataset_path = Path(str(config.get("eval_dataset_path", ""))).resolve() if config.get("eval_dataset_path") else None
    if not dataset_path.is_file():
        missing.append(str(dataset_path))
    if eval_dataset_path is not None and not eval_dataset_path.is_file():
        missing.append(str(eval_dataset_path))
    if missing:
        raise SystemExit(
            "Stage inputs are missing for "
            f"{stage_path}. Build the dataset assets first. Missing: {', '.join(missing)}"
        )


def latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        suffix = path.name.split("checkpoint-", 1)[-1]
        if suffix.isdigit():
            checkpoints.append((int(suffix), path))
    if not checkpoints:
        return None
    checkpoints.sort()
    return checkpoints[-1][1]


def resolve_resume_value(config: dict) -> str:
    resume_value = str(config.get("resume_from_checkpoint", "") or "").strip()
    if not resume_value:
        return ""
    if resume_value != "latest":
        return resume_value
    output_dir = Path(config["output_dir"]).resolve()
    checkpoint = latest_checkpoint(output_dir)
    return str(checkpoint) if checkpoint is not None else ""


def build_effective_config(config: dict) -> dict:
    effective = dict(config)
    original_resume = str(effective.get("resume_from_checkpoint", "") or "").strip()
    resolved_resume = resolve_resume_value(effective)
    if resolved_resume and resolved_resume != original_resume:
        effective["resume_from_checkpoint"] = resolved_resume
    return effective


def write_effective_config(stage_path: Path, config: dict) -> Path:
    # Keep checked-in configs immutable and pass the resolved checkpoint via a temp config.
    temp_dir = Path(tempfile.mkdtemp(prefix="dtf-staged-train-"))
    temp_path = temp_dir / stage_path.name
    temp_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return temp_path


def run_stage(stage_index: int, stage_path: Path, dry_run: bool) -> int:
    config = load_config(stage_path)
    ensure_stage_inputs_exist(config, stage_path)
    effective_config = build_effective_config(config)
    effective_config_path = write_effective_config(stage_path, effective_config)

    command = [sys.executable, str(TRAIN_SCRIPT), "--config", str(effective_config_path)]
    if dry_run:
        command.append("--dry-run")
    completed = subprocess.run(command, cwd=str(ROOT_DIR), check=False)
    if completed.returncode != 0:
        return completed.returncode

    summary = {
        "stage": stage_index,
        "config": str(stage_path),
        "effective_config": str(effective_config_path),
        "output_dir": effective_config["output_dir"],
        "resume_from_checkpoint": effective_config.get("resume_from_checkpoint", ""),
        "train_samples": effective_config.get("max_train_samples", 0),
        "eval_samples": effective_config.get("max_eval_samples", 0),
        "lora_r": effective_config.get("lora_r"),
        "lora_alpha": effective_config.get("lora_alpha"),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def main() -> int:
    args = parse_args()
    stage_configs = args.stage_configs or (EXTERNAL_STAGE_CONFIGS if args.external else DEFAULT_STAGE_CONFIGS)
    resolved_stage_paths = [Path(path).resolve() for path in stage_configs]
    for index, stage_path in enumerate(resolved_stage_paths, start=1):
        if not stage_path.is_file():
            raise SystemExit(f"Missing stage config: {stage_path}")
        code = run_stage(index, stage_path, args.dry_run)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

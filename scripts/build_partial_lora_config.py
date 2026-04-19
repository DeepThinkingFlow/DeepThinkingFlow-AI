#!/usr/bin/env python3
"""Build a safer partial-scope LoRA config without mutating the base config."""

from __future__ import annotations

import argparse
from pathlib import Path

from deepthinkingflow_exit_codes import OK
from deepthinkingflow_json_io import load_json_file, write_json_file

ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a partial-scope LoRA training config for safer incremental runs."
    )
    parser.add_argument("--base-config", default="training/DeepThinkingFlow-lora/config.local-safe.json")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional adapter output_dir override for the derived partial config.",
    )
    parser.add_argument("--max-train-samples", type=int, default=8)
    parser.add_argument("--max-eval-samples", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--resume-from-checkpoint", default="")
    return parser.parse_args()


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (ROOT_DIR / path).resolve()


def main() -> int:
    args = parse_args()
    base_config_path = resolve_repo_path(args.base_config)
    output_path = resolve_repo_path(args.output)
    config = load_json_file(base_config_path, "base config")

    config["max_train_samples"] = max(1, int(args.max_train_samples))
    config["max_eval_samples"] = max(1, int(args.max_eval_samples))
    config["num_train_epochs"] = max(0.1, float(args.num_train_epochs))
    config["learning_rate"] = float(args.learning_rate)
    config["gradient_accumulation_steps"] = max(1, int(args.gradient_accumulation_steps))
    config["resume_from_checkpoint"] = str(args.resume_from_checkpoint).strip()
    if args.output_dir.strip():
        config["output_dir"] = str(resolve_repo_path(args.output_dir))
    config["derived_from_config"] = str(base_config_path)
    config["partial_training_profile"] = {
        "mode": "safe-partial-lora",
        "output_dir": config.get("output_dir", ""),
        "max_train_samples": config["max_train_samples"],
        "max_eval_samples": config["max_eval_samples"],
        "num_train_epochs": config["num_train_epochs"],
        "learning_rate": config["learning_rate"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "resume_from_checkpoint": config["resume_from_checkpoint"],
    }

    write_json_file(output_path, config)
    print(output_path)
    return OK


if __name__ == "__main__":
    raise SystemExit(main())

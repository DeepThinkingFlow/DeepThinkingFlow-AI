#!/usr/bin/env python3
"""Run a consolidated preflight over runtime, bundle, training, and external-host readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import deepthinkingflow_system_check as system_check
import deepthinkingflow_env as dtf_env
import preflight_deepthinkingflow_training as preflight_train
import validate_behavior_bundle as bundle_validator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a consolidated preflight across the DeepThinkingFlow project."
    )
    parser.add_argument(
        "--bundle",
        default="behavior/DeepThinkingFlow",
        help="Behavior bundle directory to validate.",
    )
    parser.add_argument(
        "--model-dir",
        default="runtime/transformers/DeepThinkingFlow",
        help="Model directory used for inference system checks.",
    )
    parser.add_argument(
        "--training-config",
        default="training/DeepThinkingFlow-lora/config.example.json",
        help="Training config to evaluate.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Reserved for compatibility; output is JSON by default.",
    )
    return parser.parse_args()


def summarize_status(report: dict) -> dict[str, object]:
    dependency_status = dtf_env.detect_dependency_status()
    external_status = dtf_env.detect_external_runtime_status()
    return {
        "dependency_status": dependency_status,
        "external_runtime_status": external_status,
    }


def main() -> int:
    args = parse_args()
    bundle_dir = Path(args.bundle).resolve()
    model_dir = Path(args.model_dir).resolve()
    training_config = Path(args.training_config).resolve()
    training_config_payload = json.loads(training_config.read_text(encoding="utf-8"))

    inference_report = system_check.build_report("inference", model_dir)
    training_report = system_check.build_report("training", model_dir)
    bundle_summary = bundle_validator.validate_bundle(bundle_dir)
    training_env = {
        "memory": preflight_train.memory_snapshot(),
        "gpu": preflight_train.detect_gpu(),
    }
    training_model_info = preflight_train.infer_weight_size(str(training_config_payload["model_name_or_path"]))
    training_feasibility = preflight_train.classify_feasibility(
        training_config_payload,
        training_env,
        training_model_info,
    )

    warnings = []
    warnings.extend(system_check.format_warning_lines(inference_report))
    warnings.extend(system_check.format_warning_lines(training_report))
    if not training_feasibility["can_attempt_local_training"]:
        warnings.extend(f"[warn] - {reason}" for reason in training_feasibility["reasons"])

    result = {
        "schema_version": "dtf-project-preflight/v1",
        "bundle": str(bundle_dir),
        "model_dir": str(model_dir),
        "training_config": str(training_config),
        "status": summarize_status({}),
        "claim_boundary": {
            "raw_base_checkpoint_can_be_described_as_skill_aligned": False,
            "weight_level_adherence_requires_training_artifacts": True,
            "dataset_and_skill_changes_alone_modify_weights": False,
        },
    }
    result["bundle_validation"] = bundle_summary
    result["inference_system_check"] = inference_report
    result["training_system_check"] = training_report
    result["training_feasibility"] = {
        "environment": training_env,
        "model_info": training_model_info,
        "feasibility": training_feasibility,
    }
    result["warnings"] = warnings
    result["ready"] = {
        "bundle_valid": True,
        "inference_soft_gate_clear": not inference_report["warnings"],
        "training_soft_gate_clear": not training_report["warnings"],
        "training_locally_feasible": training_feasibility["can_attempt_local_training"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

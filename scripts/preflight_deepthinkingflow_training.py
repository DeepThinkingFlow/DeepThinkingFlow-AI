#!/usr/bin/env python3
"""Estimate whether a DeepThinkingFlow training config is feasible on the current machine."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from deepthinkingflow_exit_codes import OK, PRECONDITION_FAILED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight DeepThinkingFlow training against current machine resources."
    )
    parser.add_argument(
        "--config",
        default="training/DeepThinkingFlow-lora/config.example.json",
        help="Training config to evaluate.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def memory_snapshot() -> dict[str, int]:
    meminfo: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw_value = line.split(":", 1)
        value_kib = int(raw_value.strip().split()[0])
        meminfo[key] = value_kib * 1024
    return {
        "mem_total_bytes": meminfo.get("MemTotal", 0),
        "mem_available_bytes": meminfo.get("MemAvailable", 0),
        "swap_total_bytes": meminfo.get("SwapTotal", 0),
        "swap_free_bytes": meminfo.get("SwapFree", 0),
    }


def detect_gpu() -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    has_nvidia = bool(nvidia_smi)
    return {
        "has_nvidia_smi": has_nvidia,
        "nvidia_smi_path": nvidia_smi or "",
    }


def infer_weight_size(model_ref: str) -> dict[str, Any]:
    path = Path(model_ref)
    if path.exists():
        model_file = path / "model.safetensors"
        if model_file.is_file():
            size = model_file.stat().st_size
            return {"resolved_model_dir": str(path.resolve()), "weight_file_bytes": size}
    return {"resolved_model_dir": model_ref, "weight_file_bytes": 0}


def classify_feasibility(config: dict[str, Any], env: dict[str, Any], model_info: dict[str, Any]) -> dict[str, Any]:
    available = env["memory"]["mem_available_bytes"]
    swap_free = env["memory"]["swap_free_bytes"]
    weight_size = model_info["weight_file_bytes"]
    total_host_memory = available + swap_free

    reasons: list[str] = []
    recommendations: list[str] = []
    local_safe = True

    if not env["gpu"]["has_nvidia_smi"]:
        reasons.append("No CUDA GPU detected on this machine.")
    if weight_size and total_host_memory and weight_size > total_host_memory:
        local_safe = False
        reasons.append("Base weight file is larger than currently available RAM+swap budget.")
    if weight_size and env["memory"]["mem_total_bytes"] and weight_size > env["memory"]["mem_total_bytes"]:
        local_safe = False
        reasons.append("Base weight file alone exceeds physical RAM.")
    if config.get("use_qlora") and not env["gpu"]["has_nvidia_smi"]:
        local_safe = False
        reasons.append("QLoRA is not practical here because there is no CUDA GPU.")
    if Path(config["model_name_or_path"]).name == "DeepThinkingFlow":
        local_safe = False
        reasons.append("DeepThinkingFlow 20B full base training is not realistic on this host.")
        recommendations.append("Use the tiny smoke config for local verification.")
        recommendations.append("Move 20B LoRA/QLoRA to a stronger GPU machine.")
    else:
        recommendations.append("This looks suitable for local smoke training.")

    return {
        "can_attempt_local_training": local_safe,
        "reasons": reasons,
        "recommendations": recommendations,
    }


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    env = {
        "memory": memory_snapshot(),
        "gpu": detect_gpu(),
    }
    model_info = infer_weight_size(config["model_name_or_path"])
    feasibility = classify_feasibility(config, env, model_info)
    summary = {
        "config": str(config_path),
        "model_name_or_path": config["model_name_or_path"],
        "environment": env,
        "model_info": model_info,
        "feasibility": feasibility,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not feasibility["can_attempt_local_training"]:
        return PRECONDITION_FAILED
    return OK


if __name__ == "__main__":
    raise SystemExit(main())

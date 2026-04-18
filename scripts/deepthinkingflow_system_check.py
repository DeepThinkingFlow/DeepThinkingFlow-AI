#!/usr/bin/env python3
"""System requirement checks for DeepThinkingFlow runtime and training."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

GIB = 1024 ** 3

PROFILES = {
    "inference": {
        "ram_gib_min": 16.0,
        "cpu_logical_min": 4,
        "gpu_required": False,
        "gpu_vram_gib_min": 16.0,
    },
    "training": {
        "ram_gib_min": 24.0,
        "cpu_logical_min": 8,
        "gpu_required": False,
        "gpu_vram_gib_min": 24.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether the current machine is likely to handle DeepThinkingFlow runtime or training."
    )
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default="inference",
        help="Requirement profile to evaluate.",
    )
    parser.add_argument(
        "--model-dir",
        default="runtime/transformers/DeepThinkingFlow",
        help="Optional model directory used to estimate local weight size.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the raw JSON report.",
    )
    return parser.parse_args()


def read_meminfo() -> dict[str, int]:
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.is_file():
        return {}
    result: dict[str, int] = {}
    for line in meminfo_path.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition(":")
        parts = value.strip().split()
        if not parts:
            continue
        try:
            result[key] = int(parts[0]) * 1024
        except ValueError:
            continue
    return result


def detect_ram_bytes() -> tuple[int | None, int | None]:
    meminfo = read_meminfo()
    total = meminfo.get("MemTotal")
    available = meminfo.get("MemAvailable")
    return total, available


def detect_cpu_info() -> dict[str, Any]:
    return {
        "logical_cores": os.cpu_count() or 0,
        "architecture": platform.machine() or "",
        "processor": platform.processor() or "",
    }


def detect_nvidia_gpus() -> list[dict[str, Any]]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []
    command = [
        nvidia_smi,
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return []
    gpus: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) != 2:
            continue
        try:
            vram_mib = int(parts[1])
        except ValueError:
            continue
        gpus.append(
            {
                "name": parts[0],
                "memory_total_mib": vram_mib,
                "memory_total_gib": round(vram_mib / 1024, 2),
            }
        )
    return gpus


def estimate_local_weight_size_gib(model_dir: Path) -> float | None:
    if not model_dir.exists():
        return None
    candidates = list(model_dir.glob("model*.safetensors")) + list(model_dir.glob("pytorch_model*.bin"))
    if not candidates:
        return None
    total = sum(path.stat().st_size for path in candidates if path.is_file())
    if total <= 0:
        return None
    return round(total / GIB, 2)


def build_report(profile: str, model_dir: Path) -> dict[str, Any]:
    requirements = PROFILES[profile]
    ram_total, ram_available = detect_ram_bytes()
    cpu = detect_cpu_info()
    gpus = detect_nvidia_gpus()
    largest_gpu_gib = max((gpu["memory_total_gib"] for gpu in gpus), default=0.0)
    local_weight_gib = estimate_local_weight_size_gib(model_dir)

    warnings: list[str] = []
    if ram_total is not None and (ram_total / GIB) < requirements["ram_gib_min"]:
        warnings.append(
            f"System RAM is below the suggested minimum for {profile}: "
            f"{ram_total / GIB:.2f} GiB < {requirements['ram_gib_min']:.2f} GiB."
        )
    if cpu["logical_cores"] < requirements["cpu_logical_min"]:
        warnings.append(
            f"Logical CPU cores are below the suggested minimum for {profile}: "
            f"{cpu['logical_cores']} < {requirements['cpu_logical_min']}."
        )
    if gpus:
        if largest_gpu_gib < requirements["gpu_vram_gib_min"]:
            warnings.append(
                f"GPU VRAM is below the suggested minimum for {profile}: "
                f"{largest_gpu_gib:.2f} GiB < {requirements['gpu_vram_gib_min']:.2f} GiB."
            )
    else:
        warnings.append(
            "No supported NVIDIA GPU was detected. DeepThinkingFlow can still continue, "
            "but performance may be limited and large-model inference or training may fall back to CPU."
        )

    if local_weight_gib is not None and ram_total is not None and local_weight_gib > (ram_total / GIB):
        warnings.append(
            f"Estimated local weight size ({local_weight_gib:.2f} GiB) exceeds physical RAM "
            f"({ram_total / GIB:.2f} GiB). Loading may stall or swap heavily."
        )

    return {
        "profile": profile,
        "requirements": requirements,
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu": cpu,
            "ram_total_gib": round((ram_total or 0) / GIB, 2) if ram_total is not None else None,
            "ram_available_gib": round((ram_available or 0) / GIB, 2) if ram_available is not None else None,
            "gpus": gpus,
            "largest_gpu_vram_gib": round(largest_gpu_gib, 2) if gpus else None,
            "model_dir": str(model_dir),
            "local_weight_gib": local_weight_gib,
        },
        "warnings": warnings,
        "can_continue": True,
    }


def format_warning_lines(report: dict[str, Any]) -> list[str]:
    warnings = report.get("warnings", [])
    if not warnings:
        return []
    lines = [
        f"[warn] DeepThinkingFlow minimum requirement check for {report['profile']} found potential bottlenecks:",
    ]
    for warning in warnings:
        lines.append(f"[warn] - {warning}")
    lines.append("[warn] Continuing anyway because this is a soft gate, not a hard stop.")
    return lines


def main() -> int:
    args = parse_args()
    report = build_report(args.profile, Path(args.model_dir).resolve())
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    lines = format_warning_lines(report)
    if lines:
        print("\n".join(lines))
    else:
        print(f"[ok] DeepThinkingFlow minimum requirement check passed for {args.profile}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

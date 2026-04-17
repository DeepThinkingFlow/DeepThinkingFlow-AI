#!/usr/bin/env python3
"""Environment helpers for DeepThinkingFlow runtime and training scripts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def candidate_site_packages() -> list[Path]:
    venv_dir = ROOT_DIR / ".venv-tools"
    candidates: list[Path] = []
    for site_path in venv_dir.glob("lib/python*/site-packages"):
        candidates.append(site_path)
    return sorted(candidates)


def inject_local_site_packages() -> list[str]:
    injected: list[str] = []
    for site_path in candidate_site_packages():
        site_str = str(site_path)
        if site_str not in sys.path:
            sys.path.insert(0, site_str)
            injected.append(site_str)
    return injected


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def detect_dependency_status() -> dict[str, bool]:
    inject_local_site_packages()
    modules = ["transformers", "datasets", "peft", "torch", "accelerate", "safetensors"]
    return {name: module_available(name) for name in modules}

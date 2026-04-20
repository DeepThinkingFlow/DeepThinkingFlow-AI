#!/usr/bin/env python3
"""Report a unified readiness view for optional CUDA and Apple acceleration paths."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from deepthinkingflow_apple.backend import apple_backend_status
from deepthinkingflow_cuda.backend import cuda_backend_status


def classify_backend(status: dict) -> dict:
    capability_matrix = status.get("capability_matrix", {})
    live_capabilities = [name for name, enabled in capability_matrix.items() if enabled]
    false_capabilities = [name for name, enabled in capability_matrix.items() if not enabled]
    return {
        "backend_name": status.get("backend_name", ""),
        "maturity": status.get("maturity", "unknown"),
        "safe_without_conflict": bool(status.get("conflicts_with_default_transformers_runtime") is False),
        "effective_today": bool(status.get("bottom_line", {}).get("effective_for_real_inference_acceleration_today", False)),
        "ready_to_load_native_extension": bool(status.get("build_blockers", {}).get("ready_to_load_native_extension", False)),
        "missing_requirements": list(status.get("build_blockers", {}).get("missing_requirements", [])),
        "live_capabilities": live_capabilities,
        "missing_capabilities": false_capabilities,
    }


def main() -> int:
    cuda = cuda_backend_status()
    apple = apple_backend_status()
    payload = {
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "cuda": classify_backend(cuda),
        "apple": classify_backend(apple),
        "recommendation": {
            "keep_optional_backends_enabled_in_repo": True,
            "safe_for_current_default_runtime": True,
            "best_next_step": (
                "Choose one acceleration path and implement real runtime integration there. "
                "Do not split effort evenly across both paths while both remain scaffold-only."
            ),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

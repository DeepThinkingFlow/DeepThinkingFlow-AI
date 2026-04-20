#!/usr/bin/env python3
"""Produce a stricter doctor report for optional native acceleration backends."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from deepthinkingflow_apple.backend import apple_backend_status
from deepthinkingflow_cuda.backend import cuda_backend_status


def summarize_backend(status: dict[str, object]) -> dict[str, object]:
    capability_matrix = dict(status.get("capability_matrix", {}))
    missing_capabilities = [name for name, enabled in capability_matrix.items() if not enabled]
    live_capabilities = [name for name, enabled in capability_matrix.items() if enabled]
    build_blockers = dict(status.get("build_blockers", {}))
    missing_requirements = list(build_blockers.get("missing_requirements", []))
    next_required_steps = list(status.get("next_required_steps", []))
    return {
        "backend_name": status.get("backend_name", ""),
        "maturity": status.get("maturity", "unknown"),
        "effective_today": bool(status.get("bottom_line", {}).get("effective_for_real_inference_acceleration_today", False)),
        "extension_built": bool(status.get("extension_built", False)),
        "extension_importable": bool(status.get("extension_importable", False)),
        "ready_to_attempt_native_build": bool(build_blockers.get("ready_to_attempt_native_build", False)),
        "ready_to_load_native_extension": bool(build_blockers.get("ready_to_load_native_extension", False)),
        "missing_requirements": missing_requirements,
        "live_capabilities": live_capabilities,
        "missing_capabilities": missing_capabilities,
        "highest_verified_claim": (
            "native-runtime-verified"
            if bool(build_blockers.get("ready_to_load_native_extension", False))
            and len(missing_capabilities) == 0
            else "scaffold-only"
        ),
        "next_required_steps": next_required_steps,
    }


def main() -> int:
    cuda = summarize_backend(cuda_backend_status())
    apple = summarize_backend(apple_backend_status())
    payload = {
        "schema_version": "dtf-accelerator-doctor/v1",
        "project": "DeepThinkingFlow",
        "cuda": cuda,
        "apple": apple,
        "global_verdict": {
            "native_acceleration_ready_today": False,
            "reason": (
                "Neither optional backend currently has a verified native extension load plus "
                "a real end-to-end generation path."
            ),
            "release_claim_ceiling": "scaffold-only",
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

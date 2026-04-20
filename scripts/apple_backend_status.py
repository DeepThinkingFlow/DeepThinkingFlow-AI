#!/usr/bin/env python3
"""Report DeepThinkingFlow Apple Silicon backend scaffold/build readiness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from deepthinkingflow_apple.backend import apple_backend_status, recommended_cmake_configure_command


def main() -> int:
    payload = apple_backend_status()
    payload["recommended_configure_command"] = recommended_cmake_configure_command()
    payload["build_files"] = {
        "cmakelists": str((Path("apple_backend") / "CMakeLists.txt").resolve()),
        "bindings": str((Path("apple_backend") / "src" / "python_bindings.mm").resolve()),
        "metal_runtime": str((Path("apple_backend") / "src" / "metal_runtime.mm").resolve()),
    }
    payload["performance_contract"] = {
        "prefer_unified_memory_views": True,
        "prefer_mps_graph": True,
        "prefer_binary_archive_cache": True,
        "prefer_coreml_for_ane_only": True,
        "avoid_pytorch_mps_for_production": True,
    }
    payload["build_modes"] = {
        "cpu_fallback_configure_command": [
            "cmake",
            "-S",
            str((Path("apple_backend")).resolve()),
            "-B",
            str((Path("apple_backend") / "build-cpu").resolve()),
            "-DDTF_APPLE_SILICON_ENABLED=OFF",
        ],
        "apple_silicon_configure_command": payload["recommended_configure_command"],
    }
    payload["failure_modes_to_avoid"] = [
        "using_pytorch_mps_for_production_inference",
        "treating_unified_memory_like_discrete_gpu_memory",
        "skipping_binary_archive_cache_for_msl_kernels",
        "using_coreml_path_without_ane_goal",
        "failing_to_quantize_20b_model_for_bandwidth_and_fit",
    ]
    payload["bottom_line"] = {
        "safe_to_keep_in_repo_without_runtime_conflict": True,
        "effective_for_real_inference_acceleration_today": False,
        "why": "This path is an explicit optional scaffold. It does not intercept the default runtime unless the native extension is built and called deliberately.",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

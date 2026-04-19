#!/usr/bin/env python3
"""Report DeepThinkingFlow CUDA backend scaffold/build readiness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from deepthinkingflow_cuda.backend import cuda_backend_status, recommended_cmake_configure_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report the DeepThinkingFlow CUDA backend build/runtime readiness."
    )
    parser.add_argument(
        "--cuda-arch",
        default="89",
        help="Exact CUDA SM architecture to suggest in the configure command, for example 89 or 90.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = cuda_backend_status()
    payload["recommended_configure_command"] = recommended_cmake_configure_command(args.cuda_arch)
    payload["build_files"] = {
        "cmakelists": str((Path("cuda_backend") / "CMakeLists.txt").resolve()),
        "bindings": str((Path("cuda_backend") / "src" / "python_bindings.cpp").resolve()),
        "cuda_runtime": str((Path("cuda_backend") / "src" / "cuda_runtime.cu").resolve()),
    }
    payload["performance_contract"] = {
        "require_exact_arch": True,
        "prefer_cuda_malloc_async": True,
        "prefer_stream_overlap": True,
        "prefer_fused_kernels": True,
        "profile_with_nsight_compute_first": True,
    }
    payload["build_modes"] = {
        "cpu_fallback_configure_command": [
            "cmake",
            "-S",
            str(Path("cuda_backend").resolve()),
            "-B",
            str((Path("cuda_backend") / "build-cpu").resolve()),
            "-DDTF_CUDA_ENABLED=OFF",
        ],
        "cuda_configure_command": payload["recommended_configure_command"] + ["-DDTF_CUDA_ENABLED=ON", "-DDTF_CUDA_STRICT=ON"],
    }
    payload["failure_modes_to_avoid"] = [
        "generic_or_multi_arch_cuda_build",
        "building_cuda_backend_without_nvcc",
        "treating_quantization_as_optional_when_vram_is_below_target",
        "profiling_only_at_the_end",
        "running_without_async_allocator_or_stream_plan",
    ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

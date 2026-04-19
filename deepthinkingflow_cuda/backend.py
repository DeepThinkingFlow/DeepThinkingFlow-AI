"""Runtime helpers around the optional DeepThinkingFlow CUDA extension."""

from __future__ import annotations

import importlib
import re
import shutil
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
CUDA_BACKEND_DIR = ROOT_DIR / "cuda_backend"


class CUDABackendUnavailable(RuntimeError):
    """Raised when the optional CUDA backend extension is unavailable."""


def _load_extension():
    try:
        return importlib.import_module("_dtf_cuda_backend")
    except ModuleNotFoundError as exc:
        raise CUDABackendUnavailable(
            "The DeepThinkingFlow CUDA extension is not built yet. "
            "Configure and build cuda_backend/ with CMake + pybind11 first."
        ) from exc


def recommended_cmake_configure_command(cuda_arch: str) -> list[str]:
    normalized_arch = str(cuda_arch).strip()
    if not re.fullmatch(r"\d{2,3}", normalized_arch):
        raise ValueError("cuda_arch must be an exact SM number such as 89 or 90")
    build_dir = CUDA_BACKEND_DIR / "build"
    return [
        "cmake",
        "-S",
        str(CUDA_BACKEND_DIR),
        "-B",
        str(build_dir),
        f"-DCMAKE_CUDA_ARCHITECTURES={normalized_arch}",
    ]


def cuda_backend_status() -> dict[str, Any]:
    nvcc_path = shutil.which("nvcc")
    cmake_path = shutil.which("cmake")
    extension_built = False
    for candidate in ROOT_DIR.rglob("_dtf_cuda_backend*.so"):
        if candidate.is_file():
            extension_built = True
            break
    return {
        "cuda_backend_dir": str(CUDA_BACKEND_DIR),
        "cmake_available": cmake_path is not None,
        "nvcc_available": nvcc_path is not None,
        "cmake_path": cmake_path or "",
        "nvcc_path": nvcc_path or "",
        "extension_built": extension_built,
        "cpu_fallback_supported": True,
        "strict_cuda_mode_supported": True,
        "required_stack": {
            "languages": ["C++20", "Python", "CUDA C++"],
            "libraries": [
                "CUTLASS 3.x",
                "cuBLASLt",
                "Flash Attention 2",
                "bitsandbytes",
                "NCCL",
                "pybind11",
            ],
        },
    }


def create_backend():
    module = _load_extension()
    return module.CudaBackend()

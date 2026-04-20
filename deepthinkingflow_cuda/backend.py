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
        status = cuda_backend_status()
        raise CUDABackendUnavailable(
            "The DeepThinkingFlow CUDA extension is not built yet. "
            "Configure and build cuda_backend/ with CMake + pybind11 first. "
            f"Missing requirements: {', '.join(status['build_blockers']['missing_requirements']) or 'unknown'}."
        ) from exc


def _extension_importable() -> bool:
    try:
        importlib.import_module("_dtf_cuda_backend")
        return True
    except ModuleNotFoundError:
        return False


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
    cxx_path = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    extension_built = False
    for candidate in ROOT_DIR.rglob("_dtf_cuda_backend*.so"):
        if candidate.is_file():
            extension_built = True
            break
    extension_importable = _extension_importable()
    missing_build_requirements = []
    if cmake_path is None:
        missing_build_requirements.append("cmake")
    if cxx_path is None:
        missing_build_requirements.append("cxx-compiler")
    if nvcc_path is None:
        missing_build_requirements.append("nvcc")
    try:
        import pybind11  # type: ignore
    except ModuleNotFoundError:
        missing_build_requirements.append("pybind11")
    next_required_steps = [
        "install-cmake",
        "install-cxx-compiler",
        "install-nvcc",
        "install-pybind11",
        "configure-cmake-for-exact-sm",
        "build-native-extension",
        "verify-extension-import",
        "wire-cutlass-gemm-runtime",
        "wire-flash-attention-runtime",
        "wire-kv-cache-decode-path",
        "wire-end-to-end-generation-path",
        "collect-benchmark-evidence",
    ]
    return {
        "backend_name": "DeepThinkingFlow CUDA backend",
        "maturity": "scaffold-only",
        "accelerates_current_python_runtime_by_itself": False,
        "conflicts_with_default_transformers_runtime": False,
        "activation_mode": "manual-opt-in",
        "cuda_backend_dir": str(CUDA_BACKEND_DIR),
        "cmake_available": cmake_path is not None,
        "nvcc_available": nvcc_path is not None,
        "cxx_available": cxx_path is not None,
        "cmake_path": cmake_path or "",
        "nvcc_path": nvcc_path or "",
        "cxx_path": cxx_path or "",
        "extension_built": extension_built,
        "extension_importable": extension_importable,
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
        "integration_boundary": {
            "never_auto_overrides_default_runtime": True,
            "requires_explicit_build_step": True,
            "requires_explicit_call_site_integration": True,
            "current_role": "capability scaffold and verification surface",
        },
        "readiness_summary": {
            "build_scaffold_ready": True,
            "host_detection_ready": True,
            "kernel_optimization_ready": False,
            "end_to_end_inference_ready": False,
        },
        "build_blockers": {
            "missing_requirements": missing_build_requirements,
            "ready_to_attempt_native_build": len(missing_build_requirements) == 0,
            "ready_to_load_native_extension": extension_built and extension_importable,
        },
        "next_required_steps": next_required_steps,
        "capability_matrix": {
            "host_detection": True,
            "native_extension_loading": extension_built and extension_importable,
            "cutlass_gemm_integration": False,
            "flash_attention_runtime_path": False,
            "quantized_matmul_runtime_path": False,
            "kv_cache_decode_path": False,
            "end_to_end_generation_path": False,
            "benchmarked_latency_evidence": False,
        },
    }


def create_backend():
    if not CUDA_BACKEND_DIR.is_dir():
        raise CUDABackendUnavailable("cuda_backend/ is missing from the project tree.")
    status = cuda_backend_status()
    if not status["build_blockers"]["ready_to_load_native_extension"]:
        raise CUDABackendUnavailable(
            "CUDA backend native extension is not ready to load. "
            f"Missing requirements: {', '.join(status['build_blockers']['missing_requirements']) or 'none'}, "
            f"extension_built={status['extension_built']}, "
            f"extension_importable={status['extension_importable']}."
        )
    module = _load_extension()
    return module.CudaBackend()

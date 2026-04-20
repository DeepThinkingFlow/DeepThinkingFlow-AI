"""Runtime helpers around the optional DeepThinkingFlow Apple Silicon extension."""

from __future__ import annotations

import importlib
import platform
import shutil
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
APPLE_BACKEND_DIR = ROOT_DIR / "apple_backend"


class AppleBackendUnavailable(RuntimeError):
    """Raised when the optional Apple Silicon backend extension is unavailable."""


def _load_extension():
    try:
        return importlib.import_module("_dtf_apple_backend")
    except ModuleNotFoundError as exc:
        status = apple_backend_status()
        raise AppleBackendUnavailable(
            "The DeepThinkingFlow Apple Silicon extension is not built yet. "
            "Configure and build apple_backend/ with CMake + pybind11 first. "
            f"Missing requirements: {', '.join(status['build_blockers']['missing_requirements']) or 'unknown'}."
        ) from exc


def _extension_importable() -> bool:
    try:
        importlib.import_module("_dtf_apple_backend")
        return True
    except ModuleNotFoundError:
        return False


def recommended_cmake_configure_command() -> list[str]:
    build_dir = APPLE_BACKEND_DIR / "build"
    return [
        "cmake",
        "-S",
        str(APPLE_BACKEND_DIR),
        "-B",
        str(build_dir),
        "-DDTF_APPLE_SILICON_ENABLED=ON",
        "-DDTF_APPLE_STRICT=ON",
    ]


def apple_backend_status() -> dict[str, Any]:
    cmake_path = shutil.which("cmake")
    cxx_path = shutil.which("c++") or shutil.which("clang++")
    extension_built = False
    for candidate in ROOT_DIR.rglob("_dtf_apple_backend*.so"):
        if candidate.is_file():
            extension_built = True
            break
    extension_importable = _extension_importable()
    machine = platform.machine().lower()
    system_name = platform.system()
    is_macos = system_name == "Darwin"
    is_apple_silicon = is_macos and machine in {"arm64", "aarch64"}
    missing_build_requirements = []
    if cmake_path is None:
        missing_build_requirements.append("cmake")
    if cxx_path is None:
        missing_build_requirements.append("objcxx-compiler")
    try:
        import pybind11  # type: ignore
    except ModuleNotFoundError:
        missing_build_requirements.append("pybind11")
    if not is_macos:
        missing_build_requirements.append("macos-host")
    elif not is_apple_silicon:
        missing_build_requirements.append("apple-silicon-host")
    next_required_steps = [
        "install-cmake",
        "install-objcxx-compiler",
        "install-pybind11",
        "build-on-macos-apple-silicon-host",
        "configure-cmake-for-apple-backend",
        "build-native-extension",
        "verify-extension-import",
        "wire-mlx-inference-runtime",
        "wire-mpsgraph-fusion-path",
        "wire-kv-cache-decode-path",
        "wire-end-to-end-generation-path",
        "collect-benchmark-evidence",
    ]
    return {
        "backend_name": "DeepThinkingFlow Apple Silicon backend",
        "maturity": "scaffold-only",
        "accelerates_current_python_runtime_by_itself": False,
        "conflicts_with_default_transformers_runtime": False,
        "activation_mode": "manual-opt-in",
        "apple_backend_dir": str(APPLE_BACKEND_DIR),
        "cmake_available": cmake_path is not None,
        "cxx_available": cxx_path is not None,
        "cmake_path": cmake_path or "",
        "cxx_path": cxx_path or "",
        "extension_built": extension_built,
        "extension_importable": extension_importable,
        "system": system_name,
        "machine": machine,
        "is_macos": is_macos,
        "is_apple_silicon": is_apple_silicon,
        "cpu_fallback_supported": True,
        "required_stack": {
            "languages": ["Swift / Objective-C++", "C++20", "Python", "Metal Shading Language"],
            "frameworks": ["MLX", "Metal Performance Shaders", "Core ML", "Accelerate / BNNS"],
            "engines": ["llama.cpp", "MLX-LM", "Ollama"],
            "quantization": ["GGUF Q4_K_M", "MLX quantization"],
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
            "metal_runtime_path": False,
            "mlx_inference_runtime_path": False,
            "mps_graph_fusion_path": False,
            "coreml_ane_path": False,
            "kv_cache_decode_path": False,
            "end_to_end_generation_path": False,
            "benchmarked_latency_evidence": False,
        },
    }


def create_backend():
    if not APPLE_BACKEND_DIR.is_dir():
        raise AppleBackendUnavailable("apple_backend/ is missing from the project tree.")
    status = apple_backend_status()
    if not status["build_blockers"]["ready_to_load_native_extension"]:
        raise AppleBackendUnavailable(
            "Apple backend native extension is not ready to load. "
            f"Missing requirements: {', '.join(status['build_blockers']['missing_requirements']) or 'none'}, "
            f"extension_built={status['extension_built']}, "
            f"extension_importable={status['extension_importable']}."
        )
    module = _load_extension()
    return module.AppleBackend()

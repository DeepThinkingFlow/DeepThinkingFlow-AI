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
        raise AppleBackendUnavailable(
            "The DeepThinkingFlow Apple Silicon extension is not built yet. "
            "Configure and build apple_backend/ with CMake + pybind11 first."
        ) from exc


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
    extension_built = False
    for candidate in ROOT_DIR.rglob("_dtf_apple_backend*.so"):
        if candidate.is_file():
            extension_built = True
            break
    machine = platform.machine().lower()
    system_name = platform.system()
    is_macos = system_name == "Darwin"
    is_apple_silicon = is_macos and machine in {"arm64", "aarch64"}
    return {
        "apple_backend_dir": str(APPLE_BACKEND_DIR),
        "cmake_available": cmake_path is not None,
        "cmake_path": cmake_path or "",
        "extension_built": extension_built,
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
    }


def create_backend():
    module = _load_extension()
    return module.AppleBackend()

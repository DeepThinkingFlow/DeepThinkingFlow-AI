#!/usr/bin/env python3
"""Report MLX-first adapter readiness for DeepThinkingFlow on Apple Silicon."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from deepthinkingflow_apple.backend import apple_backend_status
from deepthinkingflow_apple.mlx_adapter import MLXUnavailable, mlx_runtime_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report DeepThinkingFlow MLX-first inference adapter readiness."
    )
    parser.add_argument(
        "--model-dir",
        default="runtime/transformers/DeepThinkingFlow",
        help="Model directory that the MLX adapter will target.",
    )
    parser.add_argument(
        "--quantize-4bit",
        action="store_true",
        help="Report the adapter in 4-bit quantized mode.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = {
        "apple_backend": apple_backend_status(),
        "model_dir": str(Path(args.model_dir).resolve()),
    }
    try:
        payload["mlx_adapter"] = mlx_runtime_status(args.model_dir, quantize_4bit=args.quantize_4bit)
        payload["mlx_available"] = True
    except MLXUnavailable as exc:
        payload["mlx_available"] = False
        payload["mlx_error"] = str(exc)
        payload["mlx_adapter"] = {
            "runtime_contract": {
                "unified_memory_first": True,
                "copy_avoidance_required": True,
                "mlx_native_forward_path": True,
                "mpsgraph_fusion_next": True,
                "coreml_ane_path_optional": True,
            }
        }
    payload["next_steps"] = [
        "Implement MLX-native weight loader from safetensors/config into unified-memory tensors.",
        "Add MPSGraph fused path after MLX forward correctness is stable.",
        "Add 4-bit MLX quantization lane before considering Core ML / ANE conversion.",
    ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

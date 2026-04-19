#!/usr/bin/env python3
"""Run a provisional MoE forward check on quantized expert weights and report output ranges."""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


MXFP4_LUT = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a provisional dequant-on-the-fly MoE forward and report output ranges."
    )
    parser.add_argument("--weights", default="original/model.safetensors", help="Path to safetensors weights.")
    parser.add_argument(
        "--config",
        default="runtime/transformers/DeepThinkingFlow/config.json",
        help="Path to model config.json.",
    )
    parser.add_argument("--layer-index", type=int, default=0, help="MoE block index.")
    parser.add_argument("--seq-len", type=int, default=8, help="Synthetic sequence length.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for synthetic input.")
    parser.add_argument(
        "--input-scale",
        type=float,
        default=None,
        help="Optional multiplier applied to random normal input. Use 0.1 to simulate embedding-scale activations.",
    )
    parser.add_argument("--group-size", type=int, default=32, help="Logical values represented by each scale group.")
    parser.add_argument(
        "--activation",
        choices=("silu", "swiglu", "both"),
        default="both",
        help="Activation to evaluate for the expert FFN.",
    )
    return parser.parse_args()


def load_header_and_offsets(weights_path: Path) -> tuple[dict[str, object], int]:
    with weights_path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    return header, 8 + header_len


def dtype_for_safetensors(dtype_name: str) -> np.dtype:
    mapping = {
        "U8": np.uint8,
        "BF16": np.uint16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype for local MoE check: {dtype_name}")
    return mapping[dtype_name]


def bf16_to_fp32(values: np.ndarray) -> np.ndarray:
    widened = values.astype(np.uint32) << 16
    return widened.view(np.float32)


def load_tensor_np(weights_path: Path, name: str) -> np.ndarray:
    header, data_start = load_header_and_offsets(weights_path)
    spec = header[name]
    begin, end = spec["data_offsets"]
    dtype_name = spec["dtype"]
    dtype = dtype_for_safetensors(dtype_name)
    shape = tuple(spec["shape"])
    with weights_path.open("rb") as handle:
        handle.seek(data_start + begin)
        raw = handle.read(end - begin)
    array = np.frombuffer(raw, dtype=dtype).reshape(shape)
    if dtype_name == "BF16":
        return bf16_to_fp32(array)
    return array


def decode_mxfp4(blocks: np.ndarray) -> np.ndarray:
    lo = np.bitwise_and(blocks, 0x0F)
    hi = np.bitwise_and(np.right_shift(blocks, 4), 0x0F)
    stacked = np.stack([MXFP4_LUT[lo], MXFP4_LUT[hi]], axis=-1)
    return stacked.reshape(*blocks.shape[:-1], blocks.shape[-1] * 2)


def decode_ue8(scales: np.ndarray) -> np.ndarray:
    return np.power(2.0, scales.astype(np.float32) - 127.0)


def expand_scales(scales: np.ndarray, group_size: int) -> np.ndarray:
    return np.repeat(scales[..., None], group_size, axis=-1)


def dequant_selected(blocks: np.ndarray, scales: np.ndarray, group_size: int) -> np.ndarray:
    decoded = decode_mxfp4(blocks)
    expanded = expand_scales(decode_ue8(scales), group_size)
    return (decoded * expanded).reshape(decoded.shape[0], decoded.shape[1], -1)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def rms_norm(x: np.ndarray, scale: np.ndarray, eps: float) -> np.ndarray:
    squared_mean = np.mean(np.square(x), axis=-1, keepdims=True)
    normalized = x * np.reciprocal(np.sqrt(squared_mean + eps, dtype=np.float32))
    return normalized * scale


def summarize(name: str, tensor: np.ndarray) -> dict[str, float | str]:
    return {
        "name": name,
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "mean": float(tensor.mean()),
        "abs_max": float(np.abs(tensor).max()),
        "std": float(tensor.std()),
    }


def run_moe_forward(
    x: np.ndarray,
    norm_scale: np.ndarray,
    gate_w: np.ndarray,
    gate_b: np.ndarray,
    mlp1_blocks: np.ndarray,
    mlp1_scales: np.ndarray,
    mlp1_bias: np.ndarray,
    mlp2_blocks: np.ndarray,
    mlp2_scales: np.ndarray,
    mlp2_bias: np.ndarray,
    *,
    experts_per_tok: int,
    activation: str,
    group_size: int,
    eps: float,
) -> dict[str, object]:
    x_normed = rms_norm(x, norm_scale, eps)
    logits = x_normed @ gate_w.T + gate_b
    topk_idx = np.argsort(logits, axis=-1)[:, -experts_per_tok:]
    topk_logits = np.take_along_axis(logits, topk_idx, axis=-1)
    topk_logits = topk_logits - np.max(topk_logits, axis=-1, keepdims=True)
    topk_scores = np.exp(topk_logits)
    topk_scores = topk_scores / np.sum(topk_scores, axis=-1, keepdims=True)

    out = np.zeros_like(x_normed, dtype=np.float32)
    selected_dequant_stats: list[dict[str, float | str]] = []
    structural_issue: str | None = None

    for route_slot in range(experts_per_tok):
        expert_ids = topk_idx[:, route_slot]
        scores = topk_scores[:, route_slot:route_slot + 1].astype(np.float32)
        w1 = dequant_selected(mlp1_blocks[expert_ids], mlp1_scales[expert_ids], group_size)
        w2 = dequant_selected(mlp2_blocks[expert_ids], mlp2_scales[expert_ids], group_size)
        pre = np.einsum("bi,boi->bo", x_normed, w1, optimize=True) + mlp1_bias[expert_ids]
        if activation == "swiglu":
            gate, up = np.split(pre, 2, axis=-1)
            hidden = silu(gate) * up
        else:
            hidden = silu(pre)
        if hidden.shape[-1] != w2.shape[-1]:
            structural_issue = (
                f"activation '{activation}' produced hidden width {hidden.shape[-1]} "
                f"but mlp2 expects {w2.shape[-1]}"
            )
            break
        proj = np.einsum("bo,boi->bi", hidden, w2, optimize=True) + mlp2_bias[expert_ids]
        out = out + scores * proj
        if route_slot == 0:
            selected_dequant_stats.append(summarize("w1_selected", w1))
            selected_dequant_stats.append(summarize("w2_selected", w2))
            selected_dequant_stats.append(summarize("pre_activation", pre))
            selected_dequant_stats.append(summarize("hidden_after_activation", hidden))
            selected_dequant_stats.append(summarize("proj_after_mlp2", proj))

    residual = x + out
    return {
        "activation": activation,
        "compatible_with_mlp2": structural_issue is None,
        "structural_issue": structural_issue,
        "x_normed": summarize("x_normed", x_normed),
        "logits": summarize("router_logits", logits),
        "topk_scores": summarize("topk_scores", topk_scores),
        "output": None if structural_issue else summarize("moe_output", out),
        "residual_output": None if structural_issue else summarize("residual_output", residual),
        "selected_stats": selected_dequant_stats,
        "topk_indices_first_token": [int(v) for v in topk_idx[0].tolist()],
    }


def main() -> int:
    args = parse_args()
    weights_path = Path(args.weights).resolve()
    config = json.loads(Path(args.config).resolve().read_text(encoding="utf-8"))
    hidden_size = int(config["hidden_size"])
    experts_per_tok = int(config["num_experts_per_tok"])
    eps = float(config.get("rms_norm_eps", 1e-5))
    layer_prefix = f"block.{args.layer_index}.mlp"

    norm_scale = load_tensor_np(weights_path, f"{layer_prefix}.norm.scale")
    gate_w = load_tensor_np(weights_path, f"{layer_prefix}.gate.weight")
    gate_b = load_tensor_np(weights_path, f"{layer_prefix}.gate.bias")
    mlp1_blocks = load_tensor_np(weights_path, f"{layer_prefix}.mlp1_weight.blocks")
    mlp1_scales = load_tensor_np(weights_path, f"{layer_prefix}.mlp1_weight.scales")
    mlp1_bias = load_tensor_np(weights_path, f"{layer_prefix}.mlp1_bias")
    mlp2_blocks = load_tensor_np(weights_path, f"{layer_prefix}.mlp2_weight.blocks")
    mlp2_scales = load_tensor_np(weights_path, f"{layer_prefix}.mlp2_weight.scales")
    mlp2_bias = load_tensor_np(weights_path, f"{layer_prefix}.mlp2_bias")

    rng = np.random.default_rng(args.seed)
    x = rng.standard_normal((args.seq_len, hidden_size), dtype=np.float32)
    if args.input_scale is None:
        x = x / math.sqrt(hidden_size)
        input_mode = "small-rms"
    else:
        x = x * float(args.input_scale)
        input_mode = f"scaled-normal:{args.input_scale}"

    activations = ["silu", "swiglu"] if args.activation == "both" else [args.activation]
    results = [
        run_moe_forward(
            x,
            norm_scale,
            gate_w,
            gate_b,
            mlp1_blocks,
            mlp1_scales,
            mlp1_bias,
            mlp2_blocks,
            mlp2_scales,
            mlp2_bias,
            experts_per_tok=experts_per_tok,
            activation=activation,
            group_size=args.group_size,
            eps=eps,
        )
        for activation in activations
    ]

    payload = {
        "weights": str(weights_path),
        "config_hidden_act": config.get("hidden_act"),
        "layer_index": args.layer_index,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "input_mode": input_mode,
        "input_scale": args.input_scale,
        "group_size": args.group_size,
        "rms_norm_eps": eps,
        "input": summarize("synthetic_input", x),
        "results": results,
        "claim_boundary": {
            "synthetic_input_only": True,
            "provisional_moe_forward": True,
            "activation_type_not_proven": True,
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

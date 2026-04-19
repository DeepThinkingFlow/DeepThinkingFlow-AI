"""MLX-first inference adapter scaffold for DeepThinkingFlow on Apple Silicon."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class MLXUnavailable(RuntimeError):
    """Raised when MLX is not available in the current Python environment."""


def _mlx_core():
    try:
        return importlib.import_module("mlx.core")
    except ModuleNotFoundError as exc:
        raise MLXUnavailable(
            "mlx is not installed in the current environment. Install requirements-apple-backend.txt on Apple Silicon."
        ) from exc


def _mlx_nn():
    try:
        return importlib.import_module("mlx.nn")
    except ModuleNotFoundError as exc:
        raise MLXUnavailable(
            "mlx.nn is not available in the current environment. Install MLX on Apple Silicon first."
        ) from exc


def load_deepthinkingflow_weight_shapes_with_mlx(weights_path: str | Path) -> dict[str, list[int]]:
    mx = _mlx_core()
    loaded = mx.load(str(Path(weights_path).resolve()))
    return {name: list(value.shape) for name, value in loaded.items()}


def verify_deepthinkingflow_first_block_shapes(weight_shapes: dict[str, list[int]], config: dict[str, Any]) -> dict[str, Any]:
    hidden_size = int(config["hidden_size"])
    num_attention_heads = int(config["num_attention_heads"])
    num_key_value_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])

    q_dim = num_attention_heads * head_dim
    kv_dim = num_key_value_heads * head_dim
    expected = {
        "block.0.attn.qkv.weight": [q_dim + kv_dim + kv_dim, hidden_size],
        "block.0.attn.out.weight": [hidden_size, q_dim],
        "block.0.mlp.gate.weight": [int(config.get("num_local_experts", 0) or 0), hidden_size],
    }

    checks: dict[str, Any] = {
        "expected": expected,
        "actual": {name: weight_shapes.get(name) for name in expected},
        "passed": True,
        "problems": [],
        "gqa": {
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "repeat_kv_factor": num_attention_heads // num_key_value_heads,
        },
    }
    for name, shape in expected.items():
        actual = weight_shapes.get(name)
        if actual != shape:
            checks["passed"] = False
            checks["problems"].append(
                f"{name} shape mismatch: expected {shape}, found {actual}"
            )
    return checks


def deepthinkingflow_attention_dimensions(config: dict[str, Any]) -> dict[str, int]:
    hidden_size = int(config["hidden_size"])
    num_attention_heads = int(config["num_attention_heads"])
    num_key_value_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    q_dim = num_attention_heads * head_dim
    kv_dim = num_key_value_heads * head_dim
    return {
        "hidden_size": hidden_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "q_dim": q_dim,
        "kv_dim": kv_dim,
        "repeat_kv_factor": num_attention_heads // num_key_value_heads,
    }


def deepthinkingflow_layer_type(config: dict[str, Any], layer_index: int) -> str:
    layer_types = config.get("layer_types") or []
    if not layer_types:
        return "full_attention"
    return str(layer_types[layer_index])


def dry_run_deepthinkingflow_attention_shapes(seq_len: int, config: dict[str, Any]) -> dict[str, Any]:
    dims = deepthinkingflow_attention_dimensions(config)
    q_dim = dims["q_dim"]
    kv_dim = dims["kv_dim"]
    num_q = dims["num_attention_heads"]
    num_kv = dims["num_key_value_heads"]
    head_dim = dims["head_dim"]
    n_rep = dims["repeat_kv_factor"]
    return {
        "input": [seq_len, dims["hidden_size"]],
        "qkv_linear": [seq_len, q_dim + kv_dim + kv_dim],
        "q_split": [seq_len, q_dim],
        "k_split": [seq_len, kv_dim],
        "v_split": [seq_len, kv_dim],
        "q_reshaped": [seq_len, num_q, head_dim],
        "k_reshaped": [seq_len, num_kv, head_dim],
        "v_reshaped": [seq_len, num_kv, head_dim],
        "k_repeated": [seq_len, num_kv * n_rep, head_dim],
        "v_repeated": [seq_len, num_kv * n_rep, head_dim],
        "attn_scores": [num_q, seq_len, seq_len],
        "attn_output_heads": [seq_len, num_q, head_dim],
        "attn_output_flat": [seq_len, q_dim],
        "attn_out_projected": [seq_len, dims["hidden_size"]],
        "repeat_kv_factor": n_rep,
    }


def dry_run_deepthinkingflow_block_attention_shapes(
    seq_len: int,
    config: dict[str, Any],
    *,
    layer_index: int = 0,
    cached_seq_len: int = 0,
) -> dict[str, Any]:
    dims = deepthinkingflow_attention_dimensions(config)
    layer_type = deepthinkingflow_layer_type(config, layer_index)
    sliding_window = int(config.get("sliding_window", 128))
    new_total_kv = cached_seq_len + seq_len
    visible_kv = min(new_total_kv, sliding_window) if layer_type == "sliding_attention" else new_total_kv
    num_q = dims["num_attention_heads"]
    head_dim = dims["head_dim"]
    return {
        "layer_index": layer_index,
        "layer_type": layer_type,
        "input": [seq_len, dims["hidden_size"]],
        "q": [seq_len, num_q, head_dim],
        "k_new": [seq_len, dims["num_key_value_heads"], head_dim],
        "v_new": [seq_len, dims["num_key_value_heads"], head_dim],
        "kv_cache_before": {
            "k": [cached_seq_len, dims["num_key_value_heads"], head_dim],
            "v": [cached_seq_len, dims["num_key_value_heads"], head_dim],
        },
        "kv_cache_after_append": {
            "k": [new_total_kv, dims["num_key_value_heads"], head_dim],
            "v": [new_total_kv, dims["num_key_value_heads"], head_dim],
        },
        "kv_cache_after_trim": {
            "k": [visible_kv, dims["num_key_value_heads"], head_dim],
            "v": [visible_kv, dims["num_key_value_heads"], head_dim],
        },
        "k_repeated": [visible_kv, num_q, head_dim],
        "v_repeated": [visible_kv, num_q, head_dim],
        "attn_scores": [num_q, seq_len, visible_kv],
        "attn_output_heads": [seq_len, num_q, head_dim],
        "attn_output_flat": [seq_len, dims["q_dim"]],
        "attn_out_projected": [seq_len, dims["hidden_size"]],
    }


def list_block_mlp_keys_from_shapes(weight_shapes: dict[str, list[int]], layer_index: int = 0) -> dict[str, list[int]]:
    prefix = f"block.{layer_index}.mlp."
    return {name: shape for name, shape in weight_shapes.items() if name.startswith(prefix)}


def unpack_fp4_packed_blocks(blocks: Any, mx: Any) -> Any:
    lo = mx.bitwise_and(blocks, 0xF)
    hi = mx.right_shift(blocks, 4)
    unpacked = mx.stack([lo, hi], axis=-1)
    return mx.reshape(unpacked, (*blocks.shape[:-1], blocks.shape[-1] * 2))


def expand_group_scales(scales: Any, group_size: int, mx: Any) -> Any:
    expanded = mx.repeat(scales[..., None], group_size, axis=-1)
    return mx.reshape(expanded, (*scales.shape[:-1], scales.shape[-1] * group_size))


def dequant_expert_weight(
    blocks: Any,
    scales: Any,
    mx: Any,
    *,
    zero_point: float = 8.0,
    group_size: int = 32,
    scale_divisor: float = 255.0,
) -> Any:
    unpacked = unpack_fp4_packed_blocks(blocks, mx).astype(mx.float32)
    scale_values = scales.astype(mx.float32)
    if scale_divisor > 0:
        scale_values = scale_values / scale_divisor
    scales_expanded = expand_group_scales(scale_values, group_size, mx).astype(mx.float32)
    return (unpacked - zero_point) * scales_expanded


def summarize_tensor_range(tensor: Any, mx: Any) -> dict[str, float]:
    return {
        "min": float(mx.min(tensor).item()),
        "max": float(mx.max(tensor).item()),
        "mean": float(mx.mean(tensor).item()),
        "abs_max": float(mx.max(mx.abs(tensor)).item()),
    }


def inspect_deepthinkingflow_moe_ffn_metadata(
    weight_shapes: dict[str, list[int]],
    config: dict[str, Any],
    *,
    layer_index: int = 0,
) -> dict[str, Any]:
    hidden_size = int(config["hidden_size"])
    num_local_experts = int(config["num_local_experts"])
    experts_per_tok = int(config["num_experts_per_tok"])
    config_intermediate_size = int(config["intermediate_size"])

    prefix = f"block.{layer_index}.mlp."
    gate_weight = weight_shapes.get(f"{prefix}gate.weight")
    gate_bias = weight_shapes.get(f"{prefix}gate.bias")
    mlp1_blocks = weight_shapes.get(f"{prefix}mlp1_weight.blocks")
    mlp1_scales = weight_shapes.get(f"{prefix}mlp1_weight.scales")
    mlp1_bias = weight_shapes.get(f"{prefix}mlp1_bias")
    mlp2_blocks = weight_shapes.get(f"{prefix}mlp2_weight.blocks")
    mlp2_scales = weight_shapes.get(f"{prefix}mlp2_weight.scales")
    mlp2_bias = weight_shapes.get(f"{prefix}mlp2_bias")
    norm_scale = weight_shapes.get(f"{prefix}norm.scale")

    problems: list[str] = []

    expected = {
        "gate.weight": [num_local_experts, hidden_size],
        "gate.bias": [num_local_experts],
        "mlp1_bias": None,
        "mlp2_bias": [num_local_experts, hidden_size],
        "norm.scale": [hidden_size],
    }
    actual = {
        "gate.weight": gate_weight,
        "gate.bias": gate_bias,
        "mlp1_bias": mlp1_bias,
        "mlp2_bias": mlp2_bias,
        "norm.scale": norm_scale,
    }
    for name, shape in expected.items():
        if shape is not None and actual.get(name) != shape:
            problems.append(f"{prefix}{name} expected {shape} but found {actual.get(name)}")

    block_width = None
    blocks_per_row = None
    quant_pack_factor = 2
    inferred_hidden = None
    inferred_expansion = None
    inferred_post_gate = None
    if mlp1_blocks and len(mlp1_blocks) == 4:
        block_width = int(mlp1_blocks[-1])
        blocks_per_row = int(mlp1_blocks[-2])
        inferred_hidden = blocks_per_row * block_width * quant_pack_factor
        if inferred_hidden != hidden_size:
            problems.append(
                f"{prefix}mlp1_weight.blocks encodes hidden_size {inferred_hidden}, expected {hidden_size}"
            )
        inferred_expansion = int(mlp1_blocks[1])
        if mlp1_blocks[0] != num_local_experts:
            problems.append(
                f"{prefix}mlp1_weight.blocks expected expert dim {num_local_experts} but found {mlp1_blocks[0]}"
            )
    else:
        problems.append(f"{prefix}mlp1_weight.blocks missing or malformed: {mlp1_blocks}")

    if mlp1_bias and inferred_expansion is not None and mlp1_bias != [num_local_experts, inferred_expansion]:
        problems.append(
            f"{prefix}mlp1_bias expected {[num_local_experts, inferred_expansion]} but found {mlp1_bias}"
        )

    if mlp1_scales and mlp1_blocks:
        expected_scales = mlp1_blocks[:-1]
        if mlp1_scales != expected_scales:
            problems.append(
                f"{prefix}mlp1_weight.scales expected {expected_scales} but found {mlp1_scales}"
            )
    else:
        problems.append(f"{prefix}mlp1_weight.scales missing or malformed: {mlp1_scales}")

    if mlp2_blocks and len(mlp2_blocks) == 4:
        if mlp2_blocks[:2] != [num_local_experts, hidden_size]:
            problems.append(
                f"{prefix}mlp2_weight.blocks expected leading dims {[num_local_experts, hidden_size]} but found {mlp2_blocks[:2]}"
            )
        inferred_post_gate = int(mlp2_blocks[-2]) * int(mlp2_blocks[-1]) * quant_pack_factor
    else:
        problems.append(f"{prefix}mlp2_weight.blocks missing or malformed: {mlp2_blocks}")

    if mlp2_scales and mlp2_blocks:
        expected_scales = mlp2_blocks[:-1]
        if mlp2_scales != expected_scales:
            problems.append(
                f"{prefix}mlp2_weight.scales expected {expected_scales} but found {mlp2_scales}"
            )
    else:
        problems.append(f"{prefix}mlp2_weight.scales missing or malformed: {mlp2_scales}")

    router_shape = {
        "router_logits": ["seq", num_local_experts],
        "topk_indices": ["seq", experts_per_tok],
        "topk_weights": ["seq", experts_per_tok],
        "expert_hidden": ["seq", experts_per_tok, hidden_size],
        "expert_expansion": ["seq", experts_per_tok, inferred_expansion],
        "expert_post_gate": ["seq", experts_per_tok, inferred_post_gate],
    }

    quantization = {
        "storage": "quantized-blocks-plus-scales",
        "mlp1_blocks_shape": mlp1_blocks,
        "mlp1_scales_shape": mlp1_scales,
        "mlp2_blocks_shape": mlp2_blocks,
        "mlp2_scales_shape": mlp2_scales,
        "block_width": block_width,
        "blocks_per_row": blocks_per_row,
        "quant_pack_factor": quant_pack_factor,
        "inferred_hidden_size": inferred_hidden,
        "inferred_expansion_size": inferred_expansion,
        "inferred_post_gate_size": inferred_post_gate,
        "dense_float_ffn_available_directly": False,
    }

    return {
        "layer_index": layer_index,
        "passed": not problems,
        "problems": problems,
        "router": {
            "num_local_experts": num_local_experts,
            "experts_per_tok": experts_per_tok,
            "gate_weight_shape": gate_weight,
            "gate_bias_shape": gate_bias,
            "expected_runtime_shapes": router_shape,
        },
        "ffn": {
            "hidden_size": hidden_size,
            "config_intermediate_size": config_intermediate_size,
            "inferred_expansion_size": inferred_expansion,
            "inferred_post_gate_size": inferred_post_gate,
            "norm_scale_shape": norm_scale,
            "mlp1_bias_shape": mlp1_bias,
            "mlp2_bias_shape": mlp2_bias,
        },
        "quantization": quantization,
        "claim_boundary": {
            "weight_storage_is_dense_ffn": False,
            "needs_dequant_or_native_kernel_for_true_forward": True,
            "shape_level_validation_only": True,
        },
    }


@dataclass(slots=True)
class KVCache:
    k: list[Any] = field(default_factory=list)
    v: list[Any] = field(default_factory=list)

    def get(self, layer_idx: int) -> tuple[Any | None, Any | None]:
        if layer_idx >= len(self.k):
            return None, None
        return self.k[layer_idx], self.v[layer_idx]

    def seq_len(self, layer_idx: int) -> int:
        if layer_idx >= len(self.k):
            return 0
        return int(self.k[layer_idx].shape[0])

    def update(
        self,
        layer_idx: int,
        new_k: Any,
        new_v: Any,
        mx: Any,
        *,
        layer_type: str,
        sliding_window: int,
    ) -> tuple[Any, Any]:
        cached_k, cached_v = self.get(layer_idx)
        if cached_k is None:
            combined_k = new_k
            combined_v = new_v
        else:
            combined_k = mx.concatenate([cached_k, new_k], axis=0)
            combined_v = mx.concatenate([cached_v, new_v], axis=0)

        if layer_type == "sliding_attention":
            combined_k = combined_k[-sliding_window:]
            combined_v = combined_v[-sliding_window:]

        while len(self.k) <= layer_idx:
            self.k.append(None)
            self.v.append(None)
        self.k[layer_idx] = combined_k
        self.v[layer_idx] = combined_v
        return combined_k, combined_v


@dataclass(slots=True)
class MLXAdapterConfig:
    model_dir: str
    quantize_4bit: bool = False
    quant_group_size: int = 64
    prefer_unified_memory_views: bool = True


class MLXInferenceAdapter:
    """Thin MLX-oriented scaffold for loading weights and running a native forward path."""

    def __init__(self, config: MLXAdapterConfig) -> None:
        self.config = config
        self.mx = _mlx_core()
        self.nn = _mlx_nn()
        self.model_dir = Path(config.model_dir).resolve()
        self._weights: dict[str, Any] = {}
        self._loaded = False

    def status(self) -> dict[str, Any]:
        return {
            "model_dir": str(self.model_dir),
            "loaded": self._loaded,
            "quantize_4bit": self.config.quantize_4bit,
            "quant_group_size": self.config.quant_group_size,
            "prefer_unified_memory_views": self.config.prefer_unified_memory_views,
            "weights_loaded": len(self._weights),
        }

    def load_weight_map(self, weight_map: dict[str, Any]) -> dict[str, Any]:
        arrays: dict[str, Any] = {}
        array_ctor = self.mx.array
        for name, value in weight_map.items():
            arrays[name] = array_ctor(value)
        if self.config.quantize_4bit:
            arrays = self.quantize_weight_map(arrays)
        self._weights = arrays
        self._loaded = True
        return arrays

    def load_safetensors_direct(self, weights_path: str | Path) -> dict[str, Any]:
        loaded = self.mx.load(str(Path(weights_path).resolve()))
        arrays = {name: value for name, value in loaded.items()}
        if self.config.quantize_4bit:
            arrays = self.quantize_weight_map(arrays)
        self._weights = arrays
        self._loaded = True
        return arrays

    def quantize_weight_map(self, weight_map: dict[str, Any]) -> dict[str, Any]:
        quantize_fn = getattr(self.mx, "quantize", None)
        if quantize_fn is None:
            raise MLXUnavailable("mlx.core.quantize is unavailable in this MLX build.")
        quantized: dict[str, Any] = {}
        for name, value in weight_map.items():
            quantized[name] = quantize_fn(value, bits=4, group_size=self.config.quant_group_size)
        return quantized

    def dequant_expert_weight(
        self,
        blocks_name: str,
        scales_name: str,
        *,
        zero_point: float = 8.0,
        group_size: int = 32,
        scale_divisor: float = 255.0,
    ) -> Any:
        if not self._loaded:
            raise RuntimeError("Weights are not loaded. Call load_safetensors_direct() first.")
        return dequant_expert_weight(
            self._weights[blocks_name],
            self._weights[scales_name],
            self.mx,
            zero_point=zero_point,
            group_size=group_size,
            scale_divisor=scale_divisor,
        )

    def dequant_layer_expert_weights(
        self,
        *,
        layer_index: int,
        projection: str,
        zero_point: float = 8.0,
        group_size: int = 32,
        scale_divisor: float = 255.0,
    ) -> Any:
        prefix = f"block.{layer_index}.mlp.{projection}_weight"
        return self.dequant_expert_weight(
            f"{prefix}.blocks",
            f"{prefix}.scales",
            zero_point=zero_point,
            group_size=group_size,
            scale_divisor=scale_divisor,
        )

    def forward_linear(self, inputs: Any, weight_name: str, bias_name: str | None = None) -> Any:
        if not self._loaded:
            raise RuntimeError("Weights are not loaded. Call load_weight_map() first.")
        if weight_name not in self._weights:
            raise KeyError(f"Missing weight '{weight_name}' in MLX adapter.")

        x = self.mx.array(inputs)
        weight = self._weights[weight_name]
        output = self.mx.matmul(x, weight)
        if bias_name:
            bias = self._weights[bias_name]
            output = output + bias
        return output

    def rms_norm(self, inputs: Any, norm_weight_name: str, *, eps: float) -> Any:
        if not self._loaded:
            raise RuntimeError("Weights are not loaded. Call load_safetensors_direct() first.")
        x = self.mx.array(inputs)
        weight = self._weights[norm_weight_name]
        squared_mean = self.mx.mean(self.mx.square(x), axis=-1, keepdims=True)
        normalized = x * self.mx.rsqrt(squared_mean + eps)
        return normalized * weight

    def repeat_kv(self, x: Any, n_rep: int) -> Any:
        if n_rep < 1:
            raise ValueError("n_rep must be >= 1")
        if n_rep == 1:
            return x
        return self.mx.repeat(x, n_rep, axis=1)

    def qkv_split(self, qkv: Any, config: dict[str, Any]) -> tuple[Any, Any, Any]:
        dims = deepthinkingflow_attention_dimensions(config)
        q_dim = dims["q_dim"]
        kv_dim = dims["kv_dim"]
        q, k, v = self.mx.split(qkv, [q_dim, q_dim + kv_dim], axis=-1)
        q = self.mx.reshape(q, (q.shape[0], dims["num_attention_heads"], dims["head_dim"]))
        k = self.mx.reshape(k, (k.shape[0], dims["num_key_value_heads"], dims["head_dim"]))
        v = self.mx.reshape(v, (v.shape[0], dims["num_key_value_heads"], dims["head_dim"]))
        return q, k, v

    def full_attention(self, q: Any, k: Any, v: Any, *, scale: float | None = None) -> Any:
        q_t = self.mx.transpose(q, (1, 0, 2))
        k_t = self.mx.transpose(k, (1, 2, 0))
        v_t = self.mx.transpose(v, (1, 0, 2))
        attn_scale = scale if scale is not None else (q.shape[-1] ** -0.5)
        scores = self.mx.matmul(q_t, k_t) * attn_scale
        weights = self.mx.softmax(scores, axis=-1)
        output = self.mx.matmul(weights, v_t)
        return self.mx.transpose(output, (1, 0, 2))

    def sliding_window_attention(
        self,
        q: Any,
        k: Any,
        v: Any,
        *,
        window: int,
        scale: float | None = None,
    ) -> Any:
        seq_q = int(q.shape[0])
        seq_k = int(k.shape[0])
        q_t = self.mx.transpose(q, (1, 0, 2))
        k_t = self.mx.transpose(k, (1, 2, 0))
        v_t = self.mx.transpose(v, (1, 0, 2))
        attn_scale = scale if scale is not None else (q.shape[-1] ** -0.5)
        scores = self.mx.matmul(q_t, k_t) * attn_scale

        q_positions = self.mx.arange(seq_k - seq_q, seq_k)
        k_positions = self.mx.arange(seq_k)
        rel = k_positions[None, :] - q_positions[:, None]
        mask = self.mx.logical_or(rel > 0, rel < -window)
        scores = self.mx.where(mask[None, :, :], float("-inf"), scores)
        weights = self.mx.softmax(scores, axis=-1)
        output = self.mx.matmul(weights, v_t)
        return self.mx.transpose(output, (1, 0, 2))

    def attention_for_layer(
        self,
        q: Any,
        k: Any,
        v: Any,
        *,
        layer_type: str,
        window: int,
        scale: float | None = None,
    ) -> Any:
        if layer_type == "sliding_attention":
            return self.sliding_window_attention(q, k, v, window=window, scale=scale)
        return self.full_attention(q, k, v, scale=scale)

    def block_attention_prep(
        self,
        inputs: Any,
        config: dict[str, Any],
        *,
        layer_index: int = 0,
        kv_cache: KVCache | None = None,
    ) -> dict[str, Any]:
        if not self._loaded:
            raise RuntimeError("Weights are not loaded. Call load_safetensors_direct() first.")

        eps = float(config.get("rms_norm_eps", 1e-5))
        dims = deepthinkingflow_attention_dimensions(config)
        layer_type = deepthinkingflow_layer_type(config, layer_index)
        sliding_window = int(config.get("sliding_window", 128))
        normed = self.rms_norm(inputs, f"block.{layer_index}.attn.norm.scale", eps=eps)
        qkv = self.forward_linear(normed, f"block.{layer_index}.attn.qkv.weight", f"block.{layer_index}.attn.qkv.bias")
        q, k_new, v_new = self.qkv_split(qkv, config)

        cache_before_k = None
        cache_before_v = None
        if kv_cache is not None:
            cache_before_k, cache_before_v = kv_cache.get(layer_index)
            k_all, v_all = kv_cache.update(
                layer_index,
                k_new,
                v_new,
                self.mx,
                layer_type=layer_type,
                sliding_window=sliding_window,
            )
        else:
            k_all, v_all = k_new, v_new

        k_rep = self.repeat_kv(k_all, dims["repeat_kv_factor"])
        v_rep = self.repeat_kv(v_all, dims["repeat_kv_factor"])
        attn_heads = self.attention_for_layer(
            q,
            k_rep,
            v_rep,
            layer_type=layer_type,
            window=sliding_window,
        )
        attn_flat = self.mx.reshape(attn_heads, (attn_heads.shape[0], dims["q_dim"]))
        attn_out = self.forward_linear(attn_flat, f"block.{layer_index}.attn.out.weight", f"block.{layer_index}.attn.out.bias")
        return {
            "layer_type": layer_type,
            "sliding_window": sliding_window,
            "normed": normed,
            "qkv": qkv,
            "q": q,
            "k_new": k_new,
            "v_new": v_new,
            "cache_before_k": cache_before_k,
            "cache_before_v": cache_before_v,
            "k_all": k_all,
            "v_all": v_all,
            "k_repeated": k_rep,
            "v_repeated": v_rep,
            "attn_heads": attn_heads,
            "attn_flat": attn_flat,
            "attn_out": attn_out,
            "shapes": {
                "q": list(q.shape),
                "k_new": list(k_new.shape),
                "v_new": list(v_new.shape),
                "k_all": list(k_all.shape),
                "v_all": list(v_all.shape),
                "k_repeated": list(k_rep.shape),
                "v_repeated": list(v_rep.shape),
                "attn_heads": list(attn_heads.shape),
                "attn_out": list(attn_out.shape),
            },
            "cache_seq_len_after": int(k_all.shape[0]),
        }

    def attention_prep(self, inputs: Any, config: dict[str, Any], *, layer_index: int = 0) -> dict[str, Any]:
        if not self._loaded:
            raise RuntimeError("Weights are not loaded. Call load_safetensors_direct() first.")
        eps = float(config.get("rms_norm_eps", 1e-5))
        dims = deepthinkingflow_attention_dimensions(config)
        normed = self.rms_norm(inputs, f"block.{layer_index}.attn.norm.scale", eps=eps)
        qkv = self.forward_linear(normed, f"block.{layer_index}.attn.qkv.weight", f"block.{layer_index}.attn.qkv.bias")
        q, k, v = self.qkv_split(qkv, config)
        k_rep = self.repeat_kv(k, dims["repeat_kv_factor"])
        v_rep = self.repeat_kv(v, dims["repeat_kv_factor"])
        attn_heads = self.full_attention(q, k_rep, v_rep)
        attn_flat = self.mx.reshape(attn_heads, (attn_heads.shape[0], dims["q_dim"]))
        attn_out = self.forward_linear(attn_flat, f"block.{layer_index}.attn.out.weight", f"block.{layer_index}.attn.out.bias")
        return {
            "normed": normed,
            "qkv": qkv,
            "q": q,
            "k": k,
            "v": v,
            "k_repeated": k_rep,
            "v_repeated": v_rep,
            "attn_heads": attn_heads,
            "attn_flat": attn_flat,
            "attn_out": attn_out,
            "shapes": {
                "normed": list(normed.shape),
                "qkv": list(qkv.shape),
                "q": list(q.shape),
                "k": list(k.shape),
                "v": list(v.shape),
                "k_repeated": list(k_rep.shape),
                "v_repeated": list(v_rep.shape),
                "attn_heads": list(attn_heads.shape),
                "attn_flat": list(attn_flat.shape),
                "attn_out": list(attn_out.shape),
            },
        }


def mlx_runtime_status(model_dir: str, *, quantize_4bit: bool = True) -> dict[str, Any]:
    adapter = MLXInferenceAdapter(
        MLXAdapterConfig(
            model_dir=model_dir,
            quantize_4bit=quantize_4bit,
            quant_group_size=64,
            prefer_unified_memory_views=True,
        )
    )
    return {
        "adapter": adapter.status(),
        "runtime_contract": {
            "unified_memory_first": True,
            "copy_avoidance_required": True,
            "mlx_native_forward_path": True,
            "direct_safetensors_load_expected": True,
            "mpsgraph_fusion_next": True,
            "coreml_ane_path_optional": True,
        },
    }

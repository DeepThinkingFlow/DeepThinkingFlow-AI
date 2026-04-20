"""Inference-loop scaffold for DeepThinkingFlow on the Apple/MLX path."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .mlx_adapter import KVCache, MLXInferenceAdapter
from .tokenizer import GPTOssTokenizer


@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    reasoning_effort: str = "medium"
    include_prompt_in_history: bool = True
    stop_on_eos: bool = True


def validate_generation_config(config: GenerationConfig) -> GenerationConfig:
    if config.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    if config.temperature < 0:
        raise ValueError("temperature must be >= 0")
    if not 0 <= config.top_p <= 1:
        raise ValueError("top_p must be in [0, 1]")
    if config.top_k < 0:
        raise ValueError("top_k must be >= 0")
    if not 0 <= config.min_p <= 1:
        raise ValueError("min_p must be in [0, 1]")
    if config.repetition_penalty < 1.0:
        raise ValueError("repetition_penalty must be >= 1.0")
    if config.reasoning_effort not in {"low", "medium", "high"}:
        raise ValueError("reasoning_effort must be one of: low, medium, high")
    return config


def _flatten_values(values: Any) -> list[Any]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    else:
        values = values
    if isinstance(values, list):
        if values and isinstance(values[0], list):
            flat: list[Any] = []
            for row in values:
                flat.extend(row)
            return flat
        return list(values)
    return [values]


def apply_repetition_penalty_to_logits(logits: list[float], generated_ids: list[int], penalty: float) -> list[float]:
    if penalty <= 1.0 or not generated_ids:
        return list(logits)
    adjusted = list(logits)
    seen = set(int(token_id) for token_id in generated_ids)
    for token_id in seen:
        if 0 <= token_id < len(adjusted):
            value = adjusted[token_id]
            adjusted[token_id] = value / penalty if value > 0 else value * penalty
    return adjusted


def softmax_list(logits: list[float], temperature: float) -> list[float]:
    import math

    safe_temperature = temperature if temperature > 0 else 1.0
    scaled = [value / safe_temperature for value in logits]
    max_logit = max(scaled)
    exps = [math.exp(value - max_logit) for value in scaled]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


def build_sampling_distribution(
    logits: list[float],
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    repetition_penalty: float,
    generated_ids: list[int] | None = None,
) -> dict[str, Any]:
    validate_generation_config(
        GenerationConfig(
            max_new_tokens=1,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
        )
    )
    generated_ids = generated_ids or []
    adjusted_logits = apply_repetition_penalty_to_logits(logits, generated_ids, repetition_penalty)
    probs = softmax_list(adjusted_logits, temperature if temperature > 0 else 1.0)
    ranked = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)

    if min_p > 0:
        ranked = [item for item in ranked if item[1] >= min_p] or ranked[:1]
    if top_k > 0:
        ranked = ranked[:top_k]
    if 0 < top_p < 1.0:
        cumulative = 0.0
        nucleus: list[tuple[int, float]] = []
        for item in ranked:
            nucleus.append(item)
            cumulative += item[1]
            if cumulative >= top_p:
                break
        ranked = nucleus or ranked[:1]

    kept_total = sum(prob for _, prob in ranked) or 1.0
    normalized = [(token_id, prob / kept_total) for token_id, prob in ranked]
    return {
        "adjusted_logits": adjusted_logits,
        "kept_token_ids": [token_id for token_id, _ in normalized],
        "kept_probabilities": [prob for _, prob in normalized],
        "argmax_token_id": max(range(len(adjusted_logits)), key=adjusted_logits.__getitem__),
    }


def sample_next_token_id(
    logits: list[float],
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    repetition_penalty: float,
    generated_ids: list[int] | None = None,
) -> dict[str, Any]:
    distribution = build_sampling_distribution(
        logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        generated_ids=generated_ids,
    )
    if temperature <= 0:
        distribution["selected_token_id"] = distribution["argmax_token_id"]
        distribution["selection_mode"] = "greedy"
        return distribution
    distribution["selected_token_id"] = distribution["kept_token_ids"][0]
    distribution["selection_mode"] = "ranked-sample-scaffold"
    return distribution


def load_config(model_dir: str | Path) -> dict[str, Any]:
    model_path = Path(model_dir).resolve()
    return json.loads((model_path / "config.json").read_text(encoding="utf-8"))


def embed(input_ids: Any, weights: dict[str, Any]) -> Any:
    import mlx.core as mx

    token_ids = mx.array(input_ids, dtype=mx.int32)
    return weights["embedding.weight"][token_ids]


def lm_head(hidden_states: Any, weights: dict[str, Any]) -> Any:
    import mlx.core as mx

    return mx.matmul(hidden_states, mx.transpose(weights["unembedding.weight"], (1, 0)))


def block_forward(
    x: Any,
    adapter: MLXInferenceAdapter,
    config: dict[str, Any],
    *,
    layer_idx: int,
    kv_cache: KVCache,
) -> Any:
    attn = adapter.block_attention_prep(x, config, layer_index=layer_idx, kv_cache=kv_cache)
    # MoE path is intentionally not wired yet because exact checkpoint-runtime parity
    # still needs real reference validation on hardware that can load the checkpoint.
    return attn["attn_out"]


def prefill_hidden(
    input_ids: Any,
    adapter: MLXInferenceAdapter,
    config: dict[str, Any],
    *,
    kv_cache: KVCache,
) -> Any:
    x = embed(input_ids, adapter._weights)
    for layer_idx in range(int(config["num_hidden_layers"])):
        x = block_forward(x, adapter, config, layer_idx=layer_idx, kv_cache=kv_cache)
    return x


def decode_one_token(
    current_hidden: Any,
    adapter: MLXInferenceAdapter,
    config: dict[str, Any],
    kv_cache: KVCache,
    *,
    token_id: int,
) -> Any:
    import mlx.core as mx

    x = embed(mx.array([token_id], dtype=mx.int32), adapter._weights)
    for layer_idx in range(int(config["num_hidden_layers"])):
        x = block_forward(x, adapter, config, layer_idx=layer_idx, kv_cache=kv_cache)
    return x


def kv_cache_decode_status(config: dict[str, Any], kv_cache: KVCache) -> dict[str, Any]:
    layer_types = list(config.get("layer_types") or [])
    sliding_window = int(config.get("sliding_window", 128))
    layer_summaries = []
    for layer_idx in range(int(config["num_hidden_layers"])):
        layer_type = layer_types[layer_idx] if layer_idx < len(layer_types) else "full_attention"
        current_len = kv_cache.seq_len(layer_idx)
        expected_max = sliding_window if layer_type == "sliding_attention" else None
        layer_summaries.append(
            {
                "layer_index": layer_idx,
                "layer_type": layer_type,
                "cache_seq_len": current_len,
                "expected_max_seq_len": expected_max,
                "within_expected_limit": expected_max is None or current_len <= expected_max,
            }
        )
    return {
        "num_layers": int(config["num_hidden_layers"]),
        "sliding_window": sliding_window,
        "layers": layer_summaries,
        "all_layers_within_expected_limits": all(item["within_expected_limit"] for item in layer_summaries),
    }


def generate(
    prompt: str,
    tokenizer: GPTOssTokenizer,
    adapter: MLXInferenceAdapter,
    config: dict[str, Any],
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    reasoning_effort: str = "medium",
) -> str:
    import mlx.core as mx

    if not adapter._loaded:
        raise RuntimeError("Weights are not loaded. Call load_safetensors_direct() first.")
    prompt_package = tokenizer.build_prompt_package(
        [{"role": "user", "content": prompt}],
        reasoning_effort=reasoning_effort,
        add_generation_prompt=True,
    )
    input_ids = prompt_package["input_ids"]
    kv_cache = KVCache()
    generated: list[int] = []
    x = prefill_hidden(input_ids, adapter, config, kv_cache=kv_cache)
    stop_ids = set(int(token_id) for token_id in tokenizer.stop_token_ids())

    for _ in range(max_new_tokens):
        logits = lm_head(x[-1:], adapter._weights)
        logits_list = _flatten_values(logits[0]) if len(getattr(logits, "shape", [])) > 1 else _flatten_values(logits)
        sampling = sample_next_token_id(
            [float(value) for value in logits_list],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            generated_ids=generated,
        )
        token_id = int(sampling["selected_token_id"])
        if token_id == tokenizer.eos_id or token_id in stop_ids:
            break
        generated.append(token_id)
        x = decode_one_token(x, adapter, config, kv_cache, token_id=token_id)

    return tokenizer.decode(mx.array(generated, dtype=mx.int32))


def verify_generation_contract(
    *,
    tokenizer: GPTOssTokenizer,
    config: dict[str, Any],
    prompt: str,
    sampling: GenerationConfig | None = None,
) -> dict[str, Any]:
    sampling = sampling or GenerationConfig()
    validate_generation_config(sampling)
    prompt_package = tokenizer.build_prompt_package(
        [{"role": "user", "content": prompt}],
        reasoning_effort=sampling.reasoning_effort,
        add_generation_prompt=True,
    )
    return {
        "prompt": prompt,
        "prompt_package_ready": True,
        "rendered_prompt_non_empty": bool(prompt_package["rendered_prompt"]),
        "input_token_count": len(_flatten_values(prompt_package["input_ids"])),
        "stop_token_ids": tokenizer.stop_token_ids(),
        "sampling_contract": asdict(sampling),
        "layer_types_count": len(list(config.get("layer_types") or [])),
        "num_hidden_layers": int(config["num_hidden_layers"]),
    }


def inference_scaffold_status(model_dir: str | Path) -> dict[str, Any]:
    config = load_config(model_dir)
    generation = GenerationConfig()
    validation_rules = {
        "max_new_tokens_min": 1,
        "temperature_min": 0.0,
        "top_p_range": [0.0, 1.0],
        "top_k_min": 0,
        "min_p_range": [0.0, 1.0],
        "repetition_penalty_min": 1.0,
        "reasoning_effort_values": ["low", "medium", "high"],
    }
    return {
        "model_dir": str(Path(model_dir).resolve()),
        "embedding_key": "embedding.weight",
        "lm_head_key": "unembedding.weight",
        "num_hidden_layers": int(config["num_hidden_layers"]),
        "layer_types": list(config.get("layer_types") or []),
        "sliding_window": int(config.get("sliding_window", 128)),
        "tokenizer_required": True,
        "tokenizer_features": {
            "chat_template_rendering": True,
            "reasoning_effort_prompting": True,
            "prompt_package_builder": True,
            "stop_token_discovery": True,
            "batch_encoding": True,
        },
        "generation_features": {
            "prefill_path": True,
            "decode_loop": True,
            "kv_cache_scaffold": True,
            "kv_cache_decode_status_report": True,
            "temperature": True,
            "top_p_runtime": True,
            "top_k_runtime": True,
            "min_p_runtime": True,
            "repetition_penalty_runtime": True,
            "stop_token_runtime": True,
            "streaming_tokens": False,
        },
        "default_generation_config": asdict(generation),
        "generation_validation_rules": validation_rules,
        "claim_boundary": {
            "attention_runtime_path_wired": True,
            "moe_runtime_path_wired": False,
            "kv_cache_decode_path_fully_verified": False,
            "end_to_end_generation_path_fully_verified": False,
            "sampling_runtime_fully_verified": False,
        },
        "moe_runtime_status": "attention path wired, moe path intentionally scaffold-only",
    }

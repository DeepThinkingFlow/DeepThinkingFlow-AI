"""Inference-loop scaffold for DeepThinkingFlow on the Apple/MLX path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .mlx_adapter import KVCache, MLXInferenceAdapter
from .tokenizer import GPTOssTokenizer


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


def generate(
    prompt: str,
    tokenizer: GPTOssTokenizer,
    adapter: MLXInferenceAdapter,
    config: dict[str, Any],
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
) -> str:
    import mlx.core as mx

    if not adapter._loaded:
        raise RuntimeError("Weights are not loaded. Call load_safetensors_direct() first.")

    input_ids = tokenizer.encode(prompt)
    kv_cache = KVCache()
    generated: list[int] = []
    x = prefill_hidden(input_ids, adapter, config, kv_cache=kv_cache)

    for _ in range(max_new_tokens):
        logits = lm_head(x[-1:], adapter._weights)
        if temperature > 0:
            probs = mx.softmax(logits / temperature, axis=-1)
            next_id = mx.random.categorical(probs)
        else:
            next_id = mx.argmax(logits, axis=-1)

        token_id = int(next_id.item())
        if token_id == tokenizer.eos_id:
            break
        generated.append(token_id)

        x = embed(mx.array([token_id], dtype=mx.int32), adapter._weights)
        for layer_idx in range(int(config["num_hidden_layers"])):
            x = block_forward(x, adapter, config, layer_idx=layer_idx, kv_cache=kv_cache)

    return tokenizer.decode(mx.array(generated, dtype=mx.int32))


def inference_scaffold_status(model_dir: str | Path) -> dict[str, Any]:
    config = load_config(model_dir)
    return {
        "model_dir": str(Path(model_dir).resolve()),
        "embedding_key": "embedding.weight",
        "lm_head_key": "unembedding.weight",
        "num_hidden_layers": int(config["num_hidden_layers"]),
        "layer_types": list(config.get("layer_types") or []),
        "sliding_window": int(config.get("sliding_window", 128)),
        "tokenizer_required": True,
        "moe_runtime_status": "attention path wired, moe path intentionally scaffold-only",
    }

#!/usr/bin/env python3
"""Shared helpers for DeepThinkingFlow Transformers entrypoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_MODEL_DIR = "runtime/transformers/DeepThinkingFlow"
DEFAULT_BUNDLE_DIR = "behavior/DeepThinkingFlow"


def ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"Missing {label}: {path}")


def ensure_local_model_dir(model_path: Path) -> None:
    ensure_file(model_path / "config.json", "model config")
    ensure_file(model_path / "tokenizer.json", "tokenizer")
    ensure_file(model_path / "chat_template.jinja", "chat template")


def resolve_model_ref(model_dir: str) -> tuple[str, Path]:
    model_path = Path(model_dir).resolve()
    if model_path.exists():
        ensure_local_model_dir(model_path)
        return str(model_path), model_path
    return model_dir, model_path


def resolve_bundle_dir(bundle: str) -> Path:
    bundle_dir = Path(bundle).resolve()
    ensure_file(bundle_dir / "system_prompt.txt", "bundle system prompt")
    return bundle_dir


def load_system_prompt(bundle_dir: Path, reasoning_effort: str, reasoning_in_system: bool) -> str:
    system_prompt = (bundle_dir / "system_prompt.txt").read_text(encoding="utf-8").strip()
    if reasoning_in_system:
        system_prompt = f"{system_prompt}\n\nReasoning: {reasoning_effort}"
    return system_prompt


def get_system_memory_gib() -> float | None:
    if not hasattr(os, "sysconf"):
        return None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        return None
    return (pages * page_size) / (1024 ** 3)


def estimate_local_weight_size_gib(model_dir: Path) -> float | None:
    if not model_dir.exists():
        return None
    weight_files = sorted(model_dir.glob("model-*.safetensors"))
    if not weight_files:
        return None
    total = sum(path.stat().st_size for path in weight_files)
    return total / (1024 ** 3)


def build_low_memory_warning_payload(model_path: Path) -> dict[str, Any] | None:
    mem_gib = get_system_memory_gib()
    if mem_gib is None or mem_gib >= 16:
        return None
    weight_gib = estimate_local_weight_size_gib(model_path)
    return {
        "warning": "This machine has less than 16 GiB RAM. DeepThinkingFlow Transformers inference may fail or thrash heavily here.",
        "system_ram_gib": round(mem_gib, 2),
        "local_weight_gib": round(weight_gib, 2) if weight_gib is not None else None,
    }


def extract_final_text(decoded_completion: str) -> str:
    marker = "<|channel|>final<|message|>"
    text = decoded_completion.split(marker, 1)[-1] if marker in decoded_completion else decoded_completion
    text = text.replace("<|start|>assistant", "").replace("<|message|>", "")
    for stop in ("<|return|>", "<|call|>", "<|end|>", "<|channel|>analysis", "<|channel|>commentary"):
        if stop in text:
            text = text.split(stop, 1)[0]
    return text.strip()


def extract_analysis_text(decoded_completion: str) -> str:
    marker = "<|channel|>analysis<|message|>"
    if marker not in decoded_completion:
        return ""
    text = decoded_completion.split(marker, 1)[1]
    for stop in ("<|end|>", "<|call|>", "<|return|>", "<|channel|>final<|message|>"):
        if stop in text:
            text = text.split(stop, 1)[0]
    return text.strip()


def import_transformers_runtime() -> tuple[Any, Any]:
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise SystemExit(
            "torch and transformers are required to run generation. "
            "See the official GPT-OSS compatibility guidance: pip install -U transformers accelerate torch triton==3.4 kernels"
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_ref: str,
    *,
    device_map: str,
    torch_dtype: str,
    attn_implementation: str | None,
) -> tuple[Any, Any]:
    AutoModelForCausalLM, AutoTokenizer = import_transformers_runtime()
    tokenizer = AutoTokenizer.from_pretrained(model_ref)

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_ref, **model_kwargs)
    return tokenizer, model


def render_prompt(tokenizer: Any, messages: list[dict[str, str]], reasoning_effort: str) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort=reasoning_effort,
    )


def generate_response(
    model: Any,
    tokenizer: Any,
    *,
    messages: list[dict[str, str]],
    reasoning_effort: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, str]:
    input_device = model.get_input_embeddings().weight.device
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort=reasoning_effort,
    ).to(input_device)

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    if generate_kwargs["do_sample"]:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    outputs = model.generate(
        **inputs,
        **generate_kwargs,
    )
    completion_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    decoded_completion = tokenizer.decode(completion_ids, skip_special_tokens=False)
    analysis_text = extract_analysis_text(decoded_completion)
    final_text = extract_final_text(decoded_completion)
    return {
        "decoded_completion": decoded_completion,
        "analysis_text": analysis_text,
        "final_text": final_text,
    }

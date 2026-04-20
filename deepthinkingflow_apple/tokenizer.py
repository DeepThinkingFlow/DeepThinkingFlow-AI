"""Tokenizer scaffold for DeepThinkingFlow on the Apple/MLX path."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class GPTOssTokenizer:
    def __init__(self, model_dir: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "transformers is required for GPTOssTokenizer. Install transformers in the current environment."
            ) from exc

        self.model_dir = str(Path(model_dir).resolve())
        self.tok = AutoTokenizer.from_pretrained(self.model_dir)

    def _array(self, values: Any):
        try:
            import mlx.core as mx
        except ModuleNotFoundError:
            if hasattr(values, "tolist"):
                return values.tolist()
            return values
        return mx.array(values, dtype=mx.int32)

    def encode(self, text: str, *, add_special_tokens: bool = True):
        ids = self.tok.encode(text, add_special_tokens=add_special_tokens)
        return self._array(ids)

    def decode(self, ids, *, skip_special_tokens: bool = True) -> str:
        if hasattr(ids, "tolist"):
            token_ids = ids.tolist()
        else:
            token_ids = list(ids)
        return self.tok.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def encode_batch(
        self,
        texts: list[str],
        *,
        padding: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
        return_attention_mask: bool = False,
    ):
        out = self.tok(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors="np",
        )
        input_ids = self._array(out["input_ids"])
        if not return_attention_mask:
            return input_ids
        attention_mask = self._array(out["attention_mask"])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def render_chat(
        self,
        messages: list[dict[str, str]],
        *,
        reasoning_effort: str = "medium",
        add_generation_prompt: bool = True,
        tokenize: bool = False,
    ) -> Any:
        apply_chat_template = getattr(self.tok, "apply_chat_template", None)
        if apply_chat_template is None:
            raise RuntimeError(
                "Tokenizer does not expose apply_chat_template; this model directory is not chat-template ready."
            )
        return apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            reasoning_effort=reasoning_effort,
        )

    def build_prompt_package(
        self,
        messages: list[dict[str, str]],
        *,
        reasoning_effort: str = "medium",
        add_generation_prompt: bool = True,
    ) -> dict[str, Any]:
        rendered = self.render_chat(
            messages,
            reasoning_effort=reasoning_effort,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )
        token_ids = self.encode(str(rendered), add_special_tokens=False)
        return {
            "rendered_prompt": str(rendered),
            "input_ids": token_ids,
            "reasoning_effort": reasoning_effort,
            "messages": messages,
            "chat_template_supported": True,
        }

    def stop_token_ids(self) -> list[int]:
        eos_token_id = getattr(self.tok, "eos_token_id", None)
        generation_config = getattr(self.tok, "init_kwargs", {}) or {}
        eos_candidates: list[int] = []
        if isinstance(eos_token_id, int):
            eos_candidates.append(eos_token_id)
        elif isinstance(eos_token_id, (list, tuple)):
            eos_candidates.extend(int(value) for value in eos_token_id)
        extra_ids = generation_config.get("eos_token_id")
        if isinstance(extra_ids, int):
            eos_candidates.append(extra_ids)
        elif isinstance(extra_ids, (list, tuple)):
            eos_candidates.extend(int(value) for value in extra_ids)
        deduped: list[int] = []
        seen: set[int] = set()
        for token_id in eos_candidates:
            if token_id not in seen:
                seen.add(token_id)
                deduped.append(token_id)
        return deduped

    def special_token_ids(self) -> dict[str, int | None]:
        return {
            "bos_token_id": getattr(self.tok, "bos_token_id", None),
            "eos_token_id": getattr(self.tok, "eos_token_id", None),
            "pad_token_id": getattr(self.tok, "pad_token_id", None),
        }

    def compatibility_report(self) -> dict[str, Any]:
        chat_template = getattr(self.tok, "chat_template", None)
        return {
            "model_dir": self.model_dir,
            "chat_template_supported": getattr(self.tok, "apply_chat_template", None) is not None,
            "chat_template_present": bool(chat_template),
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_token_ids(),
            "stop_token_ids": self.stop_token_ids(),
            "padding_side": getattr(self.tok, "padding_side", None),
            "truncation_side": getattr(self.tok, "truncation_side", None),
        }

    @property
    def eos_id(self) -> int:
        return int(self.tok.eos_token_id)

    @property
    def vocab_size(self) -> int:
        return int(self.tok.vocab_size)


DeepThinkingFlowTokenizer = GPTOssTokenizer

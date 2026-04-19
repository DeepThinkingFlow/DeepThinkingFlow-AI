"""Tokenizer scaffold for DeepThinkingFlow on the Apple/MLX path."""

from __future__ import annotations

from pathlib import Path


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

    def encode(self, text: str):
        import mlx.core as mx

        ids = self.tok.encode(text, add_special_tokens=True)
        return mx.array(ids, dtype=mx.int32)

    def decode(self, ids) -> str:
        return self.tok.decode(ids.tolist(), skip_special_tokens=True)

    def encode_batch(self, texts: list[str]):
        import mlx.core as mx

        out = self.tok(texts, padding=True, return_tensors="np")
        return mx.array(out["input_ids"], dtype=mx.int32)

    @property
    def eos_id(self) -> int:
        return int(self.tok.eos_token_id)

    @property
    def vocab_size(self) -> int:
        return int(self.tok.vocab_size)

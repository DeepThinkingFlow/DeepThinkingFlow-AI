#!/usr/bin/env python3
"""Prepare OpenThoughts3 and OpenCodeInstruct into chat-templated training text.

This script:
- downloads both datasets through HuggingFace `datasets`
- filters them according to the requested rules
- converts each sample into a unified chat `text` field through
  `AutoTokenizer.apply_chat_template()`
- drops samples that exceed the target token length without truncating
- saves each processed dataset separately with `save_to_disk()`
"""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Any

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
OPEN_THOUGHTS_DATASET = "open-thoughts/OpenThoughts3-1.2M"
OPEN_CODE_DATASET = "nvidia/OpenCodeInstruct"
OPEN_THOUGHTS_ALLOWED_DOMAINS = {"math", "science"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare OpenThoughts3 and OpenCodeInstruct datasets for QLoRA training."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Tokenizer model name. Falls back to {DEFAULT_MODEL_NAME} when MODEL_NAME is unset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Base output directory for processed datasets.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=4096,
        help="Maximum allowed token length. Longer samples are dropped.",
    )
    parser.add_argument(
        "--ot3_limit",
        type=int,
        default=50000,
        help="Maximum number of OpenThoughts3 rows to keep after filtering.",
    )
    parser.add_argument(
        "--oci_limit",
        type=int,
        default=100000,
        help="Maximum number of OpenCodeInstruct rows to keep after filtering.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=2,
        help="Number of processes for datasets.map(). Use 1 for simpler debugging.",
    )
    return parser.parse_args()


def resolve_model_name(args: argparse.Namespace) -> str:
    # The environment variable wins when present, otherwise use the CLI value.
    return os.environ.get("MODEL_NAME", args.model_name)


def normalize_role(raw_role: str) -> str | None:
    lowered = str(raw_role).strip().lower()
    if lowered in {"human", "user"}:
        return "user"
    if lowered in {"gpt", "assistant"}:
        return "assistant"
    return None


def build_ot3_messages(conversations: Any) -> list[dict[str, str]] | None:
    if not isinstance(conversations, list) or not conversations:
        return None

    messages: list[dict[str, str]] = []
    for turn in conversations:
        if not isinstance(turn, dict):
            return None
        role = normalize_role(turn.get("from", ""))
        content = turn.get("value", "")
        if role is None or not isinstance(content, str) or not content.strip():
            return None
        messages.append({"role": role, "content": content.strip()})

    if not messages or messages[0]["role"] != "user" or messages[-1]["role"] != "assistant":
        return None
    return messages


def build_oci_messages(user_text: Any, assistant_text: Any) -> list[dict[str, str]] | None:
    if not isinstance(user_text, str) or not user_text.strip():
        return None
    if not isinstance(assistant_text, str) or not assistant_text.strip():
        return None
    return [
        {"role": "user", "content": user_text.strip()},
        {"role": "assistant", "content": assistant_text.strip()},
    ]


def _warn_skip(dataset_name: str, reason: str) -> None:
    warnings.warn(f"[{dataset_name}] skipped sample: {reason}", stacklevel=2)


def process_ot3_batch(batch: dict[str, list[Any]], *, tokenizer, max_seq_len: int) -> dict[str, list[Any]]:
    texts: list[str] = []
    messages_out: list[list[dict[str, str]]] = []
    sources: list[str] = []
    for conversations in batch["conversations"]:
        try:
            messages = build_ot3_messages(conversations)
            if messages is None:
                _warn_skip("OpenThoughts3", "invalid conversations format")
                continue

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(token_ids) > max_seq_len:
                continue
            texts.append(text)
            messages_out.append(messages)
            sources.append("OpenThoughts3")
        except Exception as exc:  # pragma: no cover - defensive runtime path
            _warn_skip("OpenThoughts3", str(exc))
            continue
    return {"messages": messages_out, "text": texts, "source_dataset": sources}


def process_oci_batch(batch: dict[str, list[Any]], *, tokenizer, max_seq_len: int) -> dict[str, list[Any]]:
    texts: list[str] = []
    messages_out: list[list[dict[str, str]]] = []
    sources: list[str] = []
    for user_text, assistant_text in zip(batch["input"], batch["output"]):
        try:
            messages = build_oci_messages(user_text, assistant_text)
            if messages is None:
                _warn_skip("OpenCodeInstruct", "invalid input/output fields")
                continue

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(token_ids) > max_seq_len:
                continue
            texts.append(text)
            messages_out.append(messages)
            sources.append("OpenCodeInstruct")
        except Exception as exc:  # pragma: no cover - defensive runtime path
            _warn_skip("OpenCodeInstruct", str(exc))
            continue
    return {"messages": messages_out, "text": texts, "source_dataset": sources}


def ensure_non_empty(dataset: Dataset, label: str) -> None:
    if len(dataset) == 0:
        raise SystemExit(f"{label} is empty after filtering and formatting.")


def prepare_openthoughts3(*, tokenizer, args: argparse.Namespace) -> Dataset:
    dataset = load_dataset(OPEN_THOUGHTS_DATASET, split="train")
    dataset = dataset.filter(
        lambda row: str(row.get("domain", "")).strip().lower() in OPEN_THOUGHTS_ALLOWED_DOMAINS
    )
    dataset = dataset.select(range(min(args.ot3_limit, len(dataset))))
    dataset = dataset.map(
        lambda batch: process_ot3_batch(batch, tokenizer=tokenizer, max_seq_len=args.max_seq_len),
        batched=True,
        num_proc=args.num_proc,
        remove_columns=dataset.column_names,
        desc="Formatting OpenThoughts3",
    )
    ensure_non_empty(dataset, "OpenThoughts3 processed dataset")
    return dataset


def prepare_opencodeinstruct(*, tokenizer, args: argparse.Namespace) -> Dataset:
    dataset = load_dataset(OPEN_CODE_DATASET, split="train")
    dataset = dataset.filter(
        lambda row: float(row.get("average_test_score", 0.0)) >= 0.8
    )
    dataset = dataset.select(range(min(args.oci_limit, len(dataset))))
    dataset = dataset.map(
        lambda batch: process_oci_batch(batch, tokenizer=tokenizer, max_seq_len=args.max_seq_len),
        batched=True,
        num_proc=args.num_proc,
        remove_columns=dataset.column_names,
        desc="Formatting OpenCodeInstruct",
    )
    ensure_non_empty(dataset, "OpenCodeInstruct processed dataset")
    return dataset


def verify_token_lengths(dataset: Dataset, tokenizer, max_seq_len: int, label: str) -> None:
    for index in range(len(dataset)):
        token_ids = tokenizer(dataset[index]["text"], add_special_tokens=False)["input_ids"]
        if len(token_ids) > max_seq_len:
            raise SystemExit(
                f"{label} contains a sample longer than max_seq_len at index={index}: {len(token_ids)} > {max_seq_len}"
            )


def main() -> int:
    args = parse_args()
    model_name = resolve_model_name(args)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    openthoughts_dataset = prepare_openthoughts3(tokenizer=tokenizer, args=args)
    opencode_dataset = prepare_opencodeinstruct(tokenizer=tokenizer, args=args)

    verify_token_lengths(openthoughts_dataset, tokenizer, args.max_seq_len, "OpenThoughts3")
    verify_token_lengths(opencode_dataset, tokenizer, args.max_seq_len, "OpenCodeInstruct")

    openthoughts_output = os.path.join(args.output_dir, "openthoughts3_processed")
    opencode_output = os.path.join(args.output_dir, "opencodeinstruct_processed")

    openthoughts_dataset.save_to_disk(openthoughts_output)
    opencode_dataset.save_to_disk(opencode_output)

    print(f"OpenThoughts3 final rows: {len(openthoughts_dataset)}")
    print(f"OpenCodeInstruct final rows: {len(opencode_dataset)}")
    print(f"Saved OpenThoughts3 to: {openthoughts_output}")
    print(f"Saved OpenCodeInstruct to: {opencode_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Production-oriented DeepThinkingFlow LoRA/SFT training scaffold using Transformers Trainer."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepthinkingflow_env import detect_dependency_status, inject_local_site_packages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DeepThinkingFlow LoRA adapter with a harmony-format SFT dataset."
    )
    parser.add_argument(
        "--config",
        default="training/DeepThinkingFlow-lora/config.example.json",
        help="Path to the JSON config file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and dataset, then print a summary without training.",
    )
    return parser.parse_args()


def ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"Missing {label}: {path}")


def ensure_any_file(path: Path, candidates: list[str], label: str) -> None:
    if not any((path / candidate).is_file() for candidate in candidates):
        formatted = ", ".join(candidates)
        raise ValueError(f"Missing {label} in {path}. Expected one of: {formatted}")


def resolve_model_reference(model_name_or_path: str) -> str:
    path = Path(model_name_or_path)
    if path.exists():
        return str(path.resolve())
    return model_name_or_path


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_keys(config: dict[str, Any], keys: list[str]) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def validate_messages(rows: list[dict[str, Any]], label: str) -> None:
    if not rows:
        raise ValueError(f"{label} dataset is empty.")
    for idx, row in enumerate(rows, start=1):
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{label} row {idx} is missing a non-empty messages list.")
        if messages[-1].get("role") != "assistant":
            raise ValueError(f"{label} row {idx} must end with an assistant message.")
        if "content" not in messages[-1]:
            raise ValueError(f"{label} row {idx} assistant message must include content.")
        if "thinking" in messages[-1] and not isinstance(messages[-1]["thinking"], str):
            raise ValueError(f"{label} row {idx} assistant thinking must be a string.")


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config)
    cfg.setdefault("behavior_bundle_dir", "")
    cfg.setdefault("eval_dataset_path", "")
    cfg.setdefault("base_eval_cases_path", "")
    cfg.setdefault("skill_eval_cases_path", "")
    cfg.setdefault("expected_train_dataset_path", "")
    cfg.setdefault("expected_eval_dataset_path", "")
    cfg.setdefault("val_split_ratio", 0.1)
    cfg.setdefault("seed", 42)
    cfg.setdefault("fp16", False)
    cfg.setdefault("weight_decay", 0.01)
    cfg.setdefault("warmup_ratio", 0.03)
    cfg.setdefault("logging_steps", 10)
    cfg.setdefault("eval_steps", 100)
    cfg.setdefault("save_steps", 100)
    cfg.setdefault("save_total_limit", 2)
    cfg.setdefault("gradient_checkpointing", True)
    cfg.setdefault("max_grad_norm", 1.0)
    cfg.setdefault("lr_scheduler_type", "cosine")
    cfg.setdefault("attn_implementation", "eager")
    cfg.setdefault("merge_after_train", False)
    cfg.setdefault("use_qlora", False)
    cfg.setdefault("load_in_4bit", False)
    cfg.setdefault("target_parameters", [])
    cfg.setdefault("resume_from_checkpoint", "")
    cfg.setdefault("report_to", [])
    cfg.setdefault("optim", "adamw_torch")
    cfg.setdefault("dataloader_num_workers", 0)
    cfg.setdefault("early_stopping_patience", 3)
    cfg.setdefault("train_on_full_text", False)
    cfg.setdefault("max_train_samples", 0)
    cfg.setdefault("max_eval_samples", 0)
    cfg.setdefault("ddp_find_unused_parameters", False)
    cfg.setdefault("require_all_target_modules_hit", True)
    cfg.setdefault("min_target_module_matches", len(cfg.get("target_modules", [])))
    cfg.setdefault("min_trainable_params", 1)
    return cfg


def validate_config(config: dict[str, Any]) -> None:
    ensure_keys(
        config,
        [
            "model_name_or_path",
            "dataset_path",
            "output_dir",
            "bf16",
            "num_train_epochs",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "max_seq_length",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "target_modules",
            "reasoning_effort",
        ],
    )
    if config["reasoning_effort"] not in {"low", "medium", "high"}:
        raise ValueError("reasoning_effort must be one of: low, medium, high")
    if not config["target_modules"]:
        raise ValueError("target_modules must not be empty")
    if config["eval_dataset_path"] and config["val_split_ratio"] not in (0, 0.0):
        raise ValueError("Use either eval_dataset_path or val_split_ratio, not both.")
    if not config["eval_dataset_path"] and not (0 <= float(config["val_split_ratio"]) < 1):
        raise ValueError("val_split_ratio must be in [0, 1).")
    if config["use_qlora"] and not config["load_in_4bit"]:
        raise ValueError("use_qlora=true requires load_in_4bit=true.")
    if config["bf16"] and config.get("fp16", False):
        raise ValueError("Choose bf16 or fp16, not both.")
    if config["early_stopping_patience"] < 1:
        raise ValueError("early_stopping_patience must be >= 1.")
    if int(config["min_target_module_matches"]) < 1:
        raise ValueError("min_target_module_matches must be >= 1.")
    if int(config["min_trainable_params"]) < 1:
        raise ValueError("min_trainable_params must be >= 1.")

    model_path = Path(config["model_name_or_path"]).resolve()
    if model_path.exists():
        ensure_file(model_path / "config.json", "model config")
        ensure_file(model_path / "tokenizer.json", "tokenizer")
        ensure_file(model_path / "chat_template.jinja", "chat template")
        ensure_any_file(
            model_path,
            ["model.safetensors", "model.safetensors.index.json", "pytorch_model.bin"],
            "model weights",
        )

    dataset_path = Path(config["dataset_path"]).resolve()
    ensure_file(dataset_path, "training dataset")
    if config["eval_dataset_path"]:
        ensure_file(Path(config["eval_dataset_path"]).resolve(), "eval dataset")
    if config["behavior_bundle_dir"]:
        bundle_dir = Path(config["behavior_bundle_dir"]).resolve()
        ensure_file(bundle_dir / "profile.json", "behavior bundle profile")
        ensure_file(bundle_dir / "system_prompt.txt", "behavior bundle system prompt")
    if config["base_eval_cases_path"]:
        ensure_file(Path(config["base_eval_cases_path"]).resolve(), "base eval cases")
    if config["skill_eval_cases_path"]:
        ensure_file(Path(config["skill_eval_cases_path"]).resolve(), "skill eval cases")
    if config["expected_train_dataset_path"]:
        expected_train_dataset_path = str(Path(config["expected_train_dataset_path"]).resolve())
        if expected_train_dataset_path != str(dataset_path):
            raise ValueError("dataset_path does not match expected_train_dataset_path")
    if config["expected_eval_dataset_path"] and config["eval_dataset_path"]:
        expected_eval_dataset_path = str(Path(config["expected_eval_dataset_path"]).resolve())
        actual_eval_dataset_path = str(Path(config["eval_dataset_path"]).resolve())
        if expected_eval_dataset_path != actual_eval_dataset_path:
            raise ValueError("eval_dataset_path does not match expected_eval_dataset_path")


def split_rows(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if val_ratio <= 0:
        return rows, []
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_rows = shuffled[:val_count]
    train_rows = shuffled[val_count:]
    if not train_rows:
        raise ValueError("Validation split consumed the entire dataset. Reduce val_split_ratio.")
    return train_rows, val_rows


def take_limit(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if not limit or limit <= 0:
        return rows
    return rows[: min(limit, len(rows))]


def render_messages(messages: list[dict[str, Any]], tokenizer, reasoning_effort: str, add_generation_prompt: bool) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        reasoning_effort=reasoning_effort,
    )


def encode_example(
    example: dict[str, Any],
    tokenizer,
    max_seq_length: int,
    reasoning_effort: str,
    train_on_full_text: bool,
) -> dict[str, Any] | None:
    messages = example["messages"]
    full_text = render_messages(messages, tokenizer, reasoning_effort, add_generation_prompt=False)
    prompt_text = render_messages(messages[:-1], tokenizer, reasoning_effort, add_generation_prompt=True)

    full_enc = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_length,
    )
    prompt_enc = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_length,
    )

    input_ids = full_enc["input_ids"]
    attention_mask = full_enc["attention_mask"]
    if not input_ids:
        return None

    labels = list(input_ids)
    if not train_on_full_text:
        prefix_len = min(len(prompt_enc["input_ids"]), len(labels))
        for i in range(prefix_len):
            labels[i] = -100
        if all(value == -100 for value in labels):
            return None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "text": full_text,
    }


def preprocess_rows(
    rows: list[dict[str, Any]],
    tokenizer,
    max_seq_length: int,
    reasoning_effort: str,
    train_on_full_text: bool,
) -> tuple[list[dict[str, Any]], int]:
    processed: list[dict[str, Any]] = []
    dropped = 0
    for row in rows:
        encoded = encode_example(
            row,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            reasoning_effort=reasoning_effort,
            train_on_full_text=train_on_full_text,
        )
        if encoded is None:
            dropped += 1
        else:
            processed.append(encoded)
    if not processed:
        raise ValueError("All examples were dropped during preprocessing.")
    return processed, dropped


@dataclass
class SupervisedDataCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [-100] * pad_len)

        try:
            import torch
        except Exception as exc:
            raise SystemExit("torch is required for collation during training.") from exc

        return {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in batch.items()
        }


def write_run_manifest(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run-manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _module_matches_target(module_name: str, target: str) -> bool:
    if module_name == target:
        return True
    return module_name.endswith(f".{target}")


def inspect_target_module_coverage(model, target_modules: list[str]) -> dict[str, Any]:
    named_modules = [name for name, _ in model.named_modules()]
    per_target: dict[str, list[str]] = {}
    for target in target_modules:
        matches = [name for name in named_modules if _module_matches_target(name, target)]
        per_target[target] = matches
    total_matches = sum(len(matches) for matches in per_target.values())
    missing_targets = [target for target, matches in per_target.items() if not matches]
    return {
        "per_target_match_count": {target: len(matches) for target, matches in per_target.items()},
        "matched_module_examples": {target: matches[:8] for target, matches in per_target.items()},
        "total_matches": total_matches,
        "missing_targets": missing_targets,
    }


def count_trainable_parameters(model) -> dict[str, Any]:
    trainable = 0
    total = 0
    for param in model.parameters():
        count = int(param.numel())
        total += count
        if getattr(param, "requires_grad", False):
            trainable += count
    ratio = (trainable / total) if total else 0.0
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio": round(ratio, 8),
    }


def main() -> int:
    injected_site_packages = inject_local_site_packages()
    args = parse_args()
    config_path = Path(args.config).resolve()
    ensure_file(config_path, "training config")
    config = normalize_config(load_config(config_path))
    validate_config(config)
    resolved_model_ref = resolve_model_reference(config["model_name_or_path"])

    train_rows = load_jsonl(Path(config["dataset_path"]).resolve())
    validate_messages(train_rows, "train")
    if config["eval_dataset_path"]:
        eval_rows = load_jsonl(Path(config["eval_dataset_path"]).resolve())
        validate_messages(eval_rows, "eval")
    else:
        train_rows, eval_rows = split_rows(
            train_rows,
            float(config["val_split_ratio"]),
            int(config["seed"]),
        )

    train_rows = take_limit(train_rows, int(config["max_train_samples"]))
    eval_rows = take_limit(eval_rows, int(config["max_eval_samples"]))

    summary = {
        "config": str(config_path),
        "model_name_or_path": config["model_name_or_path"],
        "resolved_model_name_or_path": resolved_model_ref,
        "behavior_bundle_dir": config["behavior_bundle_dir"],
        "train_examples_raw": len(train_rows),
        "eval_examples_raw": len(eval_rows),
        "train_examples_after_preprocess": None,
        "eval_examples_after_preprocess": None,
        "dropped_train_examples": None,
        "dropped_eval_examples": None,
        "output_dir": config["output_dir"],
        "use_qlora": bool(config["use_qlora"]),
        "load_in_4bit": bool(config["load_in_4bit"]),
        "bf16": bool(config["bf16"]),
        "fp16": bool(config.get("fp16", False)),
        "target_modules": config["target_modules"],
        "target_parameters": config.get("target_parameters", []),
        "require_all_target_modules_hit": bool(config["require_all_target_modules_hit"]),
        "min_target_module_matches": int(config["min_target_module_matches"]),
        "min_trainable_params": int(config["min_trainable_params"]),
        "base_eval_cases_path": config["base_eval_cases_path"],
        "skill_eval_cases_path": config["skill_eval_cases_path"],
        "tokenizer_precheck": "pending",
        "dependency_status": detect_dependency_status(),
        "injected_site_packages": injected_site_packages,
        "first_train_render_preview": "",
        "lora_target_coverage": None,
        "trainable_parameter_report": None,
    }

    tokenizer = None
    try:
        from transformers import AutoTokenizer
    except Exception:
        if args.dry_run:
            summary["tokenizer_precheck"] = "unavailable"
            output_dir = Path(config["output_dir"]).resolve()
            write_run_manifest(output_dir, summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        raise SystemExit(
            "Full training dependencies are missing. Install requirements from requirements-train-dtf.txt."
        )

    tokenizer = AutoTokenizer.from_pretrained(resolved_model_ref)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    processed_train, dropped_train = preprocess_rows(
        train_rows,
        tokenizer=tokenizer,
        max_seq_length=int(config["max_seq_length"]),
        reasoning_effort=config["reasoning_effort"],
        train_on_full_text=bool(config["train_on_full_text"]),
    )
    processed_eval, dropped_eval = preprocess_rows(
        eval_rows,
        tokenizer=tokenizer,
        max_seq_length=int(config["max_seq_length"]),
        reasoning_effort=config["reasoning_effort"],
        train_on_full_text=bool(config["train_on_full_text"]),
    ) if eval_rows else ([], 0)

    summary["train_examples_after_preprocess"] = len(processed_train)
    summary["eval_examples_after_preprocess"] = len(processed_eval)
    summary["dropped_train_examples"] = dropped_train
    summary["dropped_eval_examples"] = dropped_eval
    summary["tokenizer_precheck"] = "ok"
    summary["first_train_render_preview"] = processed_train[0]["text"][:1400]

    output_dir = Path(config["output_dir"]).resolve()
    write_run_manifest(output_dir, summary)
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if not all(summary["dependency_status"].get(name, False) for name in ("torch", "datasets", "peft", "accelerate")):
        raise SystemExit(
            "Training dependencies are incomplete. Install requirements from requirements-train-dtf.txt "
            "into .venv-tools or your active environment."
        )

    try:
        import torch
        from datasets import Dataset
        from peft import (
            LoraConfig,
            PeftModel,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        from transformers import (
            AutoModelForCausalLM,
            BitsAndBytesConfig,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        raise SystemExit(
            "Full training dependencies are missing. Install requirements from requirements-train-dtf.txt."
        ) from exc

    train_dataset = Dataset.from_list(processed_train)
    eval_dataset = Dataset.from_list(processed_eval) if processed_eval else None
    has_cuda = torch.cuda.is_available()
    summary["execution_device"] = "cuda" if has_cuda else "cpu"

    quantization_config = None
    training_dtype = None
    if config["bf16"]:
        training_dtype = torch.bfloat16
    elif config.get("fp16", False):
        training_dtype = torch.float16
    else:
        training_dtype = torch.float32

    if config["use_qlora"] or config["load_in_4bit"]:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=training_dtype,
        )

    model_kwargs = {
        "torch_dtype": training_dtype,
        "use_cache": False,
        "attn_implementation": config["attn_implementation"],
    }
    if has_cuda or quantization_config is not None:
        model_kwargs["device_map"] = "auto"
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(
        resolved_model_ref,
        **model_kwargs,
    )
    coverage = inspect_target_module_coverage(model, list(config["target_modules"]))
    summary["lora_target_coverage"] = coverage
    if bool(config["require_all_target_modules_hit"]) and coverage["missing_targets"]:
        raise ValueError(
            "LoRA target_modules missing on loaded model: "
            + ", ".join(coverage["missing_targets"])
        )
    if int(coverage["total_matches"]) < int(config["min_target_module_matches"]):
        raise ValueError(
            "LoRA target_modules attached to too few modules: "
            f"{coverage['total_matches']} < {int(config['min_target_module_matches'])}"
        )
    model.config.use_cache = False
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(config["gradient_checkpointing"]),
        )
    elif config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules=config["target_modules"],
        target_parameters=config.get("target_parameters", []),
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    trainable_report = count_trainable_parameters(model)
    summary["trainable_parameter_report"] = trainable_report
    if int(trainable_report["trainable_params"]) < int(config["min_trainable_params"]):
        raise ValueError(
            "LoRA trainable parameter count is below minimum gate: "
            f"{trainable_report['trainable_params']} < {int(config['min_trainable_params'])}"
        )

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(config["num_train_epochs"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config.get("per_device_eval_batch_size", config["per_device_train_batch_size"])),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        warmup_ratio=float(config["warmup_ratio"]),
        logging_steps=int(config["logging_steps"]),
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        eval_steps=int(config["eval_steps"]) if eval_dataset is not None else None,
        save_steps=int(config["save_steps"]),
        save_total_limit=int(config["save_total_limit"]),
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        greater_is_better=False if eval_dataset is not None else None,
        max_grad_norm=float(config["max_grad_norm"]),
        lr_scheduler_type=config["lr_scheduler_type"],
        bf16=bool(config["bf16"]),
        fp16=bool(config.get("fp16", False)),
        seed=int(config["seed"]),
        gradient_checkpointing=bool(config["gradient_checkpointing"]),
        dataloader_num_workers=int(config["dataloader_num_workers"]),
        dataloader_pin_memory=has_cuda,
        report_to=config["report_to"],
        optim=config["optim"],
        remove_unused_columns=False,
        ddp_find_unused_parameters=bool(config["ddp_find_unused_parameters"]),
    )

    callbacks = []
    if eval_dataset is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(config["early_stopping_patience"])))

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SupervisedDataCollator(pad_token_id=tokenizer.pad_token_id),
        callbacks=callbacks,
    )

    resume_from_checkpoint = config["resume_from_checkpoint"] or None
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metrics = dict(train_result.metrics)
    metrics.update(trainable_report)
    metrics["lora_target_total_matches"] = coverage["total_matches"]
    metrics["lora_missing_targets"] = coverage["missing_targets"]
    if eval_dataset is not None:
        metrics.update(trainer.evaluate())
    write_run_manifest(
        output_dir,
        {
            **summary,
            "train_metrics": metrics,
        },
    )

    result = {
        "adapter_dir": str(output_dir),
        "merged_model_dir": None,
        "metrics": metrics,
    }

    if config["merge_after_train"]:
        base_model = AutoModelForCausalLM.from_pretrained(
            config["model_name_or_path"],
            torch_dtype=torch.bfloat16 if config["bf16"] else torch.float16,
            device_map="auto",
            attn_implementation=config["attn_implementation"],
        )
        peft_model = PeftModel.from_pretrained(base_model, str(output_dir))
        merged_model = peft_model.merge_and_unload()
        merged_dir = output_dir.with_name(output_dir.name + "-merged")
        merged_model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        result["merged_model_dir"] = str(merged_dir)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Runtime And Training

This skill is not runtime enforcement by itself. Use it as the Codex-side guide, then use the repo scripts to apply the behavior in inference or training.

## Runtime Path

Use these files when the goal is to steer a real DeepThinkingFlow runtime:

- `behavior/DeepThinkingFlow/system_prompt.txt`
- `behavior/DeepThinkingFlow/profile.json`
- `scripts/deepthinkingflow_cli.py`
- `scripts/bootstrap_transformers_deepthinkingflow.py`
- `scripts/chat_deepthinkingflow.py`
- `scripts/render_transformers_deepthinkingflow_prompt.py`
- `scripts/run_transformers_deepthinkingflow.py`

Useful entrypoints:

- `python scripts/deepthinkingflow_cli.py --help`
- `python scripts/deepthinkingflow_cli.py chat`
- `python scripts/deepthinkingflow_cli.py run --user "..." `

Key rule:

- By default, keep chain-of-thought hidden from end users.
- Only expose analysis when you explicitly opt in for debugging or internal inspection.

## Training Path

Use these files when the goal is to improve model behavior without changing the base architecture:

- `behavior/DeepThinkingFlow/training/harmony_sft_vi.jsonl`
- `behavior/DeepThinkingFlow/training/harmony_sft_vi.train.jsonl`
- `behavior/DeepThinkingFlow/training/harmony_sft_vi.eval.jsonl`
- `scripts/prepare_harmony_sft_dataset.py`
- `training/DeepThinkingFlow-lora/config.example.json`
- `training/DeepThinkingFlow-lora/config.qlora.example.json`
- `scripts/train_transformers_deepthinkingflow_lora.py`
- `scripts/evaluate_reasoning_outputs.py`

Recommended order:

1. Prepare and split the harmony dataset.
2. Prefer the fixed train/eval split files for before/after comparisons.
3. Run dry-run on the training config.
4. Train LoRA or QLoRA adapter.
5. Evaluate outputs against both the trait checklist and the richer rubric rules.
6. Only merge adapters after the adapter-only version is validated.

## Ruthless Rules

- Do not claim the skill itself fine-tunes the model.
- Do not claim the model weights were changed unless a real training run completed.
- Do not claim production readiness without:
  - real training logs
  - eval results
  - before/after comparison
  - a rollback path

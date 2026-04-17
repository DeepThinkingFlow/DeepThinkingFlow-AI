# Runtime And Training

This skill is not runtime enforcement by itself. Use it as the Codex-side guide, then use the repo scripts to apply the behavior in inference or training.

## Runtime Path

Use these files when the goal is to steer a real DeepThinkingFlow runtime:

- `behavior/DeepThinkingFlow/system_prompt.txt`
- `behavior/DeepThinkingFlow/profile.json`
- `scripts/deepthinkingflow_cli.py`
- `scripts/bootstrap_transformers_deepthinkingflow.py`
- `scripts/chat_deepthinkingflow.py`
- `scripts/inspect_safetensors_model.py`
- `scripts/render_transformers_deepthinkingflow_prompt.py`
- `scripts/run_transformers_deepthinkingflow.py`

Useful entrypoints:

- `python scripts/deepthinkingflow_cli.py --help`
- `python scripts/deepthinkingflow_cli.py chat`
- `python scripts/deepthinkingflow_cli.py run --user "..."`
- `python scripts/deepthinkingflow_cli.py inspect-weights --path original/model.safetensors --config runtime/transformers/DeepThinkingFlow/config.json`
- `python scripts/deepthinkingflow_cli.py prepare-training-assets`

Key rule:

- By default, keep chain-of-thought hidden from end users.
- End-user output should default to `final`.
- Any optional `analysis` surface should stay short, sanitized, and debug-only.

## Training Path

Use these files when the goal is to improve model behavior without changing the base architecture:

- `behavior/DeepThinkingFlow/training/harmony_sft_vi.jsonl`
- `behavior/DeepThinkingFlow/training/harmony_sft_vi.train.jsonl`
- `behavior/DeepThinkingFlow/training/harmony_sft_vi.eval.jsonl`
- `behavior/DeepThinkingFlow/training/harmony_sft_skill_compliance_vi.jsonl`
- `behavior/DeepThinkingFlow/training/harmony_sft_skill_compliance_vi.train.jsonl`
- `behavior/DeepThinkingFlow/training/harmony_sft_skill_compliance_vi.eval.jsonl`
- `behavior/DeepThinkingFlow/training/harmony_sft_plus_skill_compliance_vi.train.jsonl`
- `behavior/DeepThinkingFlow/training/harmony_sft_plus_skill_compliance_vi.eval.jsonl`
- `behavior/DeepThinkingFlow/evals/skill_compliance_following.jsonl`
- `scripts/prepare_deepthinkingflow_training_assets.py`
- `scripts/prepare_harmony_sft_dataset.py`
- `training/DeepThinkingFlow-lora/config.example.json`
- `training/DeepThinkingFlow-lora/config.qlora.example.json`
- `scripts/train_transformers_deepthinkingflow_lora.py`
- `scripts/evaluate_reasoning_outputs.py`

Recommended order:

1. Prepare and split the harmony dataset.
2. Keep the skill-compliance split fixed so train/eval comparisons stay honest.
3. Prefer the combined `harmony_sft_plus_skill_compliance_*` train/eval files for default LoRA or QLoRA runs.
4. Run dry-run on the training config.
5. Train LoRA or QLoRA adapter.
6. Evaluate outputs against both the base reasoning evals and `skill_compliance_following.jsonl`.
7. Only merge adapters after the adapter-only version is validated.

## Ruthless Rules

- Do not claim the skill itself fine-tunes the model.
- Do not claim a raw `model.safetensors` file contains `SKILL.md`, `system_prompt.txt`, `profile.json`, CLI behavior, or datasets.
- Do not claim the model weights were changed unless a real training run completed.
- Do not claim runtime prompting alone is learned behavior.
- Do not claim visible analysis equals full chain-of-thought.
- Do not claim production readiness without:
  - real training logs
  - eval results
  - before/after comparison
  - a rollback path

#!/usr/bin/env python3
"""Unified command launcher for DeepThinkingFlow project scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
VENV_TOOLS_PYTHON = ROOT_DIR / ".venv-tools" / "bin" / "python"

COMMANDS = {
    "chat": {
        "script": "chat_deepthinkingflow.py",
        "description": "Interactive terminal chat with multi-turn history.",
    },
    "run": {
        "script": "run_transformers_deepthinkingflow.py",
        "description": "One-shot generation with JSON output.",
    },
    "inspect-weights": {
        "script": "inspect_safetensors_model.py",
        "description": "Audit a local safetensors weight file without loading tensors into RAM.",
    },
    "render-prompt": {
        "script": "render_transformers_deepthinkingflow_prompt.py",
        "description": "Render the injected chat-template prompt.",
    },
    "compose-request": {
        "script": "compose_behavior_request.py",
        "description": "Compose system/user messages from the behavior bundle.",
    },
    "validate-bundle": {
        "script": "validate_behavior_bundle.py",
        "description": "Validate profile, datasets, and eval bundle health.",
    },
    "bootstrap": {
        "script": "bootstrap_transformers_deepthinkingflow.py",
        "description": "Bootstrap a local Transformers model directory.",
    },
    "assemble-model-dir": {
        "script": "assemble_local_transformers_model_dir.py",
        "description": "Assemble metadata and local weights into a model dir.",
    },
    "prepare-sft": {
        "script": "prepare_harmony_sft_dataset.py",
        "description": "Split and prepare the harmony SFT dataset.",
    },
    "prepare-training-assets": {
        "script": "prepare_deepthinkingflow_training_assets.py",
        "description": "Build deterministic base, skill-compliance, and combined train/eval assets.",
    },
    "compile-bundle": {
        "script": "compile_behavior_bundle.py",
        "description": "Compile the behavior bundle into a compact runtime prompt pack.",
    },
    "bootstrap-training-env": {
        "script": "bootstrap_training_env.py",
        "description": "Install DeepThinkingFlow training dependencies into .venv-tools.",
    },
    "preflight-train": {
        "script": "preflight_deepthinkingflow_training.py",
        "description": "Estimate whether a training config is feasible on the current machine.",
    },
    "generate-skill-compliance": {
        "script": "generate_skill_compliance_corpus.py",
        "description": "Regenerate the expanded skill-compliance dataset and eval corpus.",
    },
    "train-lora": {
        "script": "train_transformers_deepthinkingflow_lora.py",
        "description": "Launch or dry-run the LoRA/QLoRA training pipeline.",
    },
    "eval": {
        "script": "evaluate_reasoning_outputs.py",
        "description": "Score outputs against the reasoning eval rubric.",
    },
    "report-artifacts": {
        "script": "report_deepthinkingflow_artifacts.py",
        "description": "Hash base weights, adapter outputs, eval files, and classify claim level.",
    },
}

VENV_PREFERRED_COMMANDS = {"bootstrap-training-env", "train-lora", "eval", "preflight-train"}


def print_help() -> None:
    print("DeepThinkingFlow CLI")
    print()
    print("Usage:")
    print("  python scripts/deepthinkingflow_cli.py <command> [command args]")
    print("  python scripts/deepthinkingflow_cli.py help <command>")
    print()
    print("Commands:")
    for name, meta in COMMANDS.items():
        print(f"  {name:<18} {meta['description']}")
    print()
    print("Examples:")
    print("  python scripts/deepthinkingflow_cli.py chat")
    print('  python scripts/deepthinkingflow_cli.py run --user "Phan tich prompt nay"')
    print("  python scripts/deepthinkingflow_cli.py inspect-weights --path original/model.safetensors")
    print("  python scripts/deepthinkingflow_cli.py prepare-training-assets")
    print("  python scripts/deepthinkingflow_cli.py generate-skill-compliance")
    print("  python scripts/deepthinkingflow_cli.py compile-bundle")
    print("  python scripts/deepthinkingflow_cli.py preflight-train --config training/DeepThinkingFlow-lora/config.example.json")
    print("  python scripts/deepthinkingflow_cli.py help train-lora")


def dispatch(command: str, forwarded_args: list[str]) -> int:
    script_path = SCRIPTS_DIR / COMMANDS[command]["script"]
    python_executable = str(VENV_TOOLS_PYTHON) if command in VENV_PREFERRED_COMMANDS and VENV_TOOLS_PYTHON.is_file() else sys.executable
    completed = subprocess.run(
        [python_executable, str(script_path), *forwarded_args],
        cwd=str(ROOT_DIR),
        check=False,
    )
    return completed.returncode


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] in {"-h", "--help"}:
        print_help()
        return 0

    if args[0] == "help":
        if len(args) == 1:
            print_help()
            return 0
        command = args[1]
        if command not in COMMANDS:
            print(f"Unknown command: {command}", file=sys.stderr)
            print_help()
            return 2
        return dispatch(command, ["--help"])

    command = args[0]
    if command not in COMMANDS:
        print(f"Unknown command: {command}", file=sys.stderr)
        print_help()
        return 2
    return dispatch(command, args[1:])


if __name__ == "__main__":
    raise SystemExit(main())

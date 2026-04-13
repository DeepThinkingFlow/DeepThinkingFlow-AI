#!/usr/bin/env python3
"""Unified command launcher for DeepThinkingFlow project scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent

COMMANDS = {
    "chat": {
        "script": "chat_deepthinkingflow.py",
        "description": "Interactive terminal chat with multi-turn history.",
    },
    "run": {
        "script": "run_transformers_deepthinkingflow.py",
        "description": "One-shot generation with JSON output.",
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
    "train-lora": {
        "script": "train_transformers_deepthinkingflow_lora.py",
        "description": "Launch or dry-run the LoRA/QLoRA training pipeline.",
    },
    "eval": {
        "script": "evaluate_reasoning_outputs.py",
        "description": "Score outputs against the reasoning eval rubric.",
    },
}


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
    print("  python scripts/deepthinkingflow_cli.py help train-lora")


def dispatch(command: str, forwarded_args: list[str]) -> int:
    script_path = SCRIPTS_DIR / COMMANDS[command]["script"]
    completed = subprocess.run(
        [sys.executable, str(script_path), *forwarded_args],
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

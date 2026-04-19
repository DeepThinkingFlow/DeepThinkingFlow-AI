#!/usr/bin/env python3
"""Unified command launcher for DeepThinkingFlow project scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from deepthinkingflow_exit_codes import OK, USAGE_ERROR

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
    "export-runtime": {
        "script": "export_external_runtime_assets.py",
        "description": "Export runtime-only assets for external hosts such as Ollama and Claude Code.",
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
    "prepare-datasets": {
        "script": "prepare_external_datasets.py",
        "description": "Prepare external reasoning/coding datasets into chat-templated HF datasets.",
    },
    "export-chat-jsonl": {
        "script": "export_prepared_chat_jsonl.py",
        "description": "Export a prepared HF chat dataset directory into JSONL rows with messages.",
    },
    "build-external-train-bundle": {
        "script": "build_external_training_bundle.py",
        "description": "Build train/eval JSONL assets from prepared external chat datasets.",
    },
    "prepare-training-assets": {
        "script": "prepare_deepthinkingflow_training_assets.py",
        "description": "Build deterministic base, skill-compliance, and combined train/eval assets.",
    },
    "build-partial-lora-config": {
        "script": "build_partial_lora_config.py",
        "description": "Derive a safer partial-scope LoRA config for incremental training without mutating the base config.",
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
    "preflight-all": {
        "script": "preflight_deepthinkingflow_project.py",
        "description": "Run a consolidated project preflight across bundle, runtime, training, and external hosts.",
    },
    "doctor": {
        "script": "doctor_deepthinkingflow.py",
        "description": "Run a release-style health report across verify, claim gates, host readiness, and artifacts.",
    },
    "verify": {
        "script": "verify_deepthinkingflow_project.py",
        "description": "Run a consolidated verification suite across bundle validation, preflight, and smoke tests.",
    },
    "tiny-smoke-release": {
        "script": "run_tiny_smoke_release_lane.py",
        "description": "Run a real tiny-model smoke training lane and emit artifact, verify, and release reports.",
    },
    "system-check": {
        "script": "deepthinkingflow_system_check.py",
        "description": "Check minimum RAM, CPU, GPU, and VRAM guidance without blocking startup.",
    },
    "generate-skill-compliance": {
        "script": "generate_skill_compliance_corpus.py",
        "description": "Regenerate the expanded skill-compliance dataset and eval corpus.",
    },
    "train-lora": {
        "script": "train_transformers_deepthinkingflow_lora.py",
        "description": "Launch or dry-run the LoRA/QLoRA training pipeline.",
    },
    "train-lora-staged": {
        "script": "train_deepthinkingflow_staged.py",
        "description": "Run staged LoRA training with progressive sample growth and checkpoint resume.",
    },
    "eval": {
        "script": "evaluate_reasoning_outputs.py",
        "description": "Score outputs against the reasoning eval rubric.",
    },
    "generate-eval-predictions": {
        "script": "generate_eval_predictions.py",
        "description": "Generate predictions JSONL from a base model or a base+adapter pair for eval cases.",
    },
    "compare-eval-reports": {
        "script": "compare_eval_reports.py",
        "description": "Compare baseline and candidate eval summary JSON reports.",
    },
    "report-artifacts": {
        "script": "report_deepthinkingflow_artifacts.py",
        "description": "Hash base weights, adapter outputs, eval files, and classify claim level.",
    },
    "benchmark-runtime": {
        "script": "benchmark_deepthinkingflow_runtime.py",
        "description": "Measure prompt rendering and tokenizer throughput for the runtime target.",
    },
    "aggregate-runs": {
        "script": "aggregate_deepthinkingflow_runs.py",
        "description": "Aggregate artifact, verify, and release reports into one lineage view.",
    },
    "check-promotion-readiness": {
        "script": "check_promotion_readiness.py",
        "description": "Check whether current evidence satisfies the promotion policy for a claim level.",
    },
    "release-manifest": {
        "script": "build_release_manifest.py",
        "description": "Build a release-oriented manifest from verify and artifact reports.",
    },
    "cuda-backend-status": {
        "script": "cuda_backend_status.py",
        "description": "Report CUDA backend scaffold/build readiness for the NVIDIA path.",
    },
    "apple-backend-status": {
        "script": "apple_backend_status.py",
        "description": "Report Apple Silicon backend scaffold/build readiness for the Metal/MLX path.",
    },
    "apple-mlx-status": {
        "script": "apple_mlx_adapter_status.py",
        "description": "Report MLX-first inference adapter readiness for the Apple Silicon path.",
    },
    "apple-mlx-weight-check": {
        "script": "apple_mlx_weight_loader_check.py",
        "description": "Load DeepThinkingFlow weights through MLX and verify first-block shapes.",
    },
    "apple-mlx-attn-shapes": {
        "script": "apple_mlx_attention_shape_check.py",
        "description": "Dry-run RMSNorm + QKV split + GQA attention shapes for DeepThinkingFlow.",
    },
    "apple-mlx-mlp-keys": {
        "script": "apple_mlx_mlp_key_dump.py",
        "description": "Dump the real block.N.mlp.* tensor keys from the safetensors header.",
    },
    "apple-mlx-moe-metadata": {
        "script": "apple_mlx_moe_metadata_check.py",
        "description": "Inspect quantized MoE/FFN metadata and router shapes for one DeepThinkingFlow block.",
    },
    "apple-mlx-dequant-range": {
        "script": "apple_mlx_dequant_range_check.py",
        "description": "Unpack one quantized expert projection and report provisional dequant ranges.",
    },
    "apple-mlx-moe-forward": {
        "script": "apple_mlx_moe_forward_check.py",
        "description": "Run a provisional MoE forward with dequant-on-the-fly and report output ranges.",
    },
    "apple-mlx-kv-cache": {
        "script": "apple_mlx_kv_cache_shape_check.py",
        "description": "Dry-run alternating attention layer-types and KV-cache trim behavior.",
    },
    "apple-mlx-inference-status": {
        "script": "apple_mlx_inference_scaffold_status.py",
        "description": "Report tokenizer, embedding/lm_head keys, and inference-loop scaffold readiness.",
    },
}

VENV_PREFERRED_COMMANDS = {
    "bootstrap-training-env",
    "train-lora",
    "train-lora-staged",
    "eval",
    "preflight-train",
    "benchmark-runtime",
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
    print("  python scripts/deepthinkingflow_cli.py inspect-weights --path original/model.safetensors")
    print("  python scripts/deepthinkingflow_cli.py export-runtime --target ollama --ollama-model llama3.1:8b")
    print("  python scripts/deepthinkingflow_cli.py prepare-datasets --num_proc 1 --ot3_limit 100 --oci_limit 100")
    print("  python scripts/deepthinkingflow_cli.py export-chat-jsonl --input-dir data/openthoughts3_processed --output-jsonl data/openthoughts3_processed.jsonl")
    print("  python scripts/deepthinkingflow_cli.py build-external-train-bundle --input-jsonl data/openthoughts3_processed.jsonl --input-jsonl data/opencodeinstruct_processed.jsonl --train-output data/external-train.jsonl --eval-output data/external-eval.jsonl")
    print("  python scripts/deepthinkingflow_cli.py build-partial-lora-config --output out/partial-lora-config.json")
    print("  python scripts/deepthinkingflow_cli.py prepare-training-assets")
    print("  python scripts/deepthinkingflow_cli.py generate-skill-compliance")
    print("  python scripts/deepthinkingflow_cli.py compile-bundle")
    print("  python scripts/deepthinkingflow_cli.py preflight-train --config training/DeepThinkingFlow-lora/config.example.json")
    print("  python scripts/deepthinkingflow_cli.py preflight-all")
    print("  python scripts/deepthinkingflow_cli.py doctor")
    print("  python scripts/deepthinkingflow_cli.py verify")
    print("  python scripts/deepthinkingflow_cli.py benchmark-runtime --iterations 5")
    print("  python scripts/deepthinkingflow_cli.py aggregate-runs --search-root out")
    print("  python scripts/deepthinkingflow_cli.py check-promotion-readiness --claim-level runtime-only --verify-report /tmp/dtf-verify.json")
    print("  python scripts/deepthinkingflow_cli.py cuda-backend-status --cuda-arch 89")
    print("  python scripts/deepthinkingflow_cli.py apple-backend-status")
    print("  python scripts/deepthinkingflow_cli.py apple-mlx-status --quantize-4bit")
    print("  python scripts/deepthinkingflow_cli.py apple-mlx-weight-check")
    print("  python scripts/deepthinkingflow_cli.py apple-mlx-attn-shapes --seq-len 16")
    print("  python scripts/deepthinkingflow_cli.py apple-mlx-mlp-keys --layer-index 0")
    print("  python scripts/deepthinkingflow_cli.py apple-mlx-moe-metadata --layer-index 0")
    print("  python scripts/deepthinkingflow_cli.py apple-mlx-dequant-range --projection mlp1 --expert-index 0")
    print("  python scripts/deepthinkingflow_cli.py apple-mlx-moe-forward --activation both --seq-len 8")
    print("  python scripts/deepthinkingflow_cli.py apple-mlx-kv-cache --layer-index 0 --seq-len 1 --cached-seq-len 256")
    print("  python scripts/deepthinkingflow_cli.py apple-mlx-inference-status")
    print("  python scripts/deepthinkingflow_cli.py tiny-smoke-release")
    print("  python scripts/deepthinkingflow_cli.py release-manifest --output out/release-manifest.json")
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
            return USAGE_ERROR
        return dispatch(command, ["--help"])

    command = args[0]
    if command not in COMMANDS:
        print(f"Unknown command: {command}", file=sys.stderr)
        print_help()
        return USAGE_ERROR
    return dispatch(command, args[1:])


if __name__ == "__main__":
    raise SystemExit(main())

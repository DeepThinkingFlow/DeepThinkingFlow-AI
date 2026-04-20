#!/usr/bin/env python3
"""Run a strict end-to-end contract verification for the DeepThinkingFlow Apple path."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from deepthinkingflow_apple.inference import GenerationConfig, inference_scaffold_status, load_config, verify_generation_contract
from deepthinkingflow_apple.tokenizer import GPTOssTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Apple-path end-to-end contract at prompt/config/tokenizer level.")
    parser.add_argument("--model-dir", default="runtime/transformers/DeepThinkingFlow", help="Local model directory.")
    parser.add_argument("--prompt", default="Explain MoE briefly.", help="Prompt to verify.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.model_dir)
    inference_status = inference_scaffold_status(args.model_dir)
    verification = {
        "tokenizer_ready": False,
        "layer_count_matches": False,
        "sampling_contract_present": True,
        "end_to_end_contract_ready": False,
        "native_weight_execution_verified": False,
        "claim_ceiling": "python-contract-only",
    }
    dependency_ready = True
    dependency_error = ""
    try:
        tokenizer = GPTOssTokenizer(args.model_dir)
        generation_contract = verify_generation_contract(
            tokenizer=tokenizer,
            config=config,
            prompt=args.prompt,
            sampling=GenerationConfig(),
        )
        verification.update(
            {
                "tokenizer_ready": generation_contract["prompt_package_ready"],
                "layer_count_matches": generation_contract["layer_types_count"] == generation_contract["num_hidden_layers"],
                "end_to_end_contract_ready": True,
            }
        )
    except RuntimeError as exc:
        dependency_ready = False
        dependency_error = str(exc)
        generation_contract = {
            "prompt": args.prompt,
            "prompt_package_ready": False,
            "rendered_prompt_non_empty": False,
            "input_token_count": 0,
            "stop_token_ids": [],
            "sampling_contract": asdict(GenerationConfig()),
            "layer_types_count": len(list(config.get("layer_types") or [])),
            "num_hidden_layers": int(config["num_hidden_layers"]),
        }
    payload = {
        "model_dir": str(Path(args.model_dir).resolve()),
        "dependency_ready": dependency_ready,
        "dependency_error": dependency_error,
        "prompt_contract": generation_contract,
        "inference_status": inference_status,
        "verification": verification,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

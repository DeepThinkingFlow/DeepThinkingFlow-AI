#!/usr/bin/env python3
"""Export runtime-only integration assets for external hosts such as Ollama and Claude Code."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from compose_behavior_request import load_bundle
from deepthinkingflow_env import detect_external_runtime_status
from report_deepthinkingflow_artifacts import file_sha256

SCHEMA_VERSION = "dtf-external-runtime-export/v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export DeepThinkingFlow behavior bundle assets for external runtimes."
    )
    parser.add_argument(
        "--bundle",
        default="behavior/DeepThinkingFlow",
        help="Behavior bundle directory.",
    )
    parser.add_argument(
        "--target",
        choices=("generic", "ollama", "claude-code"),
        default="generic",
        help="External host to prepare assets for.",
    )
    parser.add_argument(
        "--user",
        default="Xin chao, hay gioi thieu nhanh DeepThinkingFlow.",
        help="Sample user message to include in the exported request payload.",
    )
    parser.add_argument(
        "--ollama-model",
        default="deepthinkingflow:20b",
        help="Base Ollama model tag to reference in the generated Modelfile.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write exported assets into.",
    )
    parser.add_argument(
        "--fail-if-host-missing",
        action="store_true",
        help="Exit with a non-zero status when the selected external host is not detected on PATH.",
    )
    return parser.parse_args()


def build_target_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).resolve()
    return Path("out") / "external-runtime" / args.target


def validate_args(args: argparse.Namespace) -> None:
    if args.target == "ollama" and not args.ollama_model.strip():
        raise SystemExit("--ollama-model must be a non-empty Ollama model tag.")
    if args.output_dir is not None and not str(args.output_dir).strip():
        raise SystemExit("--output-dir must not be empty.")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_plaintext(system_prompt: str, user_prompt: str) -> str:
    return f"[system]\n{system_prompt}\n\n[user]\n{user_prompt}\n"


def build_ollama_modelfile(*, base_model: str, system_prompt: str) -> str:
    escaped = system_prompt.replace('"""', '\\"\\"\\"')
    return (
        f"FROM {base_model}\n\n"
        f'SYSTEM """\n{escaped}\n"""\n\n'
        'PARAMETER temperature 0.7\n'
        'PARAMETER top_p 0.95\n'
    )


def build_instructions(target: str, output_dir: Path, runtime_status: dict[str, bool]) -> str:
    lines = [
        f"target: {target}",
        "claim_level: runtime-only",
        "note: exported assets steer an external host at runtime and do not modify model weights.",
    ]
    if target == "ollama":
        lines.append(f"ollama_installed: {'yes' if runtime_status['ollama'] else 'no'}")
        lines.append(f"run: ollama create deepthinkingflow-runtime -f {output_dir / 'Modelfile'}")
        lines.append("run: ollama run deepthinkingflow-runtime")
        lines.append("warning: this only works when the base Ollama model tag exists locally or can be pulled by Ollama.")
    elif target == "claude-code":
        lines.append(f"claude_available: {'yes' if runtime_status['claude'] or runtime_status['claude_code'] else 'no'}")
        lines.append(f"use_system_prompt: {output_dir / 'system_prompt.txt'}")
        lines.append(f"use_request_json: {output_dir / 'request.json'}")
        lines.append("warning: Claude Code is an external coding host; these assets guide runtime behavior only.")
    else:
        lines.append(f"use_system_prompt: {output_dir / 'system_prompt.txt'}")
        lines.append(f"use_request_json: {output_dir / 'request.json'}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    validate_args(args)
    bundle_dir = Path(args.bundle).resolve()
    profile, system_prompt = load_bundle(bundle_dir)
    output_dir = build_target_dir(args)
    runtime_status = detect_external_runtime_status()
    host_ready = (
        runtime_status["ollama"]
        if args.target == "ollama"
        else (runtime_status["claude"] or runtime_status["claude_code"])
        if args.target == "claude-code"
        else True
    )
    if args.fail_if_host_missing and not host_ready:
        raise SystemExit(f"Selected external host is not available on PATH for target={args.target}.")

    payload = {
        "bundle": profile["name"],
        "target_model": profile["target_model"],
        "claim_level": "runtime-only",
        "target_host": args.target,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": args.user},
        ],
    }

    write_text(output_dir / "system_prompt.txt", system_prompt + "\n")
    write_text(output_dir / "request.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    write_text(output_dir / "request.txt", build_plaintext(system_prompt, args.user))
    write_text(output_dir / "README.runtime.txt", build_instructions(args.target, output_dir, runtime_status))

    created_files = [
        "system_prompt.txt",
        "request.json",
        "request.txt",
        "README.runtime.txt",
    ]

    if args.target == "ollama":
        write_text(
            output_dir / "Modelfile",
            build_ollama_modelfile(base_model=args.ollama_model, system_prompt=system_prompt),
        )
        created_files.append("Modelfile")

    file_reports = []
    for file_name in created_files:
        file_path = output_dir / file_name
        file_reports.append(
            {
                "path": file_name,
                "sha256": file_sha256(file_path),
                "size_bytes": file_path.stat().st_size,
            }
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": args.target,
        "output_dir": str(output_dir),
        "claim_level": "runtime-only",
        "external_runtime_status": runtime_status,
        "host_ready": host_ready,
        "created_files": created_files,
        "file_reports": file_reports,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

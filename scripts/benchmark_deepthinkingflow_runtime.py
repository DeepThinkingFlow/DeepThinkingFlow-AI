#!/usr/bin/env python3
"""Run a lightweight DeepThinkingFlow runtime benchmark without requiring full model generation."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from deepthinkingflow_json_io import now_utc_iso, write_json_file

ROOT_DIR = Path(__file__).resolve().parents[1]
VENV_TOOLS_PYTHON = ROOT_DIR / ".venv-tools" / "bin" / "python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark prompt assembly and tokenizer throughput for the DeepThinkingFlow runtime target."
    )
    parser.add_argument(
        "--model-dir",
        default="runtime/transformers/DeepThinkingFlow",
        help="Transformers model directory used to load the tokenizer.",
    )
    parser.add_argument(
        "--prompt",
        default="Hãy phân tích ngắn gọn xem release gate này còn thiếu gì.",
        help="Prompt used for the benchmark loop.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="How many timed iterations to run.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="How many warmup iterations to run before timing.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def build_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are DeepThinkingFlow. Keep visible reasoning compact, useful, and free of internal markers.",
        },
        {"role": "user", "content": prompt},
    ]


def maybe_reexec_into_venv_tools() -> None:
    if os.environ.get("DTF_ALREADY_REEXECED") == "1":
        return
    if not VENV_TOOLS_PYTHON.is_file():
        return
    completed = subprocess.run(
        [
            str(VENV_TOOLS_PYTHON),
            "-c",
            "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('transformers') else 1)",
        ],
        cwd=str(ROOT_DIR),
        check=False,
    )
    if completed.returncode != 0:
        return
    env = dict(os.environ)
    env["DTF_ALREADY_REEXECED"] = "1"
    os.execve(str(VENV_TOOLS_PYTHON), [str(VENV_TOOLS_PYTHON), __file__, *sys.argv[1:]], env)


def summarize_durations(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": round(statistics.fmean(values) * 1000.0, 3),
        "median_ms": round(statistics.median(values) * 1000.0, 3),
        "p95_ms": round(percentile(values, 0.95) * 1000.0, 3),
        "min_ms": round(min(values) * 1000.0, 3),
        "max_ms": round(max(values) * 1000.0, 3),
    }


def main() -> int:
    maybe_reexec_into_venv_tools()
    args = parse_args()
    if args.iterations < 1:
        raise SystemExit("--iterations must be >= 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")

    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        summary = {
            "schema_version": "dtf-runtime-benchmark/v1",
            "generated_at_utc": now_utc_iso(),
            "model_dir": str(Path(args.model_dir).resolve()),
            "python_executable": sys.executable,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "available": False,
            "error": (
                "transformers is required for benchmark_deepthinkingflow_runtime.py. "
                "Install runtime dependencies before running this benchmark."
            ),
        }
        if args.output:
            write_json_file(Path(args.output).resolve(), summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    model_dir = Path(args.model_dir).resolve()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    messages = build_messages(args.prompt)

    for _ in range(args.warmup):
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokenizer(rendered, add_special_tokens=False)

    render_times: list[float] = []
    tokenize_times: list[float] = []
    token_counts: list[int] = []
    rendered_chars: list[int] = []

    for _ in range(args.iterations):
        start_render = time.perf_counter()
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        end_render = time.perf_counter()

        start_tokenize = time.perf_counter()
        token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        end_tokenize = time.perf_counter()

        render_times.append(end_render - start_render)
        tokenize_times.append(end_tokenize - start_tokenize)
        token_counts.append(len(token_ids))
        rendered_chars.append(len(rendered))

    total_tokenize_seconds = max(sum(tokenize_times), sys.float_info.epsilon)
    tokenization_tokens_per_second = round(sum(token_counts) / total_tokenize_seconds, 3)

    summary: dict[str, Any] = {
        "schema_version": "dtf-runtime-benchmark/v1",
        "generated_at_utc": now_utc_iso(),
        "model_dir": str(model_dir),
        "python_executable": sys.executable,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "prompt_chars": len(args.prompt),
        "rendered_prompt_chars": {
            "min": min(rendered_chars),
            "max": max(rendered_chars),
            "mean": round(statistics.fmean(rendered_chars), 3),
        },
        "token_counts": {
            "min": min(token_counts),
            "max": max(token_counts),
            "mean": round(statistics.fmean(token_counts), 3),
        },
        "render_template_latency": summarize_durations(render_times),
        "tokenize_latency": summarize_durations(tokenize_times),
        "tokenization_tokens_per_second": tokenization_tokens_per_second,
        "notes": [
            "This benchmark measures prompt rendering and tokenization only.",
            "It does not claim end-to-end generation latency for the base weights.",
        ],
    }

    if args.output:
        write_json_file(Path(args.output).resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

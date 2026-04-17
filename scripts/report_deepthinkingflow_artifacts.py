#!/usr/bin/env python3
"""Hash DeepThinkingFlow artifacts and classify the strongest supportable claim level."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report hashes for base weights, adapter artifacts, eval outputs, and claim level."
    )
    parser.add_argument("--base-weights", default="", help="Base weight file or directory.")
    parser.add_argument("--adapter-dir", default="", help="LoRA or QLoRA adapter directory.")
    parser.add_argument("--eval-output", default="", help="Eval output JSON/JSONL file.")
    parser.add_argument("--compare-report", default="", help="Optional before/after compare report.")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def collect_path_report(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {
            "type": "file",
            "path": str(path),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    if path.is_dir():
        files = sorted(item for item in path.rglob("*") if item.is_file())
        digest = hashlib.sha256()
        file_reports = []
        for file_path in files:
            rel_path = str(file_path.relative_to(path))
            sha = file_sha256(file_path)
            digest.update(rel_path.encode("utf-8"))
            digest.update(sha.encode("utf-8"))
            file_reports.append(
                {
                    "path": rel_path,
                    "sha256": sha,
                    "size_bytes": file_path.stat().st_size,
                }
            )
        return {
            "type": "directory",
            "path": str(path),
            "sha256": digest.hexdigest(),
            "file_count": len(file_reports),
            "files": file_reports,
        }
    raise ValueError(f"Artifact path does not exist: {path}")


def detect_claim_level(base_weights: dict[str, Any] | None, adapter_dir: dict[str, Any] | None, eval_output: dict[str, Any] | None, compare_report: dict[str, Any] | None) -> str:
    if base_weights and adapter_dir and eval_output and compare_report:
        return "learned-only-after-training"
    if base_weights and eval_output and compare_report and not adapter_dir:
        return "weight-level-verified"
    if adapter_dir or eval_output:
        return "training-ready"
    return "runtime-only"


def maybe_collect(raw_path: str) -> dict[str, Any] | None:
    if not raw_path:
        return None
    return collect_path_report(Path(raw_path).resolve())


def main() -> int:
    args = parse_args()
    base_weights = maybe_collect(args.base_weights)
    adapter_dir = maybe_collect(args.adapter_dir)
    eval_output = maybe_collect(args.eval_output)
    compare_report = maybe_collect(args.compare_report)

    summary = {
        "base_weights": base_weights,
        "adapter_dir": adapter_dir,
        "eval_output": eval_output,
        "compare_report": compare_report,
        "claim_level": detect_claim_level(base_weights, adapter_dir, eval_output, compare_report),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

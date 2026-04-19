#!/usr/bin/env python3
"""Shared JSON, hashing, and subprocess helpers for DeepThinkingFlow scripts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json_file(path: Path, label: str = "json file") -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Missing {label}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl_file(path: Path, label: str = "jsonl file") -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"Missing {label}: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_json_command(command: list[str], *, cwd: Path, label: str) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"{label} failed with code {completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    return json.loads(completed.stdout)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

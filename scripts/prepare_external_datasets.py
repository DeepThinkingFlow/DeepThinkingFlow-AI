#!/usr/bin/env python3
"""Wrapper entrypoint for the root-level prepare_datasets.py script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT_DIR / "prepare_datasets.py"


def main() -> int:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *sys.argv[1:]],
        cwd=str(ROOT_DIR),
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Bootstrap the DeepThinkingFlow training environment inside .venv-tools."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT_DIR / ".venv-tools" / "bin" / "python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install DeepThinkingFlow training dependencies into .venv-tools."
    )
    parser.add_argument(
        "--requirements",
        default="requirements-train-dtf.txt",
        help="Requirements file to install.",
    )
    parser.add_argument(
        "--upgrade-pip",
        action="store_true",
        help="Upgrade pip before installing dependencies.",
    )
    parser.add_argument(
        "--force-cpu-torch",
        action="store_true",
        help="Install CPU-only PyTorch wheels even if GPU tooling is present.",
    )
    return parser.parse_args()


def has_nvidia_gpu() -> bool:
    completed = subprocess.run(
        ["bash", "-lc", "command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1"],
        cwd=str(ROOT_DIR),
        check=False,
    )
    return completed.returncode == 0


def split_requirements(requirements_path: Path) -> tuple[list[str], list[str]]:
    torch_lines: list[str] = []
    other_lines: list[str] = []
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            if stripped:
                other_lines.append(line)
            continue
        if stripped.startswith("torch"):
            torch_lines.append(stripped)
        else:
            other_lines.append(line)
    return torch_lines, other_lines


def main() -> int:
    args = parse_args()
    requirements_path = (ROOT_DIR / args.requirements).resolve()
    if not VENV_PYTHON.is_file():
        raise SystemExit(f"Missing venv python: {VENV_PYTHON}")
    if not requirements_path.is_file():
        raise SystemExit(f"Missing requirements file: {requirements_path}")

    commands = []
    if args.upgrade_pip:
        commands.append([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"])
    torch_lines, other_lines = split_requirements(requirements_path)
    use_cpu_torch = args.force_cpu_torch or not has_nvidia_gpu()
    if torch_lines:
        torch_spec = torch_lines[0]
        torch_cmd = [str(VENV_PYTHON), "-m", "pip", "install"]
        if use_cpu_torch:
            torch_cmd.extend(["--index-url", "https://download.pytorch.org/whl/cpu"])
        torch_cmd.append(torch_spec)
        commands.append(torch_cmd)
    if other_lines:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as tmp:
            tmp.write("\n".join(other_lines) + "\n")
            other_requirements_path = tmp.name
        commands.append([str(VENV_PYTHON), "-m", "pip", "install", "-r", other_requirements_path])

    for cmd in commands:
        completed = subprocess.run(cmd, cwd=str(ROOT_DIR), check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

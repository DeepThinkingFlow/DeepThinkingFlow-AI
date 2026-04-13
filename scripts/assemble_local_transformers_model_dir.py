#!/usr/bin/env python3
"""Assemble a DeepThinkingFlow Transformers model directory by overlaying local weights onto metadata files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay local DeepThinkingFlow weights onto a Transformers metadata directory."
    )
    parser.add_argument(
        "--metadata-dir",
        default="runtime/transformers/DeepThinkingFlow",
        help="Directory containing tokenizer/chat template/config metadata.",
    )
    parser.add_argument(
        "--weights-dir",
        default="original",
        help="Directory containing local weight files such as model.safetensors.",
    )
    parser.add_argument(
        "--link-name",
        default="model.safetensors",
        help="Weight filename to link into the metadata directory.",
    )
    parser.add_argument(
        "--extra-files",
        nargs="*",
        default=["dtypes.json"],
        help="Extra files from weights-dir to symlink if present.",
    )
    return parser.parse_args()


def ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"Missing {label}: {path}")


def safe_symlink(src: Path, dest: Path) -> None:
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink() and dest.resolve() == src.resolve():
            return
        raise SystemExit(f"Refusing to overwrite existing file: {dest}")
    dest.symlink_to(src)


def main() -> int:
    args = parse_args()
    metadata_dir = Path(args.metadata_dir).resolve()
    weights_dir = Path(args.weights_dir).resolve()
    ensure_file(metadata_dir / "config.json", "metadata config")
    ensure_file(metadata_dir / "tokenizer.json", "metadata tokenizer")

    weight_file = weights_dir / args.link_name
    ensure_file(weight_file, "local model weight")
    safe_symlink(weight_file, metadata_dir / args.link_name)

    linked_files = [args.link_name]
    for name in args.extra_files:
        candidate = weights_dir / name
        if candidate.is_file():
            safe_symlink(candidate, metadata_dir / name)
            linked_files.append(name)

    manifest = {
        "metadata_dir": str(metadata_dir),
        "weights_dir": str(weights_dir),
        "linked_files": linked_files,
        "note": "This directory is now suitable for real from_pretrained() loads, assuming dependencies and hardware are available.",
    }
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

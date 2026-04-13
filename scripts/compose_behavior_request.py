#!/usr/bin/env python3
"""Compose a chat request from a behavior bundle and a user prompt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"Missing {label}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose a messages payload from a behavior bundle."
    )
    parser.add_argument(
        "--bundle",
        required=True,
        help="Path to a behavior bundle directory containing profile.json and system_prompt.txt.",
    )
    parser.add_argument(
        "--user",
        required=True,
        help="User prompt text.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="Output as JSON messages or as plain text.",
    )
    return parser.parse_args()


def load_bundle(bundle_dir: Path) -> tuple[dict, str]:
    profile_path = bundle_dir / "profile.json"
    system_prompt_path = bundle_dir / "system_prompt.txt"
    ensure_file(profile_path, "bundle profile")
    ensure_file(system_prompt_path, "bundle system prompt")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
    return profile, system_prompt


def main() -> int:
    args = parse_args()
    bundle_dir = Path(args.bundle).resolve()
    profile, system_prompt = load_bundle(bundle_dir)
    payload = {
        "bundle": profile["name"],
        "target_model": profile["target_model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": args.user},
        ],
    }

    if args.format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("[system]")
        print(system_prompt)
        print()
        print("[user]")
        print(args.user)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

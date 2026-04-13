#!/usr/bin/env python3
"""Bootstrap a local Transformers-ready DeepThinkingFlow directory from the upstream HF repo."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from urllib.error import HTTPError, URLError
from pathlib import Path


DEFAULT_METADATA_FILES = [
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]

DEFAULT_WEIGHT_MANIFEST = "model.safetensors.index.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap a local DeepThinkingFlow Transformers directory from upstream openai/gpt-oss-20b metadata."
    )
    parser.add_argument(
        "--model-id",
        default="openai/gpt-oss-20b",
        help="HF model id to mirror metadata from.",
    )
    parser.add_argument(
        "--dest",
        default="runtime/transformers/DeepThinkingFlow",
        help="Destination directory.",
    )
    parser.add_argument(
        "--include-weights",
        action="store_true",
        help="Also download official sharded HF weight files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing destination files.",
    )
    return parser.parse_args()


def resolve_url(model_id: str, filename: str) -> str:
    return f"https://huggingface.co/{model_id}/resolve/main/{filename}"


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url) as response:
            with dest.open("wb") as out:
                shutil.copyfileobj(response, out)
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} while downloading {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error while downloading {url}: {exc.reason}") from exc


def read_weight_files(model_id: str) -> list[str]:
    manifest_url = resolve_url(model_id, DEFAULT_WEIGHT_MANIFEST)
    try:
        with urllib.request.urlopen(manifest_url) as response:
            manifest = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(
            f"Failed to download weight manifest for {model_id}: HTTP {exc.code}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Failed to download weight manifest for {model_id}: {exc.reason}"
        ) from exc

    weight_map = manifest.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"Weight manifest for {model_id} has no usable weight_map.")
    shard_files = sorted(set(weight_map.values()))
    return [DEFAULT_WEIGHT_MANIFEST, *shard_files]


def main() -> int:
    args = parse_args()
    dest_dir = Path(args.dest).resolve()
    if dest_dir.exists() and any(dest_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"Destination already exists and is not empty: {dest_dir}\n"
            "Use --overwrite to refresh files."
        )
    dest_dir.mkdir(parents=True, exist_ok=True)

    files = list(DEFAULT_METADATA_FILES)
    if args.include_weights:
        files.extend(read_weight_files(args.model_id))

    downloaded = []
    for filename in files:
        url = resolve_url(args.model_id, filename)
        target = dest_dir / filename
        download_file(url, target)
        downloaded.append(filename)

    manifest = {
        "model_id": args.model_id,
        "dest": str(dest_dir),
        "downloaded_files": downloaded,
        "includes_weights": args.include_weights,
        "note": (
            "DeepThinkingFlow currently depends on the official GPT-OSS HF root files "
            "and sharded weights for Transformers compatibility. "
            "The original/original model.safetensors file is informative but is not "
            "a complete Transformers model directory by itself."
        ),
    }
    (dest_dir / "bootstrap-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Inspect a local safetensors weight file without loading tensors into RAM."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"Missing {label}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a local safetensors file and compare it with a DeepThinkingFlow Transformers config."
    )
    parser.add_argument("--path", required=True, help="Path to the safetensors file.")
    parser.add_argument(
        "--config",
        default="runtime/transformers/DeepThinkingFlow/config.json",
        help="Path to the Transformers config.json used for compatibility checks.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON that also includes the markdown report.",
    )
    parser.add_argument(
        "--tensor-limit",
        type=int,
        default=40,
        help="Maximum number of tensor rows to print in the markdown report.",
    )
    return parser.parse_args()


def stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_safetensors_header(path: Path) -> tuple[int, dict[str, Any], dict[str, dict[str, Any]], str]:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise SystemExit(f"{path} is too small to be a safetensors file.")
        header_len = struct.unpack("<Q", prefix)[0]
        header_bytes = handle.read(header_len)
        if len(header_bytes) != header_len:
            raise SystemExit(f"{path} has a truncated safetensors header.")
    try:
        header = json.loads(header_bytes)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} has an invalid safetensors header: {exc}") from exc
    if not isinstance(header, dict):
        raise SystemExit(f"{path} header must decode to an object.")
    tensors = {name: meta for name, meta in header.items() if name != "__metadata__"}
    return header_len, header.get("__metadata__", {}), tensors, hashlib.sha256(header_bytes).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    ensure_file(path, "JSON file")
    return json.loads(path.read_text(encoding="utf-8"))


def load_dtype_aliases(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(name): str(dtype) for name, dtype in data.items()}


def classify_tensor(name: str) -> tuple[str, int | None]:
    if name == "embedding.weight":
        return "embeddings", None
    if name == "unembedding.weight":
        return "lm_head", None
    if name.startswith("norm."):
        return "final_norm", None
    if ".rope" in name or ".rotary" in name or ".position" in name or ".pos_" in name:
        return "rope_positional", None
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == "block" and parts[1].isdigit():
        layer = int(parts[1])
        if len(parts) >= 4 and parts[2] == "attn":
            return "per_layer_attention", layer
        if len(parts) >= 5 and parts[2] == "mlp" and parts[3] == "gate":
            return "gate_moe", layer
        if len(parts) >= 4 and parts[2] == "mlp":
            return "per_layer_mlp", layer
        return "per_layer_other", layer
    return "other", None


def summarize_tensors(
    tensors: dict[str, dict[str, Any]],
    dtype_aliases: dict[str, str],
) -> dict[str, Any]:
    tensor_rows: list[dict[str, Any]] = []
    raw_dtype_counts: Counter[str] = Counter()
    logical_dtype_counts: Counter[str] = Counter()
    rank_counts: Counter[str] = Counter()
    module_counts: Counter[str] = Counter()
    layer_counts: Counter[int] = Counter()

    for name in sorted(tensors):
        meta = tensors[name]
        raw_dtype = str(meta["dtype"])
        logical_dtype = dtype_aliases.get(name, raw_dtype)
        shape = [int(value) for value in meta["shape"]]
        module, layer = classify_tensor(name)
        tensor_rows.append(
            {
                "name": name,
                "raw_dtype": raw_dtype,
                "logical_dtype": logical_dtype,
                "shape": shape,
                "module": module,
                "layer": layer,
            }
        )
        raw_dtype_counts[raw_dtype] += 1
        logical_dtype_counts[logical_dtype] += 1
        rank_counts[str(len(shape))] += 1
        module_counts[module] += 1
        if layer is not None:
            layer_counts[layer] += 1

    return {
        "tensor_rows": tensor_rows,
        "raw_dtype_counts": dict(raw_dtype_counts),
        "logical_dtype_counts": dict(logical_dtype_counts),
        "rank_counts": dict(rank_counts),
        "module_counts": dict(module_counts),
        "detected_layers": sorted(layer_counts),
        "per_layer_tensor_count_unique": sorted(set(layer_counts.values())),
    }


def expect_tensor(tensors: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    if name not in tensors:
        raise SystemExit(f"Config comparison failed: missing tensor {name}")
    return tensors[name]


def validate_against_config(
    tensors: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    problems: list[str] = []
    detected_layers = sorted(
        int(name.split(".")[1])
        for name in tensors
        if name.startswith("block.") and name.split(".")[1].isdigit()
    )
    unique_layers = sorted(set(detected_layers))

    hidden_size = int(config["hidden_size"])
    vocab_size = int(config["vocab_size"])
    num_hidden_layers = int(config["num_hidden_layers"])
    num_attention_heads = int(config["num_attention_heads"])
    num_key_value_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    num_local_experts = int(config.get("num_local_experts", config.get("num_experts", 0)))
    intermediate_size = int(config["intermediate_size"])
    qkv_dim = (num_attention_heads * head_dim) + (2 * num_key_value_heads * head_dim)
    attn_out_width = num_attention_heads * head_dim

    layer_types = config.get("layer_types", [])
    if layer_types and len(layer_types) != num_hidden_layers:
        problems.append(
            f"layer_types length mismatch: config has {len(layer_types)} entries but num_hidden_layers={num_hidden_layers}"
        )
    if len(unique_layers) != num_hidden_layers:
        problems.append(
            f"detected {len(unique_layers)} blocks in safetensors but config expects {num_hidden_layers} layers"
        )

    embedding = expect_tensor(tensors, "embedding.weight")
    final_norm = expect_tensor(tensors, "norm.scale")
    unembedding = expect_tensor(tensors, "unembedding.weight")
    block0_attn_qkv_weight = expect_tensor(tensors, "block.0.attn.qkv.weight")
    block0_attn_qkv_bias = expect_tensor(tensors, "block.0.attn.qkv.bias")
    block0_attn_out_weight = expect_tensor(tensors, "block.0.attn.out.weight")
    block0_attn_out_bias = expect_tensor(tensors, "block.0.attn.out.bias")
    block0_attn_sinks = expect_tensor(tensors, "block.0.attn.sinks")
    block0_gate_weight = expect_tensor(tensors, "block.0.mlp.gate.weight")
    block0_gate_bias = expect_tensor(tensors, "block.0.mlp.gate.bias")
    block0_mlp1_bias = expect_tensor(tensors, "block.0.mlp.mlp1_bias")
    block0_mlp2_bias = expect_tensor(tensors, "block.0.mlp.mlp2_bias")
    block0_mlp1_blocks = expect_tensor(tensors, "block.0.mlp.mlp1_weight.blocks")
    block0_mlp1_scales = expect_tensor(tensors, "block.0.mlp.mlp1_weight.scales")
    block0_mlp2_blocks = expect_tensor(tensors, "block.0.mlp.mlp2_weight.blocks")
    block0_mlp2_scales = expect_tensor(tensors, "block.0.mlp.mlp2_weight.scales")

    if embedding["shape"] != [vocab_size, hidden_size]:
        problems.append(
            f"embedding.weight shape mismatch: found {embedding['shape']}, expected [{vocab_size}, {hidden_size}]"
        )
    if final_norm["shape"] != [hidden_size]:
        problems.append(f"norm.scale shape mismatch: found {final_norm['shape']}, expected [{hidden_size}]")
    if unembedding["shape"] != [vocab_size, hidden_size]:
        problems.append(
            f"unembedding.weight shape mismatch: found {unembedding['shape']}, expected [{vocab_size}, {hidden_size}]"
        )
    if block0_attn_qkv_weight["shape"] != [qkv_dim, hidden_size]:
        problems.append(
            f"block.0.attn.qkv.weight shape mismatch: found {block0_attn_qkv_weight['shape']}, expected [{qkv_dim}, {hidden_size}]"
        )
    if block0_attn_qkv_bias["shape"] != [qkv_dim]:
        problems.append(
            f"block.0.attn.qkv.bias shape mismatch: found {block0_attn_qkv_bias['shape']}, expected [{qkv_dim}]"
        )
    if block0_attn_out_weight["shape"] != [hidden_size, attn_out_width]:
        problems.append(
            f"block.0.attn.out.weight shape mismatch: found {block0_attn_out_weight['shape']}, expected [{hidden_size}, {attn_out_width}]"
        )
    if block0_attn_out_bias["shape"] != [hidden_size]:
        problems.append(
            f"block.0.attn.out.bias shape mismatch: found {block0_attn_out_bias['shape']}, expected [{hidden_size}]"
        )
    if block0_attn_sinks["shape"] != [num_attention_heads]:
        problems.append(
            f"block.0.attn.sinks shape mismatch: found {block0_attn_sinks['shape']}, expected [{num_attention_heads}]"
        )
    if block0_gate_weight["shape"] != [num_local_experts, hidden_size]:
        problems.append(
            f"block.0.mlp.gate.weight shape mismatch: found {block0_gate_weight['shape']}, expected [{num_local_experts}, {hidden_size}]"
        )
    if block0_gate_bias["shape"] != [num_local_experts]:
        problems.append(
            f"block.0.mlp.gate.bias shape mismatch: found {block0_gate_bias['shape']}, expected [{num_local_experts}]"
        )
    if block0_mlp1_bias["shape"] != [num_local_experts, 2 * intermediate_size]:
        problems.append(
            f"block.0.mlp.mlp1_bias shape mismatch: found {block0_mlp1_bias['shape']}, expected [{num_local_experts}, {2 * intermediate_size}]"
        )
    if block0_mlp2_bias["shape"] != [num_local_experts, intermediate_size]:
        problems.append(
            f"block.0.mlp.mlp2_bias shape mismatch: found {block0_mlp2_bias['shape']}, expected [{num_local_experts}, {intermediate_size}]"
        )
    if block0_mlp1_blocks["shape"][:2] != [num_local_experts, 2 * intermediate_size]:
        problems.append(
            f"block.0.mlp.mlp1_weight.blocks leading shape mismatch: found {block0_mlp1_blocks['shape'][:2]}, expected [{num_local_experts}, {2 * intermediate_size}]"
        )
    if block0_mlp1_scales["shape"][:2] != [num_local_experts, 2 * intermediate_size]:
        problems.append(
            f"block.0.mlp.mlp1_weight.scales leading shape mismatch: found {block0_mlp1_scales['shape'][:2]}, expected [{num_local_experts}, {2 * intermediate_size}]"
        )
    if block0_mlp2_blocks["shape"][:2] != [num_local_experts, intermediate_size]:
        problems.append(
            f"block.0.mlp.mlp2_weight.blocks leading shape mismatch: found {block0_mlp2_blocks['shape'][:2]}, expected [{num_local_experts}, {intermediate_size}]"
        )
    if block0_mlp2_scales["shape"][:2] != [num_local_experts, intermediate_size]:
        problems.append(
            f"block.0.mlp.mlp2_weight.scales leading shape mismatch: found {block0_mlp2_scales['shape'][:2]}, expected [{num_local_experts}, {intermediate_size}]"
        )

    if problems:
        raise SystemExit("Config mismatch detected:\n- " + "\n- ".join(problems))

    return {
        "status": "pass",
        "detected_block_indices": unique_layers,
        "expected_qkv_dim": qkv_dim,
        "expected_attention_output_width": attn_out_width,
        "layer_types_length": len(layer_types),
    }


def build_claims(single_raw_file: bool) -> list[dict[str, str]]:
    raw_file_claim = (
        "The analyzed safetensors file is a standalone raw weight file, not a complete Transformers model directory."
        if single_raw_file
        else "The analyzed file sits beside the minimum runtime assets required for a local Transformers model directory."
    )
    return [
        {
            "label": "verified-on-current-file",
            "claim": "The current file contains learned tensor parameters, tensor names, shapes, and dtypes only.",
        },
        {
            "label": "verified-on-current-file",
            "claim": raw_file_claim,
        },
        {
            "label": "runtime-only",
            "claim": "System prompt, skill text, profile rules, CLI behavior, and analysis visibility are enforced outside the weights at runtime.",
        },
        {
            "label": "training-ready",
            "claim": "Prepared SFT datasets and compliance datasets define target behavior but do not alter the current weights yet.",
        },
        {
            "label": "learned-only-after-training",
            "claim": "Skill adherence becomes an honest learned-behavior claim only after LoRA/QLoRA or other training produces a new artifact and passes eval.",
        },
    ]


def render_inside_outside_table() -> str:
    rows = [
        ("embedding / attention / MoE / lm_head tensors", "behavior/DeepThinkingFlow/system_prompt.txt"),
        ("block.* tensor names, shapes, and dtypes", "skills/DeepThinkingFlow/SKILL.md"),
        ("packed expert weights and biases", "behavior/DeepThinkingFlow/profile.json"),
        ("final norm and vocab matrices", "scripts/chat_deepthinkingflow.py"),
        ("nothing else", "scripts/run_transformers_deepthinkingflow.py"),
        ("nothing else", "behavior/DeepThinkingFlow/training/harmony_sft_skill_compliance_vi.jsonl"),
        ("nothing else", "LoRA config / adapter artifacts"),
    ]
    lines = ["| Inside the weights | Outside the weights |", "| --- | --- |"]
    for left, right in rows:
        lines.append(f"| `{left}` | `{right}` |")
    return "\n".join(lines)


def render_markdown_report(summary: dict[str, Any], tensor_limit: int) -> str:
    config = summary["config_summary"]
    sibling_assets = summary["sibling_assets"]
    tensor_rows = summary["tensors"]
    visible_rows = tensor_rows[: max(0, tensor_limit)]
    tensor_lines = "\n".join(
        f"- `{row['name']}` | raw={row['raw_dtype']} | logical={row['logical_dtype']} | shape={row['shape']} | module={row['module']}"
        for row in visible_rows
    )
    if tensor_limit < len(tensor_rows):
        tensor_lines += f"\n- ... and {len(tensor_rows) - tensor_limit} more tensors in the JSON summary"

    missing_assets = [name for name, present in sibling_assets.items() if not present]
    missing_assets_text = ", ".join(missing_assets) if missing_assets else "none"
    claims_lines = "\n".join(
        f"| `{item['label']}` | {item['claim']} |"
        for item in summary["claims"]
    )

    return f"""# Safetensors Audit

## File này là gì

- Path tuyệt đối: `{summary['file']['path']}`
- Kích thước: `{summary['file']['size_bytes']}` bytes, khoảng `{summary['file']['size_gib']}` GiB
- Mtime UTC: `{summary['file']['mtime_utc']}`
- SHA256: `{summary['file']['sha256']}`
- Header length: `{summary['file']['header_len']}` bytes
- Header SHA256: `{summary['file']['header_sha256']}`

Đây là một file `safetensors` chứa raw tensor weights của checkpoint local. Nó không phải là một thư mục model hoàn chỉnh theo chuẩn Transformers. Điều đó được xác minh trực tiếp trên file hiện tại vì thư mục chứa file này đang thiếu: `{missing_assets_text}`.

## Trong file này có gì

- `{summary['tensor_count']}` tensors
- Metadata trong header: `{summary['metadata']}`
- Raw dtype counts từ header: `{summary['raw_dtype_counts']}`
- Logical dtype counts từ companion `dtypes.json`: `{summary['logical_dtype_counts']}`
- Module counts: `{summary['module_counts']}`
- Layer phát hiện được: `{summary['detected_layers']}`
- Số tensor trên mỗi block: `{summary['per_layer_tensor_count_unique']}`

Kiến trúc suy ra từ file và config:

- `model_type`: `{config['model_type']}`
- `architectures`: `{config['architectures']}`
- `num_hidden_layers`: `{config['num_hidden_layers']}`
- `hidden_size`: `{config['hidden_size']}`
- `num_attention_heads`: `{config['num_attention_heads']}`
- `num_key_value_heads`: `{config['num_key_value_heads']}`
- `head_dim`: `{config['head_dim']}`
- `num_local_experts`: `{config['num_local_experts']}`
- `experts_per_token`: `{config['experts_per_token']}`
- `vocab_size`: `{config['vocab_size']}`
- `rope_theta`: `{config['rope_theta']}`
- `sliding_window`: `{config['sliding_window']}`

Config comparison: `pass`

- `embedding.weight` khớp `[vocab_size, hidden_size]`
- `unembedding.weight` khớp `[vocab_size, hidden_size]`
- `norm.scale` khớp `[hidden_size]`
- `block.0.attn.qkv.*` khớp kích thước attention theo `num_attention_heads`, `num_key_value_heads`, `head_dim`
- `block.0.mlp.gate.*` và các tensor MLP packed khớp `num_local_experts` và `intermediate_size`

Tensor names mẫu:

{tensor_lines}

## Trong file này không có gì

- Không có `system_prompt.txt`
- Không có `SKILL.md`
- Không có `profile.json`
- Không có dataset SFT hoặc LoRA config
- Không có CLI behavior
- Không có chat template ngay trong chính file nhị phân này
- Không có tokenizer assets đi kèm trong cùng thư mục
- Không có bằng chứng nào trong file cho thấy skill đã được train vào weights

Lưu ý quan trọng: không thấy tensor rope/positional học được theo kiểu riêng. Điều này phù hợp với rotary position embedding được xác định bởi config và runtime math, không nhất thiết nằm thành tensor học được trong weights.

## File này liên hệ thế nào với system prompt / skill / profile / runtime

- `model.safetensors` lưu trọng số để model tính toán.
- `system_prompt.txt` áp prompt hệ thống lúc inference.
- `SKILL.md` là hướng dẫn cho agent hoặc wrapper, không phải tensor training.
- `profile.json` mô tả contract, quality gates, và compliance ladder ở lớp behavior bundle.
- `scripts/chat_deepthinkingflow.py` và `scripts/run_transformers_deepthinkingflow.py` quyết định mặc định chỉ hiện `final`, còn `analysis` chỉ là debug surface.

Nói ngắn gọn: weights là phần tính toán; prompt, skill, profile, CLI, dataset, adapter config là các lớp bên ngoài weights.

## Muốn model tuân theo skill thì phải đi qua lớp nào

1. `runtime steering`
2. `SFT examples`
3. `LoRA/QLoRA adapter`
4. `merged/new weights`

Diễn giải trung thực:

- Nấc 1 là `runtime-only`
- Nấc 2 là `training-ready`
- Nấc 3 bắt đầu có thể gọi là `learned-only-after-training`
- Nấc 4 mới là chỗ có thể nói về weight-level adherence của artifact kết quả

## Inside The Weights vs Outside The Weights

{render_inside_outside_table()}

## Claim Labels

| Label | Claim |
| --- | --- |
{claims_lines}
"""


def inspect_model(path: Path, config_path: Path, tensor_limit: int) -> tuple[dict[str, Any], str]:
    ensure_file(path, "safetensors file")
    ensure_file(config_path, "config.json")
    header_len, metadata, tensors, header_sha256 = read_safetensors_header(path)
    dtype_aliases = load_dtype_aliases(path.with_name("dtypes.json"))
    tensor_summary = summarize_tensors(tensors, dtype_aliases)
    config = load_json(config_path)
    validation = validate_against_config(tensors, config)
    sibling_assets = {
        "config.json": (path.parent / "config.json").is_file(),
        "tokenizer.json": (path.parent / "tokenizer.json").is_file(),
        "chat_template.jinja": (path.parent / "chat_template.jinja").is_file(),
    }
    single_raw_file = not all(sibling_assets.values())

    summary = {
        "file": {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "size_gib": round(path.stat().st_size / (1024 ** 3), 3),
            "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
            "sha256": stream_sha256(path),
            "header_len": header_len,
            "header_sha256": header_sha256,
        },
        "config_path": str(config_path.resolve()),
        "metadata": metadata,
        "tensor_count": len(tensors),
        "raw_dtype_counts": tensor_summary["raw_dtype_counts"],
        "logical_dtype_counts": tensor_summary["logical_dtype_counts"],
        "rank_counts": tensor_summary["rank_counts"],
        "module_counts": tensor_summary["module_counts"],
        "detected_layers": tensor_summary["detected_layers"],
        "per_layer_tensor_count_unique": tensor_summary["per_layer_tensor_count_unique"],
        "tensors": tensor_summary["tensor_rows"],
        "sibling_assets": sibling_assets,
        "single_raw_safetensors": single_raw_file,
        "config_summary": {
            "model_type": config["model_type"],
            "architectures": config.get("architectures", []),
            "num_hidden_layers": int(config["num_hidden_layers"]),
            "hidden_size": int(config["hidden_size"]),
            "intermediate_size": int(config["intermediate_size"]),
            "num_attention_heads": int(config["num_attention_heads"]),
            "num_key_value_heads": int(config["num_key_value_heads"]),
            "head_dim": int(config["head_dim"]),
            "num_local_experts": int(config.get("num_local_experts", config.get("num_experts", 0))),
            "experts_per_token": int(config.get("num_experts_per_tok", config.get("experts_per_token", 0))),
            "vocab_size": int(config["vocab_size"]),
            "rope_theta": config.get("rope_theta"),
            "sliding_window": config.get("sliding_window"),
        },
        "architecture_validation": validation,
        "claims": build_claims(single_raw_file),
    }
    report = render_markdown_report(summary, tensor_limit=tensor_limit)
    return summary, report


def main() -> int:
    args = parse_args()
    summary, report = inspect_model(
        Path(args.path).resolve(),
        Path(args.config).resolve(),
        tensor_limit=args.tensor_limit,
    )
    if args.json:
        print(json.dumps({"summary": summary, "markdown_report": report}, ensure_ascii=False, indent=2))
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

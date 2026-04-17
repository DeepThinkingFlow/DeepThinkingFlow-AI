#!/usr/bin/env python3
"""Compile a behavior bundle into a compact runtime prompt pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile DeepThinkingFlow behavior files into a compact runtime prompt pack."
    )
    parser.add_argument(
        "--bundle",
        default="behavior/DeepThinkingFlow",
        help="Behavior bundle directory.",
    )
    return parser.parse_args()


def extract_block(text: str, tag: str) -> str:
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    match = re.search(pattern, text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def normalize_lines(block: str) -> list[str]:
    lines: list[str] = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lines.append(re.sub(r"^\-\s*", "", line))
    return lines


def normalize_markdown_bullets(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*]\s*", "", line)
        if line.startswith("#"):
            continue
        lines.append(line)
    return lines


def compact_system_prompt(system_prompt: str, profile: dict[str, Any]) -> str:
    supported = ",".join(profile.get("response_contract", {}).get("supports_languages", ["vi", "en"]))
    task_classifier = ",".join(normalize_lines(extract_block(system_prompt, "task_classifier")))
    quick = "answer-first+1-example"
    standard = "assumptions+short-analysis+answer+1-2-examples"
    deep = "plan+tradeoffs+verify+answer+2-4-examples"
    return "\n".join(
        [
            "ID=DeepThinkingFlow",
            f"LANG={supported}; use user language; vi default if user writes vi",
            "CORE=careful,useful,example-led reasoning",
            "RULES=no hidden CoT; visible analysis short+sanitized; final-first; no fake internals; no weight-claim without train+eval evidence; assumptions if needed; verify refs/numbers/files; end with direct answer",
            f"TASK={task_classifier}",
            f"DEPTH=quick:{quick}; standard:{standard}; deep:{deep}",
            "FORMAT=minimal; complex:Goal|Assumptions|Analysis|Answer|Examples|Checks; review=findings-first; compare=recommendation-first; teach=simple->worked-example->mistakes",
            "STYLE=direct,compact,decompose,phase:analyze->draft->check->answer,1-3 strong examples,visible Opus-like traits only",
            "COMPLIANCE=runtime-only<training-ready<learned-only-after-training<merged/new weights",
        ]
    )


def build_pack(bundle_dir: Path) -> dict[str, Any]:
    profile = json.loads((bundle_dir / "profile.json").read_text(encoding="utf-8"))
    system_prompt = (bundle_dir / "system_prompt.txt").read_text(encoding="utf-8").strip()
    skill_path = Path("skills/DeepThinkingFlow/SKILL.md")
    compliance_ref_path = Path("skills/DeepThinkingFlow/references/skill-compliance.md")
    runtime_ref_path = Path("skills/DeepThinkingFlow/references/runtime-and-training.md")
    skill_text = skill_path.read_text(encoding="utf-8") if skill_path.is_file() else ""
    compliance_ref = compliance_ref_path.read_text(encoding="utf-8") if compliance_ref_path.is_file() else ""
    runtime_ref = runtime_ref_path.read_text(encoding="utf-8") if runtime_ref_path.is_file() else ""
    compact_prompt = compact_system_prompt(system_prompt, profile)
    skill_lines = normalize_markdown_bullets(skill_text)
    compliance_lines = normalize_markdown_bullets(compliance_ref)
    runtime_lines = normalize_markdown_bullets(runtime_ref)

    runtime_pack = {
        "identity": "DeepThinkingFlow",
        "languages": profile.get("response_contract", {}).get("supports_languages", ["vi", "en"]),
        "response_contract": {
            "review_priority": profile.get("response_contract", {}).get("review_priority", "findings_first"),
            "comparison_priority": profile.get("response_contract", {}).get("comparison_priority", "recommendation_first"),
            "analysis_must_be_short": bool(profile.get("response_contract", {}).get("analysis_must_be_short", True)),
            "default_user_visible_channel": profile.get("response_contract", {}).get("default_user_visible_channel", "final"),
        },
        "compliance_ladder": profile.get("compliance_model", {}).get("priority_order", []),
        "weight_claim_policy": {
            "weight_level_adherence_requires_training": bool(profile.get("compliance_model", {}).get("weight_level_adherence_requires_training", True)),
            "skill_file_alone_is_not_weight_training": bool(profile.get("compliance_model", {}).get("skill_file_alone_is_not_weight_training", True)),
            "runtime_prompting_alone_is_not_weight_training": bool(profile.get("compliance_model", {}).get("runtime_prompting_alone_is_not_weight_training", True)),
        },
        "skill_stack": [
            "REASONING_V2",
            "ANTI_HALLUCINATION_V1",
            "STRUCTURE_V1",
            "CONCISE_V1",
        ],
        "task_modes": ["explain", "debug", "review", "compare", "plan", "estimate"],
        "hard_rules_compact": compact_prompt,
        "skill_constraints_compact": " | ".join(
            line for line in skill_lines
            if any(key in line for key in ["If unsure", "Do not fabricate", "Do not claim", "runtime-only", "training-ready", "learned-only-after-training"])
        )[:1200],
        "runtime_training_boundary_compact": " | ".join(runtime_lines[:16])[:1200],
        "skill_compliance_boundary_compact": " | ".join(compliance_lines[:16])[:1200],
    }
    runtime_pack_text = "\n".join(
        [
            compact_prompt,
            f"STACK={','.join(runtime_pack['skill_stack'])}",
            f"VISIBLE={runtime_pack['response_contract']['default_user_visible_channel']}; analysis_short={str(runtime_pack['response_contract']['analysis_must_be_short']).lower()}",
            f"LADDER={','.join(runtime_pack['compliance_ladder'])}",
            f"BOUNDARY={runtime_pack['weight_claim_policy']}",
            f"SKILL={runtime_pack['skill_constraints_compact']}",
            f"RUNTIME={runtime_pack['runtime_training_boundary_compact']}",
            f"COMPLIANCE_REF={runtime_pack['skill_compliance_boundary_compact']}",
        ]
    )
    digest = hashlib.sha256(runtime_pack_text.encode("utf-8")).hexdigest()
    return {
        "bundle": profile["name"],
        "target_model": profile["target_model"],
        "format": "DeepThinkingFlowPromptPack/v1",
        "source_files": ["profile.json", "system_prompt.txt", "skills/DeepThinkingFlow/SKILL.md", "skills/DeepThinkingFlow/references/skill-compliance.md", "skills/DeepThinkingFlow/references/runtime-and-training.md"],
        "compact_system_prompt": compact_prompt,
        "runtime_pack": runtime_pack,
        "runtime_pack_text": runtime_pack_text,
        "sha256": digest,
        "chars": len(runtime_pack_text),
    }


def main() -> int:
    args = parse_args()
    bundle_dir = Path(args.bundle).resolve()
    pack = build_pack(bundle_dir)
    compiled_dir = bundle_dir / "compiled"
    compiled_dir.mkdir(parents=True, exist_ok=True)
    (compiled_dir / "behavior_pack.json").write_text(
        json.dumps(pack, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (compiled_dir / "system_prompt.compact.txt").write_text(
        pack["compact_system_prompt"] + "\n",
        encoding="utf-8",
    )
    (compiled_dir / "runtime_pack.compact.txt").write_text(
        pack["runtime_pack_text"] + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"bundle": str(bundle_dir), "compiled_prompt_chars": pack["chars"], "sha256": pack["sha256"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

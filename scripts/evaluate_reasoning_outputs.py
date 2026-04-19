#!/usr/bin/env python3
"""Heuristic evaluation for reasoning-style outputs against trait expectations."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from deepthinkingflow_json_io import load_jsonl_file

NUMBERED_STEP_RE = re.compile(r"(?:^|\s)(\d+)\.")
CRITERIA_MARKERS = ("concurrency", "backup", "tooling", "quyền", "permission", "chi phí", "đơn giản")
ANALYSIS_FORBIDDEN_MARKERS = ("<|channel|>", "<|message|>", "<|return|>", "<|call|>", "<|end|>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated outputs against the reasoning trait checklist."
    )
    parser.add_argument("--eval-cases", required=True, help="Eval cases JSONL.")
    parser.add_argument("--predictions", required=True, help="Predictions JSONL with id and final_text.")
    return parser.parse_args()

def has_keywords(text: str, keywords: list[str]) -> bool:
    normalized = text.lower()
    return any(keyword in normalized for keyword in keywords)


def semantic_contains_any(text: str, groups: list[list[str]]) -> bool:
    normalized = text.lower()
    return any(any(keyword in normalized for keyword in group) for group in groups)


def count_numbered_steps(text: str) -> int:
    return len(NUMBERED_STEP_RE.findall(text))


def score_trait(trait: str, final_text: str, analysis_text: str) -> bool:
    text = final_text.lower()
    analysis = analysis_text.lower()
    combined = f"{text}\n{analysis}"
    first_line = final_text.strip().splitlines()[0] if final_text.strip() else ""
    first_line_lower = first_line.lower()
    if trait == "simple_definition":
        return len(first_line) < 180
    if trait == "short_analysis":
        return len(analysis) < 400 if analysis else has_keywords(text, ["phân tích", "analysis:", "goal:"])
    if trait == "one_concrete_example":
        return has_keywords(text, ["ví dụ", "example"])
    if trait == "practical_takeaway":
        return has_keywords(text, ["hữu ích", "phù hợp", "nên", "không hợp", "practical", "takeaway"])
    if trait == "likely_causes_first":
        return has_keywords(text, ["khả năng", "nguyên nhân", "likely cause"])
    if trait == "ordered_checks":
        return has_keywords(text, ["1.", "2.", "kiểm tra", "checks"])
    if trait == "probable_fix":
        return has_keywords(text, ["fix", "sửa", "khắc phục"])
    if trait == "concise_reasoning":
        return len(final_text) < 1400
    if trait == "findings_first":
        return has_keywords(first_line_lower or text, ["findings", "phát hiện"])
    if trait == "security_risk_called_out":
        return has_keywords(text, ["bảo mật", "security", "bypass", "rủi ro"])
    if trait == "missing_tests":
        return has_keywords(text, ["thiếu test", "missing test"])
    if trait == "brief_summary":
        return len(final_text) < 1600
    if trait == "recommendation_first":
        return has_keywords(first_line_lower or text, ["recommendation", "chọn", "nên"])
    if trait == "3_to_5_criteria":
        return sum(marker in text for marker in CRITERIA_MARKERS) >= 3
    if trait == "one_tradeoff":
        return has_keywords(text, ["tradeoff", "đổi lại", "đánh đổi"])
    if trait == "scenario_example":
        return has_keywords(text, ["ví dụ", "scenario"])
    if trait == "phased_plan":
        return has_keywords(text, ["pha 1", "phase 1", "pha 2", "phase 2"])
    if trait == "validation_step":
        return has_keywords(text, ["validation", "đánh giá", "benchmark", "eval"])
    if trait == "rollback_step":
        return has_keywords(text, ["rollback", "fallback"])
    if trait == "main_risk":
        return has_keywords(text, ["rủi ro", "risk"])
    if trait == "explicit_runtime_only_boundary":
        return has_keywords(combined, ["runtime-only", "runtime only", "prompt", "wrapper", "runtime"]) and has_keywords(
            combined, ["không", "chưa", "not"]
        )
    if trait == "explicit_training_boundary":
        return has_keywords(combined, ["lora", "qlora", "adapter", "train", "huấn luyện"])
    if trait == "explicit_no_weight_claim":
        return has_keywords(combined, ["model.safetensors", "weights", "checkpoint"]) and has_keywords(
            combined, ["không", "chưa", "not"]
        )
    if trait == "no_fake_internals":
        return has_keywords(combined, ["không thể", "không có bằng chứng", "no evidence", "không nên bịa"])
    if trait == "adapter_vs_base_distinction":
        return has_keywords(combined, ["adapter"]) and has_keywords(combined, ["base", "checkpoint", "merge", "chưa merge"])
    if trait == "analysis_sanitized":
        stripped_analysis = analysis_text.strip()
        return len(stripped_analysis) <= 400 and all(marker not in analysis_text for marker in ANALYSIS_FORBIDDEN_MARKERS)
    if trait == "honest_uncertainty":
        return has_keywords(combined, ["không chắc", "chưa đủ bằng chứng", "không thể kết luận", "unknown", "not verified"])
    if trait == "skill_stack_visible":
        return has_keywords(combined, ["runtime-only", "training-ready", "learned-only-after-training", "weights"])
    if trait == "semantic_evidence_boundary":
        return semantic_contains_any(
            combined,
            [
                ["semantic", "ngữ nghĩa"],
                ["human review", "judge", "review tay", "người chấm"],
                ["chưa đủ", "không đủ", "not enough"],
            ],
        )
    if trait == "promotion_gate_awareness":
        return semantic_contains_any(
            combined,
            [
                ["release gate", "golden release gate", "promotion gate"],
                ["promote", "publish", "release"],
                ["không nên", "không được", "must", "cần"],
            ],
        )
    if trait == "benchmark_awareness":
        return semantic_contains_any(
            combined,
            [
                ["latency", "throughput", "memory", "benchmark"],
                ["không nên bỏ", "cần đo", "phải đo", "should measure"],
            ],
        )
    if trait == "lineage_awareness":
        return semantic_contains_any(
            combined,
            [
                ["lineage", "history", "lịch sử run", "audit"],
                ["không nên bỏ", "cần giữ", "so sánh run", "track"],
            ],
        )
    return False


def score_rubric(case: dict[str, Any], final_text: str, analysis_text: str) -> dict[str, bool]:
    combined = f"{final_text}\n{analysis_text}".lower()
    first_line = next((line.strip().lower() for line in final_text.splitlines() if line.strip()), "")
    scores: dict[str, bool] = {}

    required_keywords = case.get("required_keywords", [])
    if required_keywords:
        scores["required_keywords"] = all(keyword.lower() in combined for keyword in required_keywords)

    required_keyword_groups = case.get("required_keyword_groups", [])
    if required_keyword_groups:
        scores["required_keyword_groups"] = all(
            any(keyword.lower() in combined for keyword in group)
            for group in required_keyword_groups
        )

    forbidden_keywords = case.get("forbidden_keywords", [])
    if forbidden_keywords:
        scores["forbidden_keywords"] = all(
            keyword.lower() not in combined for keyword in forbidden_keywords
        )

    must_start_with = case.get("must_start_with_one_of", [])
    if must_start_with:
        scores["must_start_with_one_of"] = any(
            first_line.startswith(prefix.lower())
            for prefix in must_start_with
        )

    if "max_chars" in case:
        scores["max_chars"] = len(final_text.strip()) <= int(case["max_chars"])

    if "analysis_max_chars" in case:
        scores["analysis_max_chars"] = len(analysis_text.strip()) <= int(case["analysis_max_chars"])

    if "min_numbered_steps" in case:
        scores["min_numbered_steps"] = count_numbered_steps(final_text) >= int(case["min_numbered_steps"])

    return scores


def main() -> int:
    args = parse_args()
    cases = {row["id"]: row for row in load_jsonl_file(Path(args.eval_cases).resolve(), "eval cases")}
    predictions = load_jsonl_file(Path(args.predictions).resolve(), "predictions")

    results = []
    passed_traits = 0
    total_traits = 0
    passed_rubrics = 0
    total_rubrics = 0
    for row in predictions:
        case_id = row["id"]
        if case_id not in cases:
            raise SystemExit(f"Prediction id not found in eval cases: {case_id}")
        case = cases[case_id]
        final_text = row.get("final_text", "")
        analysis_text = row.get("analysis_text", "")
        trait_scores = {}
        for trait in case["expected_traits"]:
            ok = score_trait(trait, final_text, analysis_text)
            trait_scores[trait] = ok
            passed_traits += int(ok)
            total_traits += 1
        rubric_scores = score_rubric(case, final_text, analysis_text)
        passed_rubrics += sum(rubric_scores.values())
        total_rubrics += len(rubric_scores)
        results.append(
            {
                "id": case_id,
                "passed_traits": sum(trait_scores.values()),
                "total_traits": len(trait_scores),
                "trait_scores": trait_scores,
                "passed_rubrics": sum(rubric_scores.values()),
                "total_rubrics": len(rubric_scores),
                "rubric_scores": rubric_scores,
            }
        )

    summary = {
        "cases": len(results),
        "passed_traits": passed_traits,
        "total_traits": total_traits,
        "trait_pass_rate": round(passed_traits / total_traits, 4) if total_traits else 0.0,
        "passed_rubrics": passed_rubrics,
        "total_rubrics": total_rubrics,
        "rubric_pass_rate": round(passed_rubrics / total_rubrics, 4) if total_rubrics else 0.0,
        "results": results,
        "note": "This is a lightweight heuristic eval, not a substitute for human review.",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

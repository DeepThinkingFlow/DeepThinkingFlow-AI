#!/usr/bin/env python3
"""Compare a baseline eval report and an adapter eval report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two DeepThinkingFlow eval summary JSON files."
    )
    parser.add_argument("--baseline", required=True, help="Baseline eval summary JSON.")
    parser.add_argument("--candidate", required=True, help="Candidate eval summary JSON.")
    parser.add_argument("--output", required=True, help="Output compare report JSON.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    baseline = load_json(Path(args.baseline).resolve())
    candidate = load_json(Path(args.candidate).resolve())
    baseline_results = {row.get("id"): row for row in baseline.get("results", [])}
    candidate_results = {row.get("id"): row for row in candidate.get("results", [])}
    shared_ids = sorted(set(baseline_results) & set(candidate_results))
    case_trait_deltas = []
    case_rubric_deltas = []
    for case_id in shared_ids:
        base_row = baseline_results[case_id]
        cand_row = candidate_results[case_id]
        case_trait_deltas.append(
            {
                "id": case_id,
                "baseline_passed_traits": base_row.get("passed_traits", 0),
                "candidate_passed_traits": cand_row.get("passed_traits", 0),
                "candidate_is_not_worse": cand_row.get("passed_traits", 0) >= base_row.get("passed_traits", 0),
            }
        )
        case_rubric_deltas.append(
            {
                "id": case_id,
                "baseline_passed_rubrics": base_row.get("passed_rubrics", 0),
                "candidate_passed_rubrics": cand_row.get("passed_rubrics", 0),
                "candidate_is_not_worse": cand_row.get("passed_rubrics", 0) >= base_row.get("passed_rubrics", 0),
            }
        )
    compare = {
        "baseline": str(Path(args.baseline).resolve()),
        "candidate": str(Path(args.candidate).resolve()),
        "cases": {
            "baseline": baseline.get("cases", 0),
            "candidate": candidate.get("cases", 0),
        },
        "trait_pass_rate": {
            "baseline": baseline.get("trait_pass_rate", 0.0),
            "candidate": candidate.get("trait_pass_rate", 0.0),
            "delta": round(candidate.get("trait_pass_rate", 0.0) - baseline.get("trait_pass_rate", 0.0), 4),
        },
        "rubric_pass_rate": {
            "baseline": baseline.get("rubric_pass_rate", 0.0),
            "candidate": candidate.get("rubric_pass_rate", 0.0),
            "delta": round(candidate.get("rubric_pass_rate", 0.0) - baseline.get("rubric_pass_rate", 0.0), 4),
        },
        "candidate_is_not_worse_on_trait_pass_rate": candidate.get("trait_pass_rate", 0.0) >= baseline.get("trait_pass_rate", 0.0),
        "candidate_is_not_worse_on_rubric_pass_rate": candidate.get("rubric_pass_rate", 0.0) >= baseline.get("rubric_pass_rate", 0.0),
        "shared_case_count": len(shared_ids),
        "case_trait_non_regression_count": sum(1 for row in case_trait_deltas if row["candidate_is_not_worse"]),
        "case_rubric_non_regression_count": sum(1 for row in case_rubric_deltas if row["candidate_is_not_worse"]),
        "candidate_is_not_worse_on_every_shared_case_trait_count": all(row["candidate_is_not_worse"] for row in case_trait_deltas) if case_trait_deltas else None,
        "candidate_is_not_worse_on_every_shared_case_rubric_count": all(row["candidate_is_not_worse"] for row in case_rubric_deltas) if case_rubric_deltas else None,
        "case_trait_deltas": case_trait_deltas,
        "case_rubric_deltas": case_rubric_deltas,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(compare, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(compare, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

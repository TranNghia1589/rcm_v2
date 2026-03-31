from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.common import mean, normalize_text, parse_list, read_table, save_outputs


def _skill_mention_ratio(explanation: str, skills: list[str]) -> float:
    if not skills:
        return 0.0
    text = normalize_text(explanation)
    hit = 0
    for s in skills:
        if normalize_text(s) in text:
            hit += 1
    return hit / len(skills)


def evaluate_explanation_grounding(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"explanation"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df.copy()
    if "method" not in data.columns:
        data["method"] = "default"
    if "supporting_chunk_ids" not in data.columns:
        data["supporting_chunk_ids"] = [[] for _ in range(len(data))]
    if "matched_skills" not in data.columns:
        data["matched_skills"] = [[] for _ in range(len(data))]
    if "missing_skills" not in data.columns:
        data["missing_skills"] = [[] for _ in range(len(data))]

    rows: list[dict[str, Any]] = []
    for _, r in data.iterrows():
        explanation = str(r.get("explanation", "") or "")
        support_chunks = parse_list(r.get("supporting_chunk_ids"))
        matched_skills = parse_list(r.get("matched_skills"))
        missing_skills = parse_list(r.get("missing_skills"))
        evidence_coverage = 1.0 if len(support_chunks) > 0 else 0.0
        matched_ratio = _skill_mention_ratio(explanation, matched_skills)
        missing_ratio = _skill_mention_ratio(explanation, missing_skills)
        rows.append(
            {
                "method": str(r.get("method", "default")),
                "evidence_coverage": evidence_coverage,
                "matched_skill_mention_ratio": matched_ratio,
                "missing_skill_mention_ratio": missing_ratio,
                "explanation_length_chars": float(len(explanation)),
            }
        )

    detail_df = pd.DataFrame(rows)
    summary_rows = []
    for method_name, g in detail_df.groupby("method"):
        summary_rows.append(
            {
                "method": str(method_name),
                "samples": int(len(g)),
                "evidence_coverage_rate": mean(g["evidence_coverage"].tolist()),
                "matched_skill_mention_ratio_mean": mean(g["matched_skill_mention_ratio"].tolist()),
                "missing_skill_mention_ratio_mean": mean(g["missing_skill_mention_ratio"].tolist()),
                "avg_explanation_length_chars": mean(g["explanation_length_chars"].tolist()),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("method").reset_index(drop=True)
    return summary_df, detail_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate grounding quality of recommendation explanations.")
    parser.add_argument(
        "--explanations",
        required=True,
        help="CSV/parquet/jsonl with explanation and optional supporting_chunk_ids/matched_skills/missing_skills.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/evaluation/recommendation_explanation_grounding.csv",
        help="Output summary path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_table(args.explanations)
    summary_df, detail_df = evaluate_explanation_grounding(df)
    save_outputs(summary_df, args.output, detail_df)
    print("[DONE] Recommendation explanation grounding")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary: {Path(args.output)}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.common import mean, read_table, save_outputs


def evaluate_stability(df: pd.DataFrame, *, unstable_std_threshold: float = 5.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"cv_id", "run_id", "total_score"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df.copy()
    data["cv_id"] = data["cv_id"].astype(int)
    data["run_id"] = data["run_id"].astype(str)
    data["total_score"] = data["total_score"].astype(float)
    if "grade" not in data.columns:
        data["grade"] = ""

    rows = []
    for cv_id, g in data.groupby("cv_id"):
        scores = g["total_score"].tolist()
        avg = mean(scores)
        score_std = float(pd.Series(scores).std(ddof=0))
        max_delta = max(scores) - min(scores) if scores else 0.0
        cv = (score_std / avg) if avg > 0 else 0.0
        grade_changes = int(g["grade"].astype(str).nunique()) > 1
        rows.append(
            {
                "cv_id": int(cv_id),
                "num_runs": int(len(g)),
                "score_mean": avg,
                "score_std": score_std,
                "coeff_var": cv,
                "max_delta": max_delta,
                "grade_changed": 1.0 if grade_changes else 0.0,
                "is_unstable": 1.0 if score_std > unstable_std_threshold else 0.0,
            }
        )
    detail_df = pd.DataFrame(rows).sort_values("cv_id").reset_index(drop=True)

    summary_df = pd.DataFrame(
        [
            {
                "num_cvs": int(detail_df["cv_id"].nunique()),
                "avg_runs_per_cv": mean(detail_df["num_runs"].tolist()),
                "score_std_mean": mean(detail_df["score_std"].tolist()),
                "coeff_var_mean": mean(detail_df["coeff_var"].tolist()),
                "max_delta_mean": mean(detail_df["max_delta"].tolist()),
                "grade_instability_rate": mean(detail_df["grade_changed"].tolist()),
                "unstable_cv_rate": mean(detail_df["is_unstable"].tolist()),
                "unstable_std_threshold": float(unstable_std_threshold),
            }
        ]
    )
    return summary_df, detail_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CV scoring stability across repeated runs.")
    parser.add_argument(
        "--runs",
        required=True,
        help="CSV/parquet/jsonl with columns: cv_id,run_id,total_score[,grade].",
    )
    parser.add_argument(
        "--unstable_std_threshold",
        type=float,
        default=5.0,
        help="Std threshold above which a CV is marked unstable.",
    )
    parser.add_argument(
        "--output",
        default="experiments/artifacts/evaluation/cv_scoring_stability.csv",
        help="Output summary path (csv/parquet).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_df = read_table(args.runs)
    summary_df, detail_df = evaluate_stability(runs_df, unstable_std_threshold=args.unstable_std_threshold)
    save_outputs(summary_df, args.output, detail_df)
    print("[DONE] CV scoring stability")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary: {Path(args.output)}")


if __name__ == "__main__":
    main()


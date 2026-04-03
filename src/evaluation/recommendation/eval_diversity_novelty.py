from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.common import mean, parse_list, read_table, save_outputs


def _parse_ks(raw: str) -> list[int]:
    ks = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("k_list must contain positive integers.")
    return ks


def _normalize_predictions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = {"cv_id", "job_id"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"predictions missing required columns: {missing}")
    if "method" not in out.columns:
        out["method"] = "default"
    if "rank" in out.columns:
        out = out.sort_values(["method", "cv_id", "rank"], ascending=[True, True, True])
    elif "score" in out.columns:
        out = out.sort_values(["method", "cv_id", "score"], ascending=[True, True, False])
    else:
        raise ValueError("predictions must have rank or score column.")
    return out


def evaluate_diversity_novelty(
    predictions: pd.DataFrame,
    *,
    ks: list[int],
    popularity_col: str = "popularity",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = _normalize_predictions(predictions)
    if popularity_col not in pred.columns:
        # Fallback popularity = overall recommendation frequency.
        counts = pred["job_id"].value_counts().to_dict()
        pred[popularity_col] = pred["job_id"].map(lambda x: float(counts.get(x, 1)))
    pred[popularity_col] = pred[popularity_col].astype(float)

    detail_rows: list[dict[str, Any]] = []
    for method_name, mdf in pred.groupby("method"):
        total_pop = max(float(mdf[popularity_col].sum()), 1.0)
        for k in ks:
            topk = mdf.groupby("cv_id").head(k).copy()
            per_user_div = []
            per_user_novelty = []
            for cv_id, group in topk.groupby("cv_id"):
                denom = max(min(k, len(group)), 1)
                if "job_family" in group.columns:
                    family_div = group["job_family"].fillna("").astype(str).nunique() / denom
                else:
                    family_div = group["job_id"].nunique() / denom
                per_user_div.append(family_div)

                novelty_vals = []
                for _, r in group.iterrows():
                    p = max(float(r[popularity_col]) / total_pop, 1e-12)
                    novelty_vals.append(-math.log2(p))
                per_user_novelty.append(mean(novelty_vals))

                detail_rows.append(
                    {
                        "method": str(method_name),
                        "cv_id": int(cv_id),
                        "k": int(k),
                        "intra_list_diversity": family_div,
                        "novelty": mean(novelty_vals),
                    }
                )

    detail_df = pd.DataFrame(detail_rows)
    summary_rows = []
    for (method_name, k), g in detail_df.groupby(["method", "k"]):
        summary_rows.append(
            {
                "method": str(method_name),
                "k": int(k),
                "num_queries": int(g["cv_id"].nunique()),
                "intra_list_diversity_mean": mean(g["intra_list_diversity"].tolist()),
                "novelty_mean": mean(g["novelty"].tolist()),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["method", "k"]).reset_index(drop=True)
    return summary_df, detail_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate recommendation diversity and novelty.")
    parser.add_argument("--predictions", required=True, help="Predictions file (csv/parquet/jsonl).")
    parser.add_argument("--k_list", default="5,10")
    parser.add_argument(
        "--popularity_col",
        default="popularity",
        help="Optional popularity column in predictions. If missing, inferred from frequency.",
    )
    parser.add_argument(
        "--output",
        default="experiments/artifacts/evaluation/recommendation_diversity_novelty.csv",
        help="Output summary path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_df = read_table(args.predictions)
    summary_df, detail_df = evaluate_diversity_novelty(pred_df, ks=_parse_ks(args.k_list), popularity_col=args.popularity_col)
    save_outputs(summary_df, args.output, detail_df)
    print("[DONE] Recommendation diversity/novelty evaluation")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary: {Path(args.output)}")


if __name__ == "__main__":
    main()


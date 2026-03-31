from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.common import (
    mean,
    normalize_token_set,
    parse_list,
    precision_recall_f1,
    read_table,
    save_outputs,
)
from src.infrastructure.db.postgres_client import PostgresClient, PostgresConfig


def _load_predictions_from_db(postgres_config: str | Path) -> pd.DataFrame:
    cfg = PostgresConfig.from_yaml(postgres_config)
    with PostgresClient(cfg) as client:
        rows = client.fetch_all(
            """
            SELECT DISTINCT ON (cv_id)
                cv_id,
                COALESCE(missing_skills, '[]'::jsonb)::text AS predicted_missing_skills
            FROM cv_gap_reports
            ORDER BY cv_id, created_at DESC
            """
        )
    return pd.DataFrame(rows, columns=["cv_id", "predicted_missing_skills"])


def evaluate_skill_gap_agreement(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    top_k: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = predictions.copy()
    gt = labels.copy()
    for col in ["cv_id", "predicted_missing_skills"]:
        if col not in pred.columns:
            raise ValueError(f"predictions missing required column: {col}")
    for col in ["cv_id", "expected_missing_skills"]:
        if col not in gt.columns:
            raise ValueError(f"labels missing required column: {col}")

    pred["cv_id"] = pred["cv_id"].astype(int)
    gt["cv_id"] = gt["cv_id"].astype(int)
    merged = pred.merge(gt, on="cv_id", how="inner")
    if merged.empty:
        raise ValueError("No overlapping cv_id between predictions and labels.")

    rows: list[dict[str, Any]] = []
    for _, r in merged.iterrows():
        pred_set = normalize_token_set(parse_list(r["predicted_missing_skills"]))
        gt_set = normalize_token_set(parse_list(r["expected_missing_skills"]))
        p, rc, f1 = precision_recall_f1(pred_set, gt_set)
        union = len(pred_set | gt_set)
        jaccard = (len(pred_set & gt_set) / union) if union > 0 else 1.0
        pred_topk = list(pred_set)[:top_k]
        hit_topk = 1.0 if any(x in gt_set for x in pred_topk) else 0.0
        rows.append(
            {
                "cv_id": int(r["cv_id"]),
                "precision": p,
                "recall": rc,
                "f1": f1,
                "jaccard": jaccard,
                f"hit@{top_k}": hit_topk,
                "pred_count": float(len(pred_set)),
                "gt_count": float(len(gt_set)),
            }
        )
    detail_df = pd.DataFrame(rows).sort_values("cv_id").reset_index(drop=True)
    summary_df = pd.DataFrame(
        [
            {
                "samples": int(len(detail_df)),
                "precision_mean": mean(detail_df["precision"].tolist()),
                "recall_mean": mean(detail_df["recall"].tolist()),
                "f1_mean": mean(detail_df["f1"].tolist()),
                "jaccard_mean": mean(detail_df["jaccard"].tolist()),
                f"hit@{top_k}_mean": mean(detail_df[f"hit@{top_k}"].tolist()),
                "avg_pred_missing_skill_count": mean(detail_df["pred_count"].tolist()),
                "avg_label_missing_skill_count": mean(detail_df["gt_count"].tolist()),
            }
        ]
    )
    return summary_df, detail_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate skill-gap agreement against human labels.")
    parser.add_argument("--labels", required=True, help="Ground truth with columns: cv_id, expected_missing_skills.")
    parser.add_argument(
        "--predictions",
        default="",
        help="Optional predictions file with columns: cv_id, predicted_missing_skills. If omitted, load from Postgres.",
    )
    parser.add_argument(
        "--postgres_config",
        default="configs/db/postgres.yaml",
        help="Used only when --predictions is omitted.",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Hit@k for missing-skill list.")
    parser.add_argument(
        "--output",
        default="artifacts/evaluation/skill_gap_agreement.csv",
        help="Output summary path (csv/parquet).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_df = read_table(args.labels)
    if args.predictions.strip():
        pred_df = read_table(args.predictions)
    else:
        pred_df = _load_predictions_from_db(args.postgres_config)

    summary_df, detail_df = evaluate_skill_gap_agreement(pred_df, labels_df, top_k=args.top_k)
    save_outputs(summary_df, args.output, detail_df)
    print("[DONE] Skill-gap agreement")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary: {Path(args.output)}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.common import (
    mean,
    normalize_text,
    normalize_token_set,
    parse_list,
    precision_recall_f1,
    read_table,
    save_outputs,
)


DEFAULT_LIST_FIELDS = [
    "skills",
    "projects",
    "professional_company_names",
    "degree_names",
]
DEFAULT_SCALAR_FIELDS = [
    "target_role",
    "experience_years",
]


def _parse_fields(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _find_key_column(df: pd.DataFrame) -> str:
    for col in ["cv_id", "cv_key", "file_name", "source_path"]:
        if col in df.columns:
            return col
    raise ValueError("Cannot find key column. Expected one of: cv_id, cv_key, file_name, source_path")


def _safe_scalar(value: Any) -> str:
    t = normalize_text(value)
    if t in {"", "unknown", "none", "nan", "null"}:
        return ""
    return t


def evaluate_field_accuracy(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    list_fields: list[str],
    scalar_fields: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_key = _find_key_column(predictions)
    gt_key = _find_key_column(ground_truth)

    pred = predictions.copy()
    gt = ground_truth.copy()
    pred["_key"] = pred[pred_key].astype(str).str.strip().str.lower()
    gt["_key"] = gt[gt_key].astype(str).str.strip().str.lower()
    merged = pred.merge(gt, on="_key", suffixes=("_pred", "_gt"), how="inner")
    if merged.empty:
        raise ValueError("No matching records between predictions and ground-truth based on key column.")

    detail_rows: list[dict[str, Any]] = []

    for _, row in merged.iterrows():
        key = row["_key"]
        for field in list_fields:
            pred_col = f"{field}_pred"
            gt_col = f"{field}_gt"
            pred_set = normalize_token_set(parse_list(row.get(pred_col)))
            gt_set = normalize_token_set(parse_list(row.get(gt_col)))
            p, r, f1 = precision_recall_f1(pred_set, gt_set)
            detail_rows.append(
                {
                    "key": key,
                    "field": field,
                    "field_type": "list",
                    "precision": p,
                    "recall": r,
                    "f1": f1,
                    "exact_match": 1.0 if pred_set == gt_set else 0.0,
                }
            )

        for field in scalar_fields:
            pred_col = f"{field}_pred"
            gt_col = f"{field}_gt"
            pred_value = _safe_scalar(row.get(pred_col))
            gt_value = _safe_scalar(row.get(gt_col))
            exact = 1.0 if pred_value == gt_value else 0.0
            detail_rows.append(
                {
                    "key": key,
                    "field": field,
                    "field_type": "scalar",
                    "precision": exact,
                    "recall": exact,
                    "f1": exact,
                    "exact_match": exact,
                }
            )

    detail_df = pd.DataFrame(detail_rows)
    summary_rows = []
    for field, g in detail_df.groupby("field"):
        summary_rows.append(
            {
                "field": field,
                "field_type": str(g["field_type"].iloc[0]),
                "precision_mean": mean(g["precision"].tolist()),
                "recall_mean": mean(g["recall"].tolist()),
                "f1_mean": mean(g["f1"].tolist()),
                "exact_match_rate": mean(g["exact_match"].tolist()),
                "samples": int(len(g)),
            }
        )

    overall = {
        "field": "__overall__",
        "field_type": "mixed",
        "precision_mean": mean(detail_df["precision"].tolist()),
        "recall_mean": mean(detail_df["recall"].tolist()),
        "f1_mean": mean(detail_df["f1"].tolist()),
        "exact_match_rate": mean(detail_df["exact_match"].tolist()),
        "samples": int(len(detail_df)),
    }
    summary_df = pd.DataFrame(summary_rows + [overall]).sort_values("field").reset_index(drop=True)
    return summary_df, detail_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CV extraction field-level accuracy.")
    parser.add_argument("--predictions", required=True, help="Extracted CV dataset (csv/parquet/jsonl/json).")
    parser.add_argument("--ground_truth", required=True, help="Human-labeled CV dataset.")
    parser.add_argument("--list_fields", default=",".join(DEFAULT_LIST_FIELDS), help="Comma-separated list fields.")
    parser.add_argument("--scalar_fields", default=",".join(DEFAULT_SCALAR_FIELDS), help="Comma-separated scalar fields.")
    parser.add_argument(
        "--output",
        default="experiments/artifacts/evaluation/cv_extraction_field_accuracy.csv",
        help="Output summary csv/parquet path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_df = read_table(args.predictions)
    gt_df = read_table(args.ground_truth)
    summary_df, detail_df = evaluate_field_accuracy(
        pred_df,
        gt_df,
        list_fields=_parse_fields(args.list_fields),
        scalar_fields=_parse_fields(args.scalar_fields),
    )
    save_outputs(summary_df, args.output, detail_df)
    print("[DONE] CV extraction field accuracy")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary: {Path(args.output)}")


if __name__ == "__main__":
    main()


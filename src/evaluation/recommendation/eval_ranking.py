from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluation.recommendation.metrics import (
    hit_rate_at_k,
    mean,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def _load_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    ext = p.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(p)
    if ext in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if ext == ".jsonl":
        rows: list[dict] = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    raise ValueError(f"Unsupported file extension: {ext}. Use csv/parquet/jsonl")


def _validate_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _parse_ks(raw: str) -> list[int]:
    ks = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        val = int(x)
        if val <= 0:
            raise ValueError("All K values must be positive integers.")
        ks.append(val)
    if not ks:
        raise ValueError("k_list is empty.")
    return sorted(set(ks))


def evaluate_ranking(
    predictions: pd.DataFrame,
    qrels: pd.DataFrame,
    *,
    ks: list[int],
) -> pd.DataFrame:
    # Normalize types
    predictions = predictions.copy()
    qrels = qrels.copy()
    predictions["cv_id"] = predictions["cv_id"].astype(int)
    predictions["job_id"] = predictions["job_id"].astype(int)
    qrels["cv_id"] = qrels["cv_id"].astype(int)
    qrels["job_id"] = qrels["job_id"].astype(int)
    qrels["relevance"] = qrels["relevance"].fillna(0).astype(float)

    if "method" not in predictions.columns:
        predictions["method"] = "default"

    # Build relevance map
    rel_map: dict[tuple[int, int], float] = {
        (int(r.cv_id), int(r.job_id)): float(r.relevance) for r in qrels.itertuples(index=False)
    }

    # Count relevant per cv
    relevant_count_by_cv = (
        qrels[qrels["relevance"] > 0]
        .groupby("cv_id")["job_id"]
        .nunique()
        .to_dict()
    )

    rows_out: list[dict] = []
    for method_name, method_df in predictions.groupby("method"):
        per_query: list[dict] = []

        # sort by (cv_id, rank asc) if rank exists, else score desc
        if "rank" in method_df.columns:
            method_df = method_df.sort_values(["cv_id", "rank"], ascending=[True, True])
        elif "score" in method_df.columns:
            method_df = method_df.sort_values(["cv_id", "score"], ascending=[True, False])
        else:
            raise ValueError("predictions must have `rank` or `score` column.")

        for cv_id, group in method_df.groupby("cv_id"):
            pred_job_ids = [int(x) for x in group["job_id"].tolist()]
            gains = [float(rel_map.get((int(cv_id), jid), 0.0)) for jid in pred_job_ids]
            relevance = [1 if g > 0 else 0 for g in gains]
            total_relevant = int(relevant_count_by_cv.get(int(cv_id), 0))

            one = {"cv_id": int(cv_id), "total_relevant": total_relevant}
            for k in ks:
                one[f"precision@{k}"] = precision_at_k(relevance, k)
                one[f"recall@{k}"] = recall_at_k(relevance, total_relevant, k)
                one[f"hitrate@{k}"] = hit_rate_at_k(relevance, k)
                one[f"mrr@{k}"] = mrr_at_k(relevance, k)
                one[f"ndcg@{k}"] = ndcg_at_k(gains, k)
            per_query.append(one)

        per_query_df = pd.DataFrame(per_query)
        if per_query_df.empty:
            continue

        summary = {"method": str(method_name), "num_queries": int(per_query_df["cv_id"].nunique())}
        for k in ks:
            for metric in ["precision", "recall", "hitrate", "mrr", "ndcg"]:
                col = f"{metric}@{k}"
                summary[col] = mean(per_query_df[col].tolist())
        rows_out.append(summary)

    return pd.DataFrame(rows_out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate recommendation ranking metrics.")
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions file (csv/parquet/jsonl). Required cols: cv_id, job_id, and rank|score; optional method.",
    )
    parser.add_argument(
        "--qrels",
        required=True,
        help="Path to qrels file (csv/parquet/jsonl). Required cols: cv_id, job_id, relevance.",
    )
    parser.add_argument(
        "--k_list",
        default="5,10",
        help="Comma-separated K values, e.g. 5,10",
    )
    parser.add_argument(
        "--output",
        default="experiments/artifacts/evaluation/recommendation_ranking_summary.csv",
        help="Output path for summary csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ks = _parse_ks(args.k_list)

    pred_df = _load_table(args.predictions)
    qrels_df = _load_table(args.qrels)
    _validate_columns(pred_df, ["cv_id", "job_id"], "predictions")
    if "rank" not in pred_df.columns and "score" not in pred_df.columns:
        raise ValueError("predictions must contain `rank` or `score`.")
    _validate_columns(qrels_df, ["cv_id", "job_id", "relevance"], "qrels")

    summary_df = evaluate_ranking(pred_df, qrels_df, ks=ks)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path, index=False, encoding="utf-8")

    if summary_df.empty:
        print("[WARN] No result rows. Check predictions/qrels input.")
        return

    print("[DONE] Ranking evaluation summary")
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()


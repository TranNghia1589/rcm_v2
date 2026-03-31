from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

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


def _normalize_predictions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "method" not in out.columns:
        out["method"] = "default"
    if "rank" in out.columns:
        out = out.sort_values(["method", "cv_id", "rank"], ascending=[True, True, True])
    elif "score" in out.columns:
        out = out.sort_values(["method", "cv_id", "score"], ascending=[True, True, False])
    else:
        raise ValueError("predictions must have `rank` or `score` column.")
    return out


def evaluate_coverage(
    predictions: pd.DataFrame,
    jobs_catalog: pd.DataFrame,
    *,
    ks: list[int],
) -> pd.DataFrame:
    pred = _normalize_predictions(predictions)
    job_total = int(jobs_catalog["job_id"].nunique())

    rows_out: list[dict] = []
    for method_name, mdf in pred.groupby("method"):
        summary = {"method": str(method_name), "num_queries": int(mdf["cv_id"].nunique())}
        for k in ks:
            topk = mdf.groupby("cv_id").head(k)

            # Catalog coverage: unique recommended jobs / total jobs
            unique_jobs = int(topk["job_id"].nunique())
            summary[f"catalog_coverage@{k}"] = (float(unique_jobs) / float(job_total)) if job_total > 0 else 0.0

        rows_out.append(summary)

    return pd.DataFrame(rows_out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate recommendation coverage/diversity metrics.")
    parser.add_argument("--predictions", required=True, help="Predictions file (csv/parquet/jsonl).")
    parser.add_argument(
        "--jobs_catalog",
        required=True,
        help="Jobs catalog file with at least job_id column (and optional job_family).",
    )
    parser.add_argument("--k_list", default="5,10", help="Comma-separated K values.")
    parser.add_argument(
        "--output",
        default="artifacts/evaluation/recommendation_coverage_summary.csv",
        help="Output path for summary csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ks = _parse_ks(args.k_list)
    pred_df = _load_table(args.predictions)
    jobs_df = _load_table(args.jobs_catalog)

    for col in ["cv_id", "job_id"]:
        if col not in pred_df.columns:
            raise ValueError(f"predictions missing required column: {col}")
    if "job_id" not in jobs_df.columns:
        raise ValueError("jobs_catalog missing required column: job_id")

    pred_df["cv_id"] = pred_df["cv_id"].astype(int)
    pred_df["job_id"] = pred_df["job_id"].astype(int)
    jobs_df["job_id"] = jobs_df["job_id"].astype(int)

    summary_df = evaluate_coverage(pred_df, jobs_df, ks=ks)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path, index=False, encoding="utf-8")

    print("[DONE] Coverage evaluation summary")
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

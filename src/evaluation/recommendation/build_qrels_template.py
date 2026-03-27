from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build qrels template from predictions file.")
    parser.add_argument(
        "--predictions",
        required=True,
        help="Predictions file path (csv/parquet) with columns: cv_id, job_id, method.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/evaluation/qrels_template.csv",
        help="Output qrels template path.",
    )
    return parser.parse_args()


def _read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("Unsupported file extension. Use csv/parquet.")


def main() -> None:
    args = parse_args()
    in_path = Path(args.predictions)
    if not in_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {in_path}")

    df = _read_table(in_path)
    required = {"cv_id", "job_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Predictions missing required columns: {missing}")

    if "method" not in df.columns:
        df["method"] = "default"

    grouped = (
        df.groupby(["cv_id", "job_id"], as_index=False)["method"]
        .apply(lambda s: ",".join(sorted(set(str(x) for x in s))))
    )
    grouped = grouped.rename(columns={"method": "method_hits"})
    grouped["relevance"] = ""
    grouped["notes"] = ""
    grouped = grouped[["cv_id", "job_id", "relevance", "method_hits", "notes"]]
    grouped = grouped.sort_values(["cv_id", "job_id"]).reset_index(drop=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out_path, index=False, encoding="utf-8")

    print(f"[DONE] Saved qrels template: {out_path}")
    print(f"rows={len(grouped)}, cv_count={grouped['cv_id'].nunique()}")
    print("Please fill `relevance` with values in {0,1,2,3}.")


if __name__ == "__main__":
    main()

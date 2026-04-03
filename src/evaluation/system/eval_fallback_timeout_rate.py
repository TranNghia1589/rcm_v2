from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.common import mean, read_table, save_outputs


def evaluate_fallback_timeout_rate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = df.copy()
    for col in ["used_fallback", "is_timeout", "latency_ms"]:
        if col not in data.columns:
            if col == "latency_ms":
                data[col] = 0.0
            else:
                data[col] = False
    if "status_code" not in data.columns:
        data["status_code"] = 0
    if "fallback_stage" not in data.columns:
        data["fallback_stage"] = ""

    data["used_fallback"] = data["used_fallback"].astype(bool).astype(float)
    data["is_timeout"] = data["is_timeout"].astype(bool).astype(float)
    data["is_error_5xx"] = data["status_code"].astype(int).map(lambda x: 1.0 if x >= 500 else 0.0)
    data["latency_ms"] = data["latency_ms"].astype(float)

    summary_df = pd.DataFrame(
        [
            {
                "samples": int(len(data)),
                "fallback_rate": mean(data["used_fallback"].tolist()),
                "timeout_rate": mean(data["is_timeout"].tolist()),
                "server_error_rate_5xx": mean(data["is_error_5xx"].tolist()),
                "avg_latency_ms": mean(data["latency_ms"].tolist()),
            }
        ]
    )

    by_stage = (
        data.groupby("fallback_stage", dropna=False)
        .agg(
            requests=("fallback_stage", "size"),
            fallback_rate=("used_fallback", "mean"),
            timeout_rate=("is_timeout", "mean"),
            avg_latency_ms=("latency_ms", "mean"),
        )
        .reset_index()
        .rename(columns={"fallback_stage": "stage"})
    )
    detail_df = by_stage
    return summary_df, detail_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fallback and timeout rates from request logs.")
    parser.add_argument(
        "--logs",
        required=True,
        help="CSV/parquet/jsonl with columns: used_fallback,is_timeout,status_code,latency_ms,fallback_stage.",
    )
    parser.add_argument(
        "--output",
        default="experiments/artifacts/evaluation/system_fallback_timeout_rate.csv",
        help="Output summary path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logs_df = read_table(args.logs)
    summary_df, detail_df = evaluate_fallback_timeout_rate(logs_df)
    save_outputs(summary_df, args.output, detail_df)
    print("[DONE] Fallback/timeout evaluation")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary: {Path(args.output)}")


if __name__ == "__main__":
    main()


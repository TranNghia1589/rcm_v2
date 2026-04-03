from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.common import mean, read_table, save_outputs


def _rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            out[indexed[k][0]] = avg_rank
        i = j + 1
    return out


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mx = mean(x)
    my = mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denx = math.sqrt(sum((a - mx) ** 2 for a in x))
    deny = math.sqrt(sum((b - my) ** 2 for b in y))
    if denx == 0 or deny == 0:
        return 0.0
    return num / (denx * deny)


def _spearman(x: list[float], y: list[float]) -> float:
    return _pearson(_rank(x), _rank(y))


def evaluate_calibration(df: pd.DataFrame, *, tolerance: float = 10.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"cv_id", "predicted_score", "human_score"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df.copy()
    data["cv_id"] = data["cv_id"].astype(int)
    data["predicted_score"] = data["predicted_score"].astype(float)
    data["human_score"] = data["human_score"].astype(float)
    data["abs_error"] = (data["predicted_score"] - data["human_score"]).abs()
    data["squared_error"] = (data["predicted_score"] - data["human_score"]) ** 2
    data["within_tolerance"] = (data["abs_error"] <= tolerance).astype(float)

    pred = data["predicted_score"].tolist()
    human = data["human_score"].tolist()
    mae = mean(data["abs_error"].tolist())
    rmse = math.sqrt(mean(data["squared_error"].tolist()))
    pearson_r = _pearson(pred, human)
    spearman_rho = _spearman(pred, human)

    summary_df = pd.DataFrame(
        [
            {
                "samples": int(len(data)),
                "mae": mae,
                "rmse": rmse,
                "pearson_r": pearson_r,
                "spearman_rho": spearman_rho,
                "within_tolerance_rate": mean(data["within_tolerance"].tolist()),
                "tolerance_threshold": float(tolerance),
            }
        ]
    )
    return summary_df, data[["cv_id", "predicted_score", "human_score", "abs_error", "within_tolerance"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CV score calibration against human labels.")
    parser.add_argument(
        "--labels",
        required=True,
        help="CSV/parquet/jsonl with columns: cv_id,predicted_score,human_score",
    )
    parser.add_argument("--tolerance", type=float, default=10.0, help="Absolute error tolerance.")
    parser.add_argument(
        "--output",
        default="experiments/artifacts/evaluation/cv_scoring_calibration.csv",
        help="Output summary path (csv/parquet).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_df = read_table(args.labels)
    summary_df, detail_df = evaluate_calibration(labels_df, tolerance=args.tolerance)
    save_outputs(summary_df, args.output, detail_df)
    print("[DONE] CV scoring calibration")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary: {Path(args.output)}")


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.evaluation.common import mean, percentile, read_table, save_outputs


def _send_request(base_url: str, row: dict[str, Any], timeout_sec: float) -> tuple[int, float, str]:
    method = str(row.get("method", "GET")).upper()
    path = str(row.get("path", "")).strip()
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    payload = row.get("json", None)
    params = row.get("params", None)
    if isinstance(payload, str) and payload.strip().startswith("{"):
        payload = json.loads(payload)
    if isinstance(params, str) and params.strip().startswith("{"):
        params = json.loads(params)

    t0 = time.perf_counter()
    try:
        resp = requests.request(method, url, json=payload, params=params, timeout=timeout_sec)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return int(resp.status_code), latency_ms, ""
    except requests.Timeout:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return 0, latency_ms, "timeout"
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return 0, latency_ms, str(exc)


def evaluate_api_latency_reliability(
    cases_df: pd.DataFrame,
    *,
    base_url: str,
    repeat: int = 1,
    timeout_sec: float = 30.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"path"}
    missing = sorted(required - set(cases_df.columns))
    if missing:
        raise ValueError(f"cases missing required columns: {missing}")

    detail_rows: list[dict[str, Any]] = []
    for _, case in cases_df.iterrows():
        row = case.to_dict()
        endpoint_name = str(row.get("name", row.get("path")))
        for i in range(repeat):
            status, latency_ms, error = _send_request(base_url, row, timeout_sec)
            success = 1.0 if 200 <= status < 300 else 0.0
            detail_rows.append(
                {
                    "endpoint": endpoint_name,
                    "method": str(row.get("method", "GET")).upper(),
                    "status_code": int(status),
                    "success": success,
                    "latency_ms": float(latency_ms),
                    "is_timeout": 1.0 if error == "timeout" else 0.0,
                    "error": error,
                    "iteration": int(i + 1),
                }
            )

    detail_df = pd.DataFrame(detail_rows)
    summary_rows = []
    for endpoint, g in detail_df.groupby("endpoint"):
        lats = g["latency_ms"].tolist()
        summary_rows.append(
            {
                "endpoint": str(endpoint),
                "requests": int(len(g)),
                "success_rate": mean(g["success"].tolist()),
                "timeout_rate": mean(g["is_timeout"].tolist()),
                "latency_ms_mean": mean(lats),
                "latency_ms_p50": percentile(lats, 50),
                "latency_ms_p95": percentile(lats, 95),
                "latency_ms_p99": percentile(lats, 99),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("endpoint").reset_index(drop=True)
    return summary_df, detail_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate API latency/reliability from endpoint test cases.")
    parser.add_argument(
        "--cases",
        required=True,
        help="CSV/parquet/jsonl with columns: path[,method,json,params,name].",
    )
    parser.add_argument("--base_url", default="http://127.0.0.1:8000")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout_sec", type=float, default=30.0)
    parser.add_argument(
        "--output",
        default="artifacts/evaluation/system_api_latency_reliability.csv",
        help="Output summary path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases_df = read_table(args.cases)
    summary_df, detail_df = evaluate_api_latency_reliability(
        cases_df,
        base_url=args.base_url,
        repeat=args.repeat,
        timeout_sec=args.timeout_sec,
    )
    save_outputs(summary_df, args.output, detail_df)
    print("[DONE] API latency/reliability evaluation")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary: {Path(args.output)}")


if __name__ == "__main__":
    main()

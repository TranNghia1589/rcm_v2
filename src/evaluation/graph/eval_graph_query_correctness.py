from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.common import mean, parse_list, read_table, save_outputs
from src.models.graph.query_service import GraphQueryService


def _run_case(service: GraphQueryService, case: dict[str, Any]) -> list[dict[str, Any]]:
    query_type = str(case.get("query", "")).strip().lower()
    cv_id = int(case.get("cv_id"))
    limit = int(case.get("limit", 10))
    if query_type == "recommend_jobs":
        return service.recommend_jobs(cv_id=cv_id, limit=limit)
    if query_type == "skill_gap":
        return service.user_skill_gap(cv_id=cv_id, limit=limit)
    if query_type == "career_path":
        return service.career_path(cv_id=cv_id, limit=limit)
    raise ValueError(f"Unsupported query type: {query_type}")


def _extract_actual_tokens(items: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for row in items:
        for value in row.values():
            if value is None:
                continue
            if isinstance(value, list):
                for x in value:
                    t = str(x).strip().lower()
                    if t:
                        out.add(t)
            else:
                t = str(value).strip().lower()
                if t:
                    out.add(t)
    return out


def evaluate_graph_query_correctness(cases_df: pd.DataFrame, *, neo4j_config: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"query", "cv_id"}
    missing = sorted(required - set(cases_df.columns))
    if missing:
        raise ValueError(f"cases missing required columns: {missing}")
    service = GraphQueryService(neo4j_cfg_path=neo4j_config)

    rows: list[dict[str, Any]] = []
    for idx, case in cases_df.reset_index(drop=True).iterrows():
        payload = case.to_dict()
        items = _run_case(service, payload)
        actual_tokens = _extract_actual_tokens(items)
        expected_contains = {str(x).strip().lower() for x in parse_list(payload.get("expected_contains")) if str(x).strip()}
        expected_not_contains = {str(x).strip().lower() for x in parse_list(payload.get("expected_not_contains")) if str(x).strip()}

        contain_ok = 1.0 if expected_contains.issubset(actual_tokens) else 0.0
        exclude_ok = 1.0 if actual_tokens.isdisjoint(expected_not_contains) else 0.0
        case_pass = 1.0 if (contain_ok == 1.0 and exclude_ok == 1.0) else 0.0
        rows.append(
            {
                "case_id": int(idx),
                "query": str(payload.get("query")),
                "cv_id": int(payload.get("cv_id")),
                "returned_rows": int(len(items)),
                "containment_pass": contain_ok,
                "exclusion_pass": exclude_ok,
                "case_pass": case_pass,
            }
        )
    detail_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(
        [
            {
                "cases": int(len(detail_df)),
                "containment_pass_rate": mean(detail_df["containment_pass"].tolist()),
                "exclusion_pass_rate": mean(detail_df["exclusion_pass"].tolist()),
                "overall_case_pass_rate": mean(detail_df["case_pass"].tolist()),
                "avg_returned_rows": mean(detail_df["returned_rows"].tolist()),
            }
        ]
    )
    return summary_df, detail_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate graph query correctness from labeled test cases.")
    parser.add_argument(
        "--cases",
        required=True,
        help="CSV/parquet/jsonl with columns: query,cv_id[,limit,expected_contains,expected_not_contains].",
    )
    parser.add_argument("--neo4j_config", default="config/db/neo4j.yaml")
    parser.add_argument(
        "--output",
        default="experiments/artifacts/evaluation/graph_query_correctness.csv",
        help="Output summary path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases_df = read_table(args.cases)
    summary_df, detail_df = evaluate_graph_query_correctness(cases_df, neo4j_config=args.neo4j_config)
    save_outputs(summary_df, args.output, detail_df)
    print("[DONE] Graph query correctness evaluation")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary: {Path(args.output)}")


if __name__ == "__main__":
    main()


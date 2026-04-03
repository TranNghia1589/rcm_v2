from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.common import mean, normalize_text, parse_list, read_table, save_outputs


def _fact_support_ratio(answer: str, supported_facts: list[str]) -> float:
    ans = normalize_text(answer)
    if not supported_facts:
        return 0.0
    hits = 0
    for fact in supported_facts:
        if normalize_text(fact) in ans:
            hits += 1
    return hits / len(supported_facts)


def evaluate_groundedness(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"answer", "supported_facts"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df.copy()
    if "used_fallback" not in data.columns:
        data["used_fallback"] = False
    if "retrieval_count" not in data.columns:
        data["retrieval_count"] = 0

    rows: list[dict[str, Any]] = []
    for idx, row in data.reset_index(drop=True).iterrows():
        answer = str(row.get("answer", ""))
        facts = parse_list(row.get("supported_facts"))
        faith = _fact_support_ratio(answer, facts)
        rows.append(
            {
                "case_id": int(idx),
                "faithfulness": faith,
                "hallucination_rate": 1.0 - faith,
                "used_fallback": 1.0 if bool(row.get("used_fallback")) else 0.0,
                "retrieval_count": float(row.get("retrieval_count", 0) or 0),
                "has_retrieval": 1.0 if float(row.get("retrieval_count", 0) or 0) > 0 else 0.0,
                "answer_length_chars": float(len(answer)),
            }
        )

    detail_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(
        [
            {
                "samples": int(len(detail_df)),
                "faithfulness_mean": mean(detail_df["faithfulness"].tolist()),
                "hallucination_rate_mean": mean(detail_df["hallucination_rate"].tolist()),
                "fallback_rate": mean(detail_df["used_fallback"].tolist()),
                "retrieval_usage_rate": mean(detail_df["has_retrieval"].tolist()),
                "avg_answer_length_chars": mean(detail_df["answer_length_chars"].tolist()),
            }
        ]
    )
    return summary_df, detail_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG groundedness from annotated chatbot outputs.")
    parser.add_argument(
        "--answers",
        required=True,
        help="CSV/parquet/jsonl with columns: answer,supported_facts[,used_fallback,retrieval_count].",
    )
    parser.add_argument(
        "--output",
        default="experiments/artifacts/evaluation/rag_groundedness.csv",
        help="Output summary path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    answers_df = read_table(args.answers)
    summary_df, detail_df = evaluate_groundedness(answers_df)
    save_outputs(summary_df, args.output, detail_df)
    print("[DONE] RAG groundedness evaluation")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary: {Path(args.output)}")


if __name__ == "__main__":
    main()


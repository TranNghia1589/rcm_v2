from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.common import mean, parse_list, read_table, save_outputs

_CIT_PATTERN = re.compile(r"\[chunk_id=(\d+)\]")


def _extract_cited_chunk_ids(answer: str) -> list[int]:
    return [int(x) for x in _CIT_PATTERN.findall(answer or "")]


def _parse_source_chunk_ids(raw: Any) -> set[int]:
    # supports source_chunk_ids=[1,2] OR sources=[{"chunk_id":1}, ...]
    if isinstance(raw, list):
        out: set[int] = set()
        for x in raw:
            if isinstance(x, dict) and "chunk_id" in x:
                try:
                    out.add(int(x["chunk_id"]))
                except Exception:
                    pass
            else:
                try:
                    out.add(int(str(x)))
                except Exception:
                    pass
        return out
    parsed = parse_list(raw)
    out: set[int] = set()
    for x in parsed:
        try:
            out.add(int(str(x)))
        except Exception:
            continue
    return out


def evaluate_citation_correctness(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "answer" not in df.columns:
        raise ValueError("Input missing required column: answer")

    data = df.copy()
    if "sources" not in data.columns and "source_chunk_ids" not in data.columns:
        data["source_chunk_ids"] = [[] for _ in range(len(data))]

    rows: list[dict[str, Any]] = []
    for idx, row in data.reset_index(drop=True).iterrows():
        answer = str(row.get("answer", ""))
        cited = _extract_cited_chunk_ids(answer)
        provided = _parse_source_chunk_ids(row.get("sources", row.get("source_chunk_ids")))
        cited_set = set(cited)
        valid = len(cited_set & provided)
        cited_count = len(cited_set)
        precision = (valid / cited_count) if cited_count > 0 else 0.0
        rows.append(
            {
                "case_id": int(idx),
                "citation_present": 1.0 if cited_count > 0 else 0.0,
                "cited_count": float(cited_count),
                "valid_cited_count": float(valid),
                "citation_valid_precision": precision,
            }
        )

    detail_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(
        [
            {
                "samples": int(len(detail_df)),
                "citation_presence_rate": mean(detail_df["citation_present"].tolist()),
                "citation_valid_precision_mean": mean(detail_df["citation_valid_precision"].tolist()),
                "avg_cited_count": mean(detail_df["cited_count"].tolist()),
                "avg_valid_cited_count": mean(detail_df["valid_cited_count"].tolist()),
            }
        ]
    )
    return summary_df, detail_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate citation correctness in grounded chatbot answers.")
    parser.add_argument(
        "--answers",
        required=True,
        help="CSV/parquet/jsonl with columns: answer and sources/source_chunk_ids.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/evaluation/rag_citation_correctness.csv",
        help="Output summary path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    answers_df = read_table(args.answers)
    summary_df, detail_df = evaluate_citation_correctness(answers_df)
    save_outputs(summary_df, args.output, detail_df)
    print("[DONE] Citation correctness evaluation")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary: {Path(args.output)}")


if __name__ == "__main__":
    main()

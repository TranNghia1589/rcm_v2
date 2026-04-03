from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.common import mean, parse_list, read_table, save_outputs
from src.models.rag.retrieve import retrieve_chunks


def _parse_ks(raw: str) -> list[int]:
    ks = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("k_list must contain positive integers.")
    return ks


def _to_int_set(value: Any) -> set[int]:
    out: set[int] = set()
    for x in parse_list(value):
        try:
            out.add(int(str(x).strip()))
        except Exception:
            continue
    return out


def evaluate_retrieval(
    cases_df: pd.DataFrame,
    *,
    postgres_config: str | Path,
    retrieval_config: str | Path,
    embedding_config: str | Path,
    ks: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"query", "relevant_chunk_ids"}
    missing = sorted(required - set(cases_df.columns))
    if missing:
        raise ValueError(f"Missing required columns in cases file: {missing}")

    max_k = max(ks)
    rows: list[dict[str, Any]] = []

    for idx, case in cases_df.reset_index(drop=True).iterrows():
        query = str(case["query"]).strip()
        relevant = _to_int_set(case["relevant_chunk_ids"])
        if not query:
            continue
        results = retrieve_chunks(
            question=query,
            postgres_config_path=postgres_config,
            retrieval_config_path=retrieval_config,
            embedding_config_path=embedding_config,
            top_k_override=max_k,
        )
        retrieved = [int(x["chunk_id"]) for x in results]
        for k in ks:
            topk = retrieved[:k]
            topk_set = set(topk)
            hits = len(topk_set & relevant)
            precision = hits / k if k > 0 else 0.0
            recall = hits / len(relevant) if relevant else 0.0
            mrr = 0.0
            for rank, cid in enumerate(topk, start=1):
                if cid in relevant:
                    mrr = 1.0 / rank
                    break
            hitrate = 1.0 if hits > 0 else 0.0
            rows.append(
                {
                    "case_id": int(idx),
                    "query": query,
                    "k": int(k),
                    "relevant_count": int(len(relevant)),
                    "retrieved_count": int(len(topk)),
                    "hits": int(hits),
                    "precision": precision,
                    "recall": recall,
                    "mrr": mrr,
                    "hit_rate": hitrate,
                }
            )

    detail_df = pd.DataFrame(rows)
    if detail_df.empty:
        raise ValueError("No retrieval results produced. Check cases and database.")
    summary_rows = []
    for k, g in detail_df.groupby("k"):
        summary_rows.append(
            {
                "k": int(k),
                "num_queries": int(g["case_id"].nunique()),
                "precision_mean": mean(g["precision"].tolist()),
                "recall_mean": mean(g["recall"].tolist()),
                "mrr_mean": mean(g["mrr"].tolist()),
                "hit_rate_mean": mean(g["hit_rate"].tolist()),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("k").reset_index(drop=True)
    return summary_df, detail_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality from labeled queries.")
    parser.add_argument(
        "--cases",
        required=True,
        help="Cases file (csv/parquet/jsonl) with columns: query, relevant_chunk_ids.",
    )
    parser.add_argument("--postgres_config", default="config/db/postgres.yaml")
    parser.add_argument("--retrieval_config", default="config/rag/retrieval.yaml")
    parser.add_argument("--embedding_config", default="config/model/embedding.yaml")
    parser.add_argument("--k_list", default="5,10")
    parser.add_argument(
        "--output",
        default="experiments/artifacts/evaluation/rag_retrieval.csv",
        help="Output summary path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases_df = read_table(args.cases)
    summary_df, detail_df = evaluate_retrieval(
        cases_df,
        postgres_config=args.postgres_config,
        retrieval_config=args.retrieval_config,
        embedding_config=args.embedding_config,
        ks=_parse_ks(args.k_list),
    )
    save_outputs(summary_df, args.output, detail_df)
    print("[DONE] RAG retrieval evaluation")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary: {Path(args.output)}")


if __name__ == "__main__":
    main()


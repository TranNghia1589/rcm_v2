from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.infrastructure.db.pgvector_store import PgVectorStore
from src.infrastructure.db.postgres_client import PostgresClient, PostgresConfig
from src.rag.chunking import ChunkingConfig, split_text
from src.rag.embed import create_embedder_from_yaml
from src.rag.index import index_job_documents



def _pick(row: pd.Series, keys: list[str], default: str = "") -> str:
    for key in keys:
        if key in row and pd.notna(row[key]):
            value = str(row[key]).strip()
            if value:
                return value
    return default


def _load_jobs(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _rows_from_section_table(df: pd.DataFrame) -> list[dict[str, Any]]:
    group_cols = [c for c in ["job_url", "job_title_display", "company_name_raw", "location_norm"] if c in df.columns]
    if not group_cols:
        return []

    grouped = df.groupby(group_cols, dropna=False, sort=False)
    records: list[dict[str, Any]] = []
    for group_keys, group_df in grouped:
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)

        key_map = dict(zip(group_cols, group_keys))
        source_id = str(key_map.get("job_url", "")).strip()
        if not source_id:
            source_id = f"job_{len(records)+1}"

        chunks = []
        if "chunk_text_raw" in group_df.columns:
            chunks = [str(x).strip() for x in group_df["chunk_text_raw"].tolist() if str(x).strip()]
        elif "chunk_text_phobert" in group_df.columns:
            chunks = [str(x).strip() for x in group_df["chunk_text_phobert"].tolist() if str(x).strip()]

        title = str(key_map.get("job_title_display", "") or "")
        body = "\n".join(chunks)
        records.append(
            {
                "source_id": source_id,
                "title": title,
                "body": body,
                "chunks": chunks,
                "metadata": {
                    "company_name": str(key_map.get("company_name_raw", "") or ""),
                    "location": str(key_map.get("location_norm", "") or ""),
                    "ingest_source": "jobs_chatbot_sections",
                },
            }
        )
    return records


def _rows_from_job_table(df: pd.DataFrame, chunk_config: ChunkingConfig) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        source_id = _pick(row, ["job_url", "external_job_id", "job_id"])
        if not source_id:
            continue

        title = _pick(row, ["job_title_display", "job_title_clean", "job_title_raw"], "Unknown")
        body = _pick(
            row,
            ["job_text_phobert_chatbot", "job_text_sparse", "requirements_clean", "description_clean"],
        )
        if not body:
            continue

        chunks = split_text(body, chunk_config)
        records.append(
            {
                "source_id": source_id,
                "title": title,
                "body": body,
                "chunks": chunks,
                "metadata": {
                    "company_name": _pick(row, ["company_name_raw", "company_name"]),
                    "location": _pick(row, ["location_norm", "location_normalized", "location_raw"]),
                    "salary": _pick(row, ["salary_raw"]),
                    "job_family": _pick(row, ["job_family"]),
                    "seniority": _pick(row, ["seniority_final"]),
                    "ingest_source": "jobs_matching_or_chatbot",
                },
            }
        )
    return records


def build_records(df: pd.DataFrame, chunk_config: ChunkingConfig) -> list[dict[str, Any]]:
    if "chunk_text_raw" in df.columns or "chunk_text_phobert" in df.columns:
        return _rows_from_section_table(df)
    return _rows_from_job_table(df, chunk_config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest jobs into RAG tables with pgvector embeddings.")
    parser.add_argument(
        "--jobs_path",
        default=str(BASE_DIR / "artifacts" / "matching" / "jobs_chatbot_sections_v3.parquet"),
        help="Path to source jobs parquet/csv.",
    )
    parser.add_argument(
        "--postgres_config",
        default=str(BASE_DIR / "configs" / "db" / "postgres.yaml"),
        help="Path to postgres yaml config.",
    )
    parser.add_argument(
        "--pgvector_config",
        default=str(BASE_DIR / "configs" / "db" / "pgvector.yaml"),
        help="Path to pgvector yaml config.",
    )
    parser.add_argument(
        "--chunking_config",
        default=str(BASE_DIR / "configs" / "rag" / "chunking.yaml"),
        help="Path to chunking yaml config.",
    )
    parser.add_argument(
        "--embedding_config",
        default=str(BASE_DIR / "configs" / "model" / "embedding.yaml"),
        help="Path to embedding yaml config.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for quick test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    jobs_path = Path(args.jobs_path)
    if not jobs_path.exists():
        raise FileNotFoundError(f"Jobs input not found: {jobs_path}")

    chunk_config = ChunkingConfig.from_yaml(args.chunking_config)
    pg_conf = PostgresConfig.from_yaml(args.postgres_config)
    _ = Path(args.pgvector_config)  # reserved for future vector settings

    df = _load_jobs(jobs_path)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    records = build_records(df, chunk_config)
    if not records:
        print("No records to ingest.")
        return

    embedder = create_embedder_from_yaml(args.embedding_config)
    with PostgresClient(pg_conf) as client:
        store = PgVectorStore(client)
        stats = index_job_documents(
            records=records,
            vector_store=store,
            embedder=embedder,
            chunk_config=chunk_config,
            source_type="job",
        )

    print(
        f"[DONE] Indexed documents={stats.documents}, chunks={stats.chunks}, embeddings={stats.embeddings}"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.infrastructure.db.pgvector_store import PgVectorStore
from src.infrastructure.db.postgres_client import PostgresClient, PostgresConfig
from src.rag.embed import create_embedder_from_yaml


@dataclass
class RetrievalConfig:
    top_k: int = 10
    source_type: str = "job"
    max_preview_chars: int = 260

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RetrievalConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(
            top_k=int(raw.get("top_k", 10)),
            source_type=str(raw.get("source_type", "job")),
            max_preview_chars=int(raw.get("max_preview_chars", 260)),
        )


def retrieve_chunks(
    *,
    question: str,
    postgres_config_path: str | Path,
    retrieval_config_path: str | Path,
    embedding_config_path: str | Path = BASE_DIR / "configs" / "model" / "embedding.yaml",
    top_k_override: int | None = None,
    embedder: Any | None = None,
) -> list[dict[str, Any]]:
    pg_conf = PostgresConfig.from_yaml(postgres_config_path)
    retrieval_conf = RetrievalConfig.from_yaml(retrieval_config_path)
    active_embedder = embedder or create_embedder_from_yaml(embedding_config_path)
    qvec = active_embedder.embed_text(question)
    top_k = top_k_override if top_k_override is not None else retrieval_conf.top_k

    with PostgresClient(pg_conf) as client:
        store = PgVectorStore(client)
        return store.search_chunks(
            query_embedding=qvec,
            top_k=top_k,
            source_type=retrieval_conf.source_type or None,
        )


def _preview(text: str, max_chars: int) -> str:
    t = (text or "").replace("\n", " ").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 3] + "..."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic retrieval from pgvector.")
    parser.add_argument("--question", required=True, help="User question")
    parser.add_argument(
        "--postgres_config",
        default=str(BASE_DIR / "configs" / "db" / "postgres.yaml"),
        help="Path to postgres yaml config.",
    )
    parser.add_argument(
        "--retrieval_config",
        default=str(BASE_DIR / "configs" / "rag" / "retrieval.yaml"),
        help="Path to retrieval yaml config.",
    )
    parser.add_argument(
        "--embedding_config",
        default=str(BASE_DIR / "configs" / "model" / "embedding.yaml"),
        help="Path to embedding yaml config.",
    )
    parser.add_argument("--top_k", type=int, default=0, help="Optional top-k override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = retrieve_chunks(
        question=args.question,
        postgres_config_path=args.postgres_config,
        retrieval_config_path=args.retrieval_config,
        embedding_config_path=args.embedding_config,
        top_k_override=args.top_k if args.top_k > 0 else None,
    )

    cfg = RetrievalConfig.from_yaml(args.retrieval_config)
    print(f"\n[QUESTION] {args.question}")
    print(f"[RESULTS] {len(results)} chunks\n")
    for idx, item in enumerate(results, 1):
        print(
            f"{idx}. chunk_id={item['chunk_id']} | doc_id={item['document_id']} | "
            f"distance={item['distance']:.6f} | title={item['title']}"
        )
        print(f"   preview: {_preview(item['content'], cfg.max_preview_chars)}")


if __name__ == "__main__":
    main()

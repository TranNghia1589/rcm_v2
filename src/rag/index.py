from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.infrastructure.db.pgvector_store import (
    ChunkPayload,
    DocumentPayload,
    PgVectorStore,
)
from src.rag.chunking import ChunkingConfig, estimate_token_count, split_text


@dataclass
class IndexStats:
    documents: int = 0
    chunks: int = 0
    embeddings: int = 0


def index_job_documents(
    *,
    records: list[dict[str, Any]],
    vector_store: PgVectorStore,
    embedder: Any,
    chunk_config: ChunkingConfig,
    source_type: str = "job",
) -> IndexStats:
    stats = IndexStats()

    for record in records:
        source_id = str(record.get("source_id", "")).strip()
        if not source_id:
            continue

        title = str(record.get("title", "") or "").strip()
        body = str(record.get("body", "") or "").strip()
        if not body:
            continue

        document_id = vector_store.upsert_document(
            DocumentPayload(
                source_type=source_type,
                source_id=source_id,
                title=title,
                body=body,
                metadata=record.get("metadata", {}) or {},
            )
        )
        stats.documents += 1

        raw_chunks: list[str]
        if record.get("chunks"):
            raw_chunks = [str(x).strip() for x in record["chunks"] if str(x).strip()]
        else:
            raw_chunks = split_text(body, chunk_config)

        chunk_payloads: list[ChunkPayload] = []
        for idx, content in enumerate(raw_chunks):
            chunk_payloads.append(
                ChunkPayload(
                    chunk_index=idx,
                    content=content,
                    token_count=estimate_token_count(content),
                    metadata={
                        **(record.get("metadata", {}) or {}),
                        "chunk_index": idx,
                    },
                )
            )

        if not chunk_payloads:
            continue

        chunk_ids = vector_store.replace_document_chunks(document_id, chunk_payloads)
        stats.chunks += len(chunk_ids)

        embeddings = embedder.embed_texts([c.content for c in chunk_payloads])
        embedding_model_name = str(getattr(embedder, "model_name", "unknown-embedder"))
        for chunk_id, emb in zip(chunk_ids, embeddings):
            vector_store.upsert_chunk_embedding(
                chunk_id=chunk_id,
                embedding=emb,
                embedding_model=embedding_model_name,
            )
        stats.embeddings += len(embeddings)

    return stats

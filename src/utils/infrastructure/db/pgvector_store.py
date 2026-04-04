from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.utils.infrastructure.db.postgres_client import PostgresClient


def _to_vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"


@dataclass
class DocumentPayload:
    source_type: str
    source_id: str
    title: str
    body: str
    metadata: dict[str, Any]


@dataclass
class ChunkPayload:
    chunk_index: int
    content: str
    token_count: int
    metadata: dict[str, Any]


class PgVectorStore:
    def __init__(self, client: PostgresClient) -> None:
        self.client = client

    def upsert_document(self, payload: DocumentPayload) -> int:
        existing = self.client.fetch_one(
            """
            SELECT document_id
            FROM rag_documents
            WHERE source_type = %s AND source_id = %s
            ORDER BY document_id DESC
            LIMIT 1
            """,
            (payload.source_type, payload.source_id),
        )
        metadata_json = json.dumps(payload.metadata, ensure_ascii=False)
        if existing:
            document_id = int(existing[0])
            self.client.execute(
                """
                UPDATE rag_documents
                SET title = %s,
                    body = %s,
                    metadata = %s::jsonb
                WHERE document_id = %s
                """,
                (payload.title, payload.body, metadata_json, document_id),
            )
            return document_id

        row = self.client.fetch_one(
            """
            INSERT INTO rag_documents (source_type, source_id, title, body, metadata)
            VALUES (%s, %s, %s, %s, %s::jsonb)
            RETURNING document_id
            """,
            (
                payload.source_type,
                payload.source_id,
                payload.title,
                payload.body,
                metadata_json,
            ),
        )
        return int(row[0])

    def replace_document_chunks(self, document_id: int, chunks: list[ChunkPayload]) -> list[int]:
        self.client.execute(
            "DELETE FROM rag_chunks WHERE document_id = %s",
            (document_id,),
        )
        chunk_ids: list[int] = []
        for chunk in chunks:
            row = self.client.fetch_one(
                """
                INSERT INTO rag_chunks (document_id, chunk_index, content, token_count, metadata)
                VALUES (%s, %s, %s, %s, %s::jsonb)
                RETURNING chunk_id
                """,
                (
                    document_id,
                    chunk.chunk_index,
                    chunk.content,
                    chunk.token_count,
                    json.dumps(chunk.metadata, ensure_ascii=False),
                ),
            )
            chunk_ids.append(int(row[0]))
        return chunk_ids

    def upsert_chunk_embedding(
        self,
        *,
        chunk_id: int,
        embedding: list[float],
        embedding_model: str,
    ) -> None:
        self.client.execute(
            """
            INSERT INTO rag_embeddings (chunk_id, embedding, embedding_model)
            VALUES (%s, (%s)::vector, %s)
            ON CONFLICT (chunk_id)
            DO UPDATE SET
                embedding = EXCLUDED.embedding,
                embedding_model = EXCLUDED.embedding_model
            """,
            (chunk_id, _to_vector_literal(embedding), embedding_model),
        )

    def search_chunks(
        self,
        *,
        query_embedding: list[float],
        top_k: int = 10,
        source_type: str | None = None,
    ) -> list[dict[str, Any]]:
        if source_type:
            rows = self.client.fetch_all(
                """
                SELECT
                    rc.chunk_id,
                    rd.document_id,
                    rd.source_id,
                    rd.title,
                    rc.content,
                    (re.embedding <=> (%s)::vector) AS distance
                FROM rag_embeddings re
                JOIN rag_chunks rc ON rc.chunk_id = re.chunk_id
                JOIN rag_documents rd ON rd.document_id = rc.document_id
                WHERE rd.source_type = %s
                ORDER BY re.embedding <=> (%s)::vector
                LIMIT %s
                """,
                (
                    _to_vector_literal(query_embedding),
                    source_type,
                    _to_vector_literal(query_embedding),
                    top_k,
                ),
            )
        else:
            rows = self.client.fetch_all(
                """
                SELECT
                    rc.chunk_id,
                    rd.document_id,
                    rd.source_id,
                    rd.title,
                    rc.content,
                    (re.embedding <=> (%s)::vector) AS distance
                FROM rag_embeddings re
                JOIN rag_chunks rc ON rc.chunk_id = re.chunk_id
                JOIN rag_documents rd ON rd.document_id = rc.document_id
                ORDER BY re.embedding <=> (%s)::vector
                LIMIT %s
                """,
                (
                    _to_vector_literal(query_embedding),
                    _to_vector_literal(query_embedding),
                    top_k,
                ),
            )
        out: list[dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "chunk_id": int(r[0]),
                    "document_id": int(r[1]),
                    "source_id": str(r[2] or ""),
                    "title": str(r[3] or ""),
                    "content": str(r[4] or ""),
                    "distance": float(r[5]),
                }
            )
        return out

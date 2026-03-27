from __future__ import annotations

from pathlib import Path

from src.infrastructure.db.postgres_client import PostgresClient, PostgresConfig
from src.rag.retrieve import retrieve_chunks


def _score_from_distance(distance: float) -> float:
    # Smaller distance means better similarity.
    # Convert to a bounded score in [0, 1].
    return 1.0 / (1.0 + max(0.0, float(distance)))


def vector_candidates(
    *,
    question: str,
    postgres_config_path: str | Path,
    retrieval_config_path: str | Path,
    embedding_config_path: str | Path,
    top_k: int,
) -> list[dict]:
    chunks = retrieve_chunks(
        question=question,
        postgres_config_path=postgres_config_path,
        retrieval_config_path=retrieval_config_path,
        embedding_config_path=embedding_config_path,
        top_k_override=top_k,
    )
    if not chunks:
        return []

    pg_cfg = PostgresConfig.from_yaml(postgres_config_path)
    aggregated: dict[int, dict] = {}
    with PostgresClient(pg_cfg) as client:
        for c in chunks:
            row = client.fetch_one(
                """
                SELECT j.job_id, j.title, j.company_name, j.location, rd.source_id
                FROM rag_chunks rc
                JOIN rag_documents rd ON rd.document_id = rc.document_id
                LEFT JOIN jobs j ON j.job_url = rd.source_id
                WHERE rc.chunk_id = %s
                LIMIT 1
                """,
                (int(c["chunk_id"]),),
            )
            if not row:
                continue
            job_id = row[0]
            if job_id is None:
                continue
            job_id = int(job_id)
            score = _score_from_distance(float(c.get("distance", 1.0)))
            item = aggregated.get(job_id)
            if not item:
                aggregated[job_id] = {
                    "job_id": job_id,
                    "title": str(row[1] or c.get("title", "")),
                    "company_name": str(row[2] or ""),
                    "location": str(row[3] or ""),
                    "vector_score": score,
                    "supporting_chunk_ids": [int(c["chunk_id"])],
                }
            else:
                item["vector_score"] = max(float(item["vector_score"]), score)
                item["supporting_chunk_ids"].append(int(c["chunk_id"]))

    out = list(aggregated.values())
    out.sort(key=lambda x: x["vector_score"], reverse=True)
    return out

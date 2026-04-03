from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os
import yaml

from src.models.rag.embed import create_embedder_from_yaml
from src.models.rag.embedding_service_client import EmbeddingServiceClient
from src.models.rag.retrieve import retrieve_chunks


@dataclass
class RetrievalService:
    postgres_config_path: str | Path
    retrieval_config_path: str | Path
    embedding_config_path: str | Path
    embedding_service_url: str | None = None
    embedding_service_config_path: str | Path | None = None

    def search(
        self,
        *,
        question: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if not question or not question.strip():
            return []
        service_url = (self.embedding_service_url or os.getenv("EMBEDDING_SERVICE_URL", "")).strip()
        if not service_url and self.embedding_service_config_path:
            try:
                raw = yaml.safe_load(Path(self.embedding_service_config_path).read_text(encoding="utf-8")) or {}
                service_url = str(raw.get("url", "")).strip()
            except Exception:
                service_url = ""
        if service_url:
            query_embedding = EmbeddingServiceClient(service_url).embed_text(question.strip())
            return retrieve_chunks(
                question=question.strip(),
                postgres_config_path=self.postgres_config_path,
                retrieval_config_path=self.retrieval_config_path,
                embedding_config_path=self.embedding_config_path,
                query_embedding=query_embedding,
                top_k_override=top_k,
            )
        return retrieve_chunks(
            question=question.strip(),
            postgres_config_path=self.postgres_config_path,
            retrieval_config_path=self.retrieval_config_path,
            embedding_config_path=self.embedding_config_path,
            embedder=create_embedder_from_yaml(self.embedding_config_path),
            top_k_override=top_k,
        )


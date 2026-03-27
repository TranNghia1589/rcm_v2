from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.rag.embed import create_embedder_from_yaml
from src.rag.retrieve import retrieve_chunks


@dataclass
class RetrievalService:
    postgres_config_path: str | Path
    retrieval_config_path: str | Path
    embedding_config_path: str | Path

    def search(
        self,
        *,
        question: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if not question or not question.strip():
            return []
        return retrieve_chunks(
            question=question.strip(),
            postgres_config_path=self.postgres_config_path,
            retrieval_config_path=self.retrieval_config_path,
            embedding_config_path=self.embedding_config_path,
            embedder=create_embedder_from_yaml(self.embedding_config_path),
            top_k_override=top_k,
        )

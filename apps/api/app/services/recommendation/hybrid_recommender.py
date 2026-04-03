from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.models.recommendation.orchestrator import run_hybrid_recommendation


@dataclass
class HybridRecommenderService:
    postgres_config_path: str | Path
    neo4j_config_path: str | Path
    retrieval_config_path: str | Path
    embedding_config_path: str | Path
    hybrid_config_path: str | Path

    def recommend(self, *, question: str, cv_id: int) -> dict:
        return run_hybrid_recommendation(
            question=question,
            cv_id=cv_id,
            postgres_config_path=self.postgres_config_path,
            neo4j_config_path=self.neo4j_config_path,
            retrieval_config_path=self.retrieval_config_path,
            embedding_config_path=self.embedding_config_path,
            hybrid_config_path=self.hybrid_config_path,
        )


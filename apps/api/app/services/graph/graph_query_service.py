from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.models.graph.query_service import GraphQueryService as CoreGraphQueryService


@dataclass
class GraphQueryAppService:
    neo4j_config_path: str | Path

    def _core(self) -> CoreGraphQueryService:
        return CoreGraphQueryService(self.neo4j_config_path)

    def recommend_jobs(self, cv_id: int, limit: int = 10) -> list[dict]:
        return self._core().recommend_jobs(cv_id=cv_id, limit=limit)

    def skill_gap(self, cv_id: int, limit: int = 15) -> list[dict]:
        return self._core().user_skill_gap(cv_id=cv_id, limit=limit)

    def career_path(self, cv_id: int, limit: int = 5) -> list[dict]:
        return self._core().career_path(cv_id=cv_id, limit=limit)


from __future__ import annotations

from pathlib import Path
from typing import Any

from src.infrastructure.db.neo4j_client import Neo4jClient, Neo4jConfig


class GraphQueryService:
    def __init__(self, neo4j_cfg_path: str | Path) -> None:
        self.neo4j_cfg_path = neo4j_cfg_path

    def _run_query_file(self, file_path: Path, params: dict[str, Any]) -> list[dict[str, Any]]:
        if not file_path.exists():
            raise FileNotFoundError(f"Cypher file not found: {file_path}")
        query = file_path.read_text(encoding="utf-8")
        cfg = Neo4jConfig.from_yaml(self.neo4j_cfg_path)
        with Neo4jClient(cfg) as neo:
            return neo.run(query, params)

    def recommend_jobs(self, cv_id: int, limit: int = 10) -> list[dict[str, Any]]:
        root = Path(__file__).resolve().parents[2]
        q = root / "database" / "neo4j" / "queries" / "recommend_jobs.cypher"
        return self._run_query_file(q, {"cv_id": int(cv_id), "limit": int(limit)})

    def user_skill_gap(self, cv_id: int, limit: int = 15) -> list[dict[str, Any]]:
        root = Path(__file__).resolve().parents[2]
        q = root / "database" / "neo4j" / "queries" / "user_skill_gap.cypher"
        return self._run_query_file(q, {"cv_id": int(cv_id), "limit": int(limit)})

    def career_path(self, cv_id: int, limit: int = 5) -> list[dict[str, Any]]:
        root = Path(__file__).resolve().parents[2]
        q = root / "database" / "neo4j" / "queries" / "career_path.cypher"
        return self._run_query_file(q, {"cv_id": int(cv_id), "limit": int(limit)})

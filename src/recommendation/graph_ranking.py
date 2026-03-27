from __future__ import annotations

from pathlib import Path

from src.graph.query_service import GraphQueryService


def graph_candidates(
    *,
    cv_id: int,
    neo4j_config_path: str | Path,
    top_k: int,
) -> list[dict]:
    service = GraphQueryService(neo4j_cfg_path=neo4j_config_path)
    rows = service.recommend_jobs(cv_id=cv_id, limit=top_k)
    out: list[dict] = []
    for r in rows:
        out.append(
            {
                "job_id": int(r["job_id"]),
                "title": str(r.get("title", "")),
                "company_name": str(r.get("company_name", "")),
                "location": str(r.get("location", "")),
                "matched_skills": int(r.get("matched_skills", 0)),
                "total_required": int(r.get("total_required", 0)),
                "coverage": float(r.get("coverage", 0.0)),
                "graph_score": float(r.get("score", 0.0)),
            }
        )
    return out

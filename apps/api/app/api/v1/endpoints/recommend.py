from __future__ import annotations

# Backward-compat wrapper. Canonical implementation lives in apps.api.app.api.v1.recommendations.
from apps.api.app.api.v1.recommendations import (
    career_path_graph,
    get_hybrid_service,
    recommend_hybrid,
    recommend_jobs_graph,
    recommend_jobs_hybrid,
    router,
    skill_gap_graph,
)

__all__ = [
    "router",
    "get_hybrid_service",
    "recommend_jobs_graph",
    "skill_gap_graph",
    "career_path_graph",
    "recommend_jobs_hybrid",
    "recommend_hybrid",
]

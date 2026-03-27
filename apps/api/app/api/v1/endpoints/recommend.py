from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException

from apps.api.app.schemas.graph.subgraph import (
    GraphCareerPathItem,
    GraphCareerPathRequest,
    GraphCareerPathResponse,
    GraphSkillGapItem,
    GraphSkillGapRequest,
    GraphSkillGapResponse,
)
from apps.api.app.schemas.recommendation.job import (
    GraphJobRecommendation,
    GraphRecommendRequest,
    GraphRecommendResponse,
    HybridRecommendRequest,
    HybridRecommendResponse,
    HybridRecommendationItem,
)
from apps.api.app.services.graph.graph_query_service import GraphQueryAppService
from apps.api.app.services.recommendation.hybrid_recommender import HybridRecommenderService


router = APIRouter(prefix="/recommend", tags=["recommendation"])

BASE_DIR = Path(__file__).resolve().parents[6]
NEO4J_CFG = BASE_DIR / "configs" / "db" / "neo4j.yaml"
POSTGRES_CFG = BASE_DIR / "configs" / "db" / "postgres.yaml"
RETRIEVAL_CFG = BASE_DIR / "configs" / "rag" / "retrieval.yaml"
EMBEDDING_CFG = BASE_DIR / "configs" / "model" / "embedding.yaml"
HYBRID_CFG = BASE_DIR / "configs" / "recommendation" / "hybrid.yaml"
@lru_cache(maxsize=1)
def get_hybrid_service() -> HybridRecommenderService:
    return HybridRecommenderService(
        postgres_config_path=POSTGRES_CFG,
        neo4j_config_path=NEO4J_CFG,
        retrieval_config_path=RETRIEVAL_CFG,
        embedding_config_path=EMBEDDING_CFG,
        hybrid_config_path=HYBRID_CFG,
    )


@router.post("/graph/jobs", response_model=GraphRecommendResponse)
def recommend_jobs_graph(payload: GraphRecommendRequest) -> GraphRecommendResponse:
    try:
        service = GraphQueryAppService(neo4j_config_path=NEO4J_CFG)
        rows = service.recommend_jobs(cv_id=payload.cv_id, limit=payload.limit)
        items = [GraphJobRecommendation(**r) for r in rows]
        return GraphRecommendResponse(cv_id=payload.cv_id, items=items)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/graph/skill-gap", response_model=GraphSkillGapResponse)
def skill_gap_graph(payload: GraphSkillGapRequest) -> GraphSkillGapResponse:
    try:
        service = GraphQueryAppService(neo4j_config_path=NEO4J_CFG)
        rows = service.skill_gap(cv_id=payload.cv_id, limit=payload.limit)
        items = [GraphSkillGapItem(**r) for r in rows]
        return GraphSkillGapResponse(cv_id=payload.cv_id, items=items)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/graph/career-path", response_model=GraphCareerPathResponse)
def career_path_graph(payload: GraphCareerPathRequest) -> GraphCareerPathResponse:
    try:
        service = GraphQueryAppService(neo4j_config_path=NEO4J_CFG)
        rows = service.career_path(cv_id=payload.cv_id, limit=payload.limit)
        items = [GraphCareerPathItem(**r) for r in rows]
        return GraphCareerPathResponse(cv_id=payload.cv_id, items=items)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/hybrid/jobs", response_model=HybridRecommendResponse)
def recommend_jobs_hybrid(payload: HybridRecommendRequest) -> HybridRecommendResponse:
    try:
        out = get_hybrid_service().recommend(question=payload.question, cv_id=payload.cv_id)
        items = [HybridRecommendationItem(**x) for x in out.get("items", [])]
        return HybridRecommendResponse(
            cv_id=int(out["cv_id"]),
            question=str(out["question"]),
            items=items,
            explanation=str(out.get("explanation", "")),
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

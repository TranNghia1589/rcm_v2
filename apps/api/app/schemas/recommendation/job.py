from __future__ import annotations

from pydantic import BaseModel, Field


class GraphRecommendRequest(BaseModel):
    cv_id: int = Field(..., ge=1)
    limit: int = Field(default=10, ge=1, le=50)


class GraphJobRecommendation(BaseModel):
    job_id: int
    title: str
    company_name: str
    location: str
    matched_skills: int
    total_required: int
    coverage: float
    score: float


class GraphRecommendResponse(BaseModel):
    cv_id: int
    items: list[GraphJobRecommendation]


class HybridRecommendRequest(BaseModel):
    cv_id: int = Field(..., ge=1)
    question: str = Field(..., min_length=3)


class HybridRecommendationItem(BaseModel):
    job_id: int
    title: str
    company_name: str
    location: str
    matched_skills: int = 0
    total_required: int = 0
    coverage: float = 0.0
    vector_score: float = 0.0
    graph_score: float = 0.0
    hybrid_score: float = 0.0
    supporting_chunk_ids: list[int] = []


class HybridRecommendResponse(BaseModel):
    cv_id: int
    question: str
    items: list[HybridRecommendationItem]
    explanation: str = ""

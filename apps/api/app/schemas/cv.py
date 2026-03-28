from __future__ import annotations

from pydantic import BaseModel, Field


class CVScoreResponse(BaseModel):
    cv_id: int = Field(..., ge=1)
    total_score: float
    grade: str
    benchmark_role: str = ""
    skill_score: float
    experience_score: float
    project_score: float
    education_score: float
    completeness_score: float
    strengths: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    priority_skills: list[str] = Field(default_factory=list)
    development_plan_30_60_90: dict = Field(default_factory=dict)
    subscores_json: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
    model_version: str = "cv_scoring_v1"
    updated_at: str

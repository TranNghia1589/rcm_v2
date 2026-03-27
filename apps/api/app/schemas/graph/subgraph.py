from __future__ import annotations

from pydantic import BaseModel, Field


class GraphSkillGapRequest(BaseModel):
    cv_id: int = Field(..., ge=1)
    limit: int = Field(default=15, ge=1, le=100)


class GraphSkillGapItem(BaseModel):
    missing_skill: str
    freq: int


class GraphSkillGapResponse(BaseModel):
    cv_id: int
    items: list[GraphSkillGapItem]


class GraphCareerPathRequest(BaseModel):
    cv_id: int = Field(..., ge=1)
    limit: int = Field(default=5, ge=1, le=50)


class GraphCareerPathItem(BaseModel):
    role_name: str
    priority_skills: list[str]
    supporting_jobs: int


class GraphCareerPathResponse(BaseModel):
    cv_id: int
    items: list[GraphCareerPathItem]

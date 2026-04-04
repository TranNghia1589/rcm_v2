from __future__ import annotations

from pydantic import BaseModel, Field


class GuestCVSnapshot(BaseModel):
    target_role: str
    experience_years: str
    skills: list[str] = Field(default_factory=list)
    projects_count: int = 0


class GuestCVScore(BaseModel):
    overall_score: float
    grade: str
    subscores: dict[str, float] = Field(default_factory=dict)


class GuestRecommendationItem(BaseModel):
    rank: int
    job_title: str
    compatibility_percent: float
    job_description: str
    job_url: str


class GuestImproveItem(BaseModel):
    skill: str
    why: str


class GuestAnalyzeResponse(BaseModel):
    intent: str
    snapshot: GuestCVSnapshot
    score: GuestCVScore
    recommendations: list[GuestRecommendationItem] = Field(default_factory=list)
    improve_suggestions: list[GuestImproveItem] = Field(default_factory=list)

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CVUploadResponse(BaseModel):
    user_id: int
    cv_id: int
    gap_report_id: int | None = None
    file_name: str
    target_role: str | None = None
    experience_years: float | None = None
    skills: list[str] = Field(default_factory=list)
    extracted: dict[str, Any] = Field(default_factory=dict)
    gap_report: dict[str, Any] = Field(default_factory=dict)


class CVSummaryItem(BaseModel):
    cv_id: int
    user_id: int
    file_name: str
    target_role: str | None = None
    experience_years: float | None = None
    created_at: str
    updated_at: str


class CVDetailResponse(BaseModel):
    cv_id: int
    user_id: int
    file_name: str
    source_type: str
    target_role: str | None = None
    experience_years: float | None = None
    raw_text: str = ""
    parsed_json: dict[str, Any] = Field(default_factory=dict)
    education_signals: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    latest_gap_report: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class CVListResponse(BaseModel):
    user_id: int
    items: list[CVSummaryItem] = Field(default_factory=list)

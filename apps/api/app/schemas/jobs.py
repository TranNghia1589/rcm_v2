from __future__ import annotations

from pydantic import BaseModel, Field


class JobsListRequest(BaseModel):
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class JobItem(BaseModel):
    job_id: int
    title: str
    company_name: str = ""
    location: str = ""
    job_family: str = ""
    work_mode: str = ""
    is_active: bool = True
    updated_at: str


class JobsListResponse(BaseModel):
    items: list[JobItem]

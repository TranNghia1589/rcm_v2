from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from apps.api.app.schemas.jobs import JobItem, JobsListResponse
from apps.api.app.services.jobs_service import JobsQueryService


router = APIRouter(prefix="/jobs", tags=["jobs"])

BASE_DIR = Path(__file__).resolve().parents[5]
POSTGRES_CFG = BASE_DIR / "configs" / "db" / "postgres.yaml"


@lru_cache(maxsize=1)
def get_jobs_service() -> JobsQueryService:
    return JobsQueryService(postgres_config_path=POSTGRES_CFG)


@router.get("", response_model=JobsListResponse)
def list_jobs(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> JobsListResponse:
    try:
        rows = get_jobs_service().list_jobs(limit=limit, offset=offset)
        return JobsListResponse(items=[JobItem(**x) for x in rows])
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

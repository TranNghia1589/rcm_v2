from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException

from apps.api.app.schemas.cv import CVScoreResponse
from apps.api.app.services.cv_service import CVScoreService


router = APIRouter(prefix="/scoring", tags=["scoring"])

BASE_DIR = Path(__file__).resolve().parents[5]
POSTGRES_CFG = BASE_DIR / "configs" / "db" / "postgres.yaml"


@lru_cache(maxsize=1)
def get_cv_score_service() -> CVScoreService:
    return CVScoreService(postgres_config_path=POSTGRES_CFG)


@router.get("/cv/{cv_id}", response_model=CVScoreResponse)
def get_scoring(cv_id: int) -> CVScoreResponse:
    try:
        out = get_cv_score_service().get_cv_score(cv_id=cv_id)
        if out is None:
            raise HTTPException(status_code=404, detail=f"No scoring result found for cv_id={cv_id}")
        return CVScoreResponse(**out)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

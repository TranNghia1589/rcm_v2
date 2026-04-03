from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException

from apps.api.app.schemas.career import CareerAnalyzeRequest, CareerAnalyzeResponse
from apps.api.app.services.career_analysis_service import CareerAnalysisService


router = APIRouter(prefix="/career", tags=["career"])

BASE_DIR = Path(__file__).resolve().parents[5]
ROLE_PROFILES_PATH = BASE_DIR / "data" / "reference" / "final" / "role_profiles.json"


@lru_cache(maxsize=1)
def get_career_service() -> CareerAnalysisService:
    return CareerAnalysisService(role_profiles_path=ROLE_PROFILES_PATH)


@router.post("/analyze", response_model=CareerAnalyzeResponse)
def analyze_career(payload: CareerAnalyzeRequest) -> CareerAnalyzeResponse:
    try:
        out = get_career_service().analyze(
            intent=payload.intent,
            query=payload.query,
            preferred_role=payload.preferred_role,
            cv_text=payload.cv_text,
            cv_file_path=payload.cv_file_path,
            cv_filename=payload.cv_filename,
            debug=payload.debug,
        )
        return CareerAnalyzeResponse(**out)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

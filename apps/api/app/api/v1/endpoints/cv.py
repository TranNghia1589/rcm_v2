from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from apps.api.app.schemas.cv import CVDetailResponse, CVListResponse, CVUploadResponse
from apps.api.app.services.cv_service import CVService


router = APIRouter(prefix="/cv", tags=["cv"])

BASE_DIR = Path(__file__).resolve().parents[6]
POSTGRES_CFG = BASE_DIR / "configs" / "db" / "postgres.yaml"
ROLE_PROFILES = BASE_DIR / "data" / "role_profiles" / "role_profiles.json"


@lru_cache(maxsize=1)
def get_cv_service() -> CVService:
    return CVService(
        postgres_config_path=POSTGRES_CFG,
        role_profiles_path=ROLE_PROFILES,
    )


@router.post("/upload", response_model=CVUploadResponse)
async def upload_cv(
    file: UploadFile = File(...),
    user_id: int | None = Form(default=None),
    email: str | None = Form(default=None),
    full_name: str | None = Form(default=None),
    phone: str | None = Form(default=None),
    source_type: str = Form(default="upload"),
) -> CVUploadResponse:
    try:
        content = await file.read()
        result = get_cv_service().upload_cv(
            file_name=file.filename or "uploaded_cv.txt",
            content=content,
            user_id=user_id,
            email=email,
            full_name=full_name,
            phone=phone,
            source_type=source_type,
        )
        return CVUploadResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{cv_id}", response_model=CVDetailResponse)
def get_cv(cv_id: int) -> CVDetailResponse:
    try:
        result = get_cv_service().get_cv_detail(cv_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"CV {cv_id} not found")
        return CVDetailResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("", response_model=CVListResponse)
def list_user_cvs(user_id: int = Query(..., ge=1)) -> CVListResponse:
    try:
        result = get_cv_service().list_user_cvs(user_id)
        return CVListResponse(**result)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

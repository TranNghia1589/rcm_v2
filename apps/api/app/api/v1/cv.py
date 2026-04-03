from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, Body, HTTPException, Request

from apps.api.app.schemas.cv import CVScoreResponse, CVUploadResponse
from apps.api.app.services.cv_service import CVScoreService
from src.models.cv.extract_cv_info import extract_cv_info


router = APIRouter(prefix="/cv", tags=["cv"])

BASE_DIR = Path(__file__).resolve().parents[5]
POSTGRES_CFG = BASE_DIR / "config" / "db" / "postgres.yaml"


@lru_cache(maxsize=1)
def get_cv_score_service() -> CVScoreService:
    return CVScoreService(postgres_config_path=POSTGRES_CFG)


@router.get("/score/{cv_id}", response_model=CVScoreResponse)
def get_cv_score(cv_id: int) -> CVScoreResponse:
    try:
        out = get_cv_score_service().get_cv_score(cv_id=cv_id)
        if out is None:
            raise HTTPException(status_code=404, detail=f"No scoring result found for cv_id={cv_id}")
        return CVScoreResponse(**out)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/upload", response_model=CVUploadResponse)
async def upload_cv(
    request: Request,
    file_bytes: bytes = Body(..., description="Raw file bytes"),
    file_name: str = "",
) -> CVUploadResponse:
    if not file_name.strip():
        raise HTTPException(status_code=422, detail="file_name is required (example: my_cv.pdf)")
    if not file_bytes:
        raise HTTPException(status_code=422, detail="file_bytes is empty")

    suffix = Path(file_name or "").suffix.lower()
    allowed_suffixes = {".pdf", ".docx", ".txt"}
    if suffix not in allowed_suffixes:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file format: {suffix or 'unknown'}. Supported: .pdf, .docx, .txt",
        )

    temp_path = ""
    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name

        extracted = extract_cv_info(temp_path)
        return CVUploadResponse(
            file_name=file_name,
            content_type=request.headers.get("content-type", "application/octet-stream"),
            extracted=extracted,
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                pass

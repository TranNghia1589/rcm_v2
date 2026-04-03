from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from apps.api.app.schemas.guest import GuestAnalyzeResponse
from apps.api.app.services.guest_analysis_service import GuestAnalysisService


router = APIRouter(prefix="/guest", tags=["guest"])

BASE_DIR = Path(__file__).resolve().parents[5]
ROLE_PROFILES_PATH = BASE_DIR / "data" / "reference" / "final" / "role_profiles.json"
POSTGRES_CFG = BASE_DIR / "config" / "db" / "postgres.yaml"
RETRIEVAL_CFG = BASE_DIR / "config" / "rag" / "retrieval.yaml"
EMBEDDING_CFG = BASE_DIR / "config" / "model" / "embedding.yaml"
EMBEDDING_SERVICE_CFG = BASE_DIR / "config" / "model" / "embedding_service.yaml"


@lru_cache(maxsize=1)
def get_guest_service() -> GuestAnalysisService:
    return GuestAnalysisService(
        role_profiles_path=ROLE_PROFILES_PATH,
        postgres_config_path=POSTGRES_CFG,
        retrieval_config_path=RETRIEVAL_CFG,
        embedding_config_path=EMBEDDING_CFG,
        embedding_service_config_path=EMBEDDING_SERVICE_CFG,
    )


@router.post("/analyze-cv", response_model=GuestAnalyzeResponse)
async def analyze_cv_guest(
    file: UploadFile = File(...),
    question: str = Form(default="Goi y viec lam phu hop"),
    intent: str = Form(default="auto"),
    top_k: int = Form(default=5),
) -> GuestAnalyzeResponse:
    file_name = (file.filename or "").strip()
    if not file_name:
        raise HTTPException(status_code=422, detail="file name is required")

    suffix = Path(file_name).suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt"}:
        raise HTTPException(status_code=422, detail="Unsupported file format. Use .pdf, .docx, or .txt")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")

    try:
        out = get_guest_service().analyze(
            file_bytes=data,
            file_name=file_name,
            question=question,
            intent=intent,
            top_k=max(1, min(int(top_k), 10)),
        )
        return GuestAnalyzeResponse(**out)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

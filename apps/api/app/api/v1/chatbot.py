from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException

from apps.api.app.schemas.rag.chat import ChatAskRequest, ChatAskResponse
from apps.api.app.services.rag.chat_service import ChatService


router = APIRouter(prefix="/chat", tags=["chatbot"])

BASE_DIR = Path(__file__).resolve().parents[5]
POSTGRES_CFG = BASE_DIR / "configs" / "db" / "postgres.yaml"
RETRIEVAL_CFG = BASE_DIR / "configs" / "rag" / "retrieval.yaml"
PROMPTING_CFG = BASE_DIR / "configs" / "rag" / "prompting.yaml"
EMBEDDING_CFG = BASE_DIR / "configs" / "model" / "embedding.yaml"


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    return ChatService(
        postgres_config_path=POSTGRES_CFG,
        retrieval_config_path=RETRIEVAL_CFG,
        prompting_config_path=PROMPTING_CFG,
        embedding_config_path=EMBEDDING_CFG,
    )


@router.post("/ask", response_model=ChatAskResponse)
def ask_chatbot(payload: ChatAskRequest) -> ChatAskResponse:
    try:
        result = get_chat_service().ask(
            question=payload.question,
            top_k=payload.top_k,
            cv_id=payload.cv_id,
        )
        return ChatAskResponse(**result)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/ask-debug", response_model=ChatAskResponse)
def ask_chatbot_debug(payload: ChatAskRequest) -> ChatAskResponse:
    return ask_chatbot(payload)

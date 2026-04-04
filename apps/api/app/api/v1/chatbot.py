from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from apps.api.app.api.deps import get_current_user
from apps.api.app.schemas.rag.chat import (
    ChatAskRequest,
    ChatAskResponse,
    ChatHistoryLatestResponse,
    ChatHistoryMessage,
    ChatHistorySaveTurnRequest,
    ChatHistorySaveTurnResponse,
)
from apps.api.app.services.rag.chat_service import ChatService
from src.utils.infrastructure.db.postgres_client import PostgresClient, PostgresConfig


router = APIRouter(prefix="/chat", tags=["chatbot"])

BASE_DIR = Path(__file__).resolve().parents[5]
POSTGRES_CFG = BASE_DIR / "config" / "db" / "postgres.yaml"
RETRIEVAL_CFG = BASE_DIR / "config" / "rag" / "retrieval.yaml"
PROMPTING_CFG = BASE_DIR / "config" / "rag" / "prompting.yaml"
EMBEDDING_CFG = BASE_DIR / "config" / "model" / "embedding.yaml"
EMBEDDING_SERVICE_CFG = BASE_DIR / "config" / "model" / "embedding_service.yaml"


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    return ChatService(
        postgres_config_path=POSTGRES_CFG,
        retrieval_config_path=RETRIEVAL_CFG,
        prompting_config_path=PROMPTING_CFG,
        embedding_config_path=EMBEDDING_CFG,
        embedding_service_config_path=EMBEDDING_SERVICE_CFG,
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


@router.post("/history/save-turn", response_model=ChatHistorySaveTurnResponse)
def save_turn(
    payload: ChatHistorySaveTurnRequest,
    current_user: dict = Depends(get_current_user),
) -> ChatHistorySaveTurnResponse:
    user_id = int(current_user["user_id"])
    pg_cfg = PostgresConfig.from_yaml(POSTGRES_CFG)

    with PostgresClient(pg_cfg) as client:
        session_id = (payload.session_id or "").strip()
        if session_id:
            row = client.fetch_one(
                """
                SELECT session_id
                FROM chat_sessions
                WHERE session_id = %s AND user_id = %s
                LIMIT 1
                """,
                (session_id, user_id),
            )
            if not row:
                raise HTTPException(status_code=404, detail="Session not found")
        else:
            title = (payload.title or "Chat CV").strip() or "Chat CV"
            row = client.fetch_one(
                """
                INSERT INTO chat_sessions (user_id, title, model_name, internal_only)
                VALUES (%s, %s, %s, FALSE)
                RETURNING session_id
                """,
                (user_id, title[:120], "guest_analyze"),
            )
            session_id = str(row[0])

        client.execute(
            """
            INSERT INTO chat_messages (session_id, role, content, grounded, metadata)
            VALUES (%s, 'user', %s, TRUE, %s::jsonb)
            """,
            (session_id, payload.user_message, "{}"),
        )
        client.execute(
            """
            INSERT INTO chat_messages (session_id, role, content, grounded, metadata)
            VALUES (%s, 'assistant', %s, TRUE, %s::jsonb)
            """,
            (session_id, payload.assistant_message, "{}"),
        )

    return ChatHistorySaveTurnResponse(session_id=session_id, saved=True)


@router.get("/history/latest", response_model=ChatHistoryLatestResponse)
def get_latest_history(current_user: dict = Depends(get_current_user)) -> ChatHistoryLatestResponse:
    user_id = int(current_user["user_id"])
    pg_cfg = PostgresConfig.from_yaml(POSTGRES_CFG)

    with PostgresClient(pg_cfg) as client:
        row = client.fetch_one(
            """
            SELECT session_id::text, COALESCE(title, '')
            FROM chat_sessions
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_id,),
        )

        if not row:
            return ChatHistoryLatestResponse(session_id=None, title=None, messages=[])

        session_id = str(row[0])
        title = str(row[1] or "")
        rows = client.fetch_all(
            """
            SELECT role, content, created_at
            FROM chat_messages
            WHERE session_id = %s
            ORDER BY created_at ASC, message_id ASC
            LIMIT 200
            """,
            (session_id,),
        )

    messages = [
        ChatHistoryMessage(
            role=str(r[0] or "assistant"),
            content=str(r[1] or ""),
            created_at=str(r[2]),
        )
        for r in rows
    ]
    return ChatHistoryLatestResponse(session_id=session_id, title=title or None, messages=messages)

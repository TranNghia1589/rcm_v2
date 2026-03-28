from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any


class ChatAskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="User question")
    top_k: int = Field(default=5, ge=1, le=20)
    session_id: str | None = None
    user_id: int | None = None
    cv_id: int | None = None
    gap_report_id: int | None = None
    title: str | None = Field(default=None, max_length=255)


class ChatSource(BaseModel):
    chunk_id: int
    document_id: int
    title: str
    distance: float


class ChatAskResponse(BaseModel):
    answer: str
    sources: list[ChatSource]
    retrieval_count: int
    used_fallback: bool = False
    fallback_reason: str = ""
    fallback_stage: str = ""
    session_id: str | None = None
    resolved_cv_id: int | None = None
    resolved_gap_report_id: int | None = None
    history_turns_used: int = 0
    saved_to_history: bool = False
    history_error: str = ""
    debug: dict[str, Any] = Field(default_factory=dict)

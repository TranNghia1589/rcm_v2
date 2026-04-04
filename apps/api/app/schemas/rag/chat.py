from __future__ import annotations

from pydantic import BaseModel, Field


class ChatAskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="User question")
    top_k: int = Field(default=5, ge=1, le=20)
    cv_id: int | None = Field(default=None, ge=1, description="Optional CV id for personalized advice")


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


class ChatHistoryMessage(BaseModel):
    role: str
    content: str
    created_at: str


class ChatHistorySaveTurnRequest(BaseModel):
    session_id: str | None = None
    title: str | None = None
    user_message: str = Field(..., min_length=1)
    assistant_message: str = Field(..., min_length=1)


class ChatHistorySaveTurnResponse(BaseModel):
    session_id: str
    saved: bool = True


class ChatHistoryLatestResponse(BaseModel):
    session_id: str | None = None
    title: str | None = None
    messages: list[ChatHistoryMessage] = Field(default_factory=list)

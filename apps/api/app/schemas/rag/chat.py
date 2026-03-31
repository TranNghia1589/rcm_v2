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

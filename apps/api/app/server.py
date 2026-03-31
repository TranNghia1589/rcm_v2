from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from apps.api.app.api.v1.chatbot import get_chat_service
from apps.api.app.api.v1.recommendations import get_hybrid_service
from apps.api.app.api.v1.router import api_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Warm up heavy runtime objects once so first request is faster.
    if os.getenv("APP_WARMUP_ON_STARTUP", "1").strip() not in {"0", "false", "False"}:
        try:
            _ = get_chat_service()
            _ = get_hybrid_service()
        except Exception:
            # Keep API alive even if warmup components are temporarily unavailable.
            pass
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Job Recommendation + RAG Chatbot API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(api_router, prefix="/api/v1")

    @app.get("/healthz", tags=["system"])
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()

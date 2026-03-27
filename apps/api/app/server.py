from __future__ import annotations

import os

from fastapi import FastAPI

from apps.api.app.api.v1.endpoints.chatbot import get_chat_service
from apps.api.app.api.v1.endpoints.recommend import get_hybrid_service
from apps.api.app.api.v1.router import api_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Job Recommendation + RAG Chatbot API",
        version="0.1.0",
    )
    app.include_router(api_router, prefix="/api/v1")

    @app.on_event("startup")
    def warmup_runtime_services() -> None:
        # Warm up heavy runtime objects once so first request is faster.
        if os.getenv("APP_WARMUP_ON_STARTUP", "1").strip() in {"0", "false", "False"}:
            return
        try:
            _ = get_chat_service()
            _ = get_hybrid_service()
        except Exception:
            # Keep API alive even if warmup components are temporarily unavailable.
            pass

    @app.get("/healthz", tags=["system"])
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()

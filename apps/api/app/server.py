from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


def _parse_cors_origins() -> list[str]:
    raw = os.getenv(
        "APP_CORS_ALLOW_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    origins = [x.strip() for x in raw.split(",") if x.strip()]
    return origins or ["http://localhost:3000", "http://127.0.0.1:3000"]


def create_app() -> FastAPI:
    app = FastAPI(
        title="Job Recommendation + RAG Chatbot API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_parse_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api/v1")

    @app.get("/healthz", tags=["system"])
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()

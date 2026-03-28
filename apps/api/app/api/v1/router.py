from __future__ import annotations

from fastapi import APIRouter

from apps.api.app.api.v1.endpoints.chatbot import router as chatbot_router
from apps.api.app.api.v1.endpoints.cv import router as cv_router
from apps.api.app.api.v1.endpoints.recommend import router as recommend_router

api_router = APIRouter()
api_router.include_router(chatbot_router)
api_router.include_router(cv_router)
api_router.include_router(recommend_router)

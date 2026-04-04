from __future__ import annotations

from fastapi import APIRouter

from apps.api.app.api.v1.auth import router as auth_router
from apps.api.app.api.v1.career import router as career_router
from apps.api.app.api.v1.chatbot import router as chatbot_router
from apps.api.app.api.v1.cv import router as cv_router
from apps.api.app.api.v1.guest import router as guest_router
from apps.api.app.api.v1.jobs import router as jobs_router
from apps.api.app.api.v1.recommendations import router as recommend_router
from apps.api.app.api.v1.scoring import router as scoring_router

api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(career_router)
api_router.include_router(chatbot_router)
api_router.include_router(cv_router)
api_router.include_router(guest_router)
api_router.include_router(scoring_router)
api_router.include_router(jobs_router)
api_router.include_router(recommend_router)

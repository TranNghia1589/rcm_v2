from __future__ import annotations

# Backward-compat wrapper. Canonical implementation lives in apps.api.app.api.v1.cv.
from apps.api.app.api.v1.cv import get_cv_score, get_cv_score_service, router

__all__ = ["router", "get_cv_score_service", "get_cv_score"]

from __future__ import annotations

# Backward-compat wrapper. Canonical implementation lives in apps.api.app.api.v1.jobs.
from apps.api.app.api.v1.jobs import get_jobs_service, list_jobs, router

__all__ = ["router", "get_jobs_service", "list_jobs"]

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.infrastructure.db.postgres_client import PostgresClient, PostgresConfig


@dataclass
class JobsQueryService:
    postgres_config_path: str | Path

    def list_jobs(self, *, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        cfg = PostgresConfig.from_yaml(self.postgres_config_path)
        with PostgresClient(cfg) as client:
            rows = client.fetch_all(
                """
                SELECT
                    job_id,
                    title,
                    COALESCE(company_name, ''),
                    COALESCE(location, ''),
                    COALESCE(job_family, ''),
                    COALESCE(work_mode, ''),
                    COALESCE(is_active, TRUE),
                    updated_at
                FROM jobs
                ORDER BY updated_at DESC, job_id DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset),
            )

        out: list[dict[str, Any]] = []
        for row in rows:
            (
                job_id,
                title,
                company_name,
                location,
                job_family,
                work_mode,
                is_active,
                updated_at,
            ) = row
            out.append(
                {
                    "job_id": int(job_id),
                    "title": str(title),
                    "company_name": str(company_name),
                    "location": str(location),
                    "job_family": str(job_family),
                    "work_mode": str(work_mode),
                    "is_active": bool(is_active),
                    "updated_at": str(updated_at),
                }
            )
        return out

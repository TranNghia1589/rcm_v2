from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.infrastructure.db.postgres_client import PostgresClient, PostgresConfig


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return []
        try:
            parsed = json.loads(txt)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def _to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return {}
        try:
            parsed = json.loads(txt)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


@dataclass
class CVScoreService:
    postgres_config_path: str | Path

    def get_cv_score(self, *, cv_id: int) -> dict[str, Any] | None:
        cfg = PostgresConfig.from_yaml(self.postgres_config_path)
        with PostgresClient(cfg) as client:
            row = client.fetch_one(
                """
                SELECT
                    cv_id, total_score, grade, COALESCE(benchmark_role, ''),
                    skill_score, experience_score, project_score, education_score, completeness_score,
                    strengths, missing_skills, priority_skills, development_plan_30_60_90,
                    subscores_json, metadata, model_version, updated_at
                FROM cv_scoring_results
                WHERE cv_id = %s
                """,
                (cv_id,),
            )

        if not row:
            return None

        (
            cv_id_v,
            total_score,
            grade,
            benchmark_role,
            skill_score,
            experience_score,
            project_score,
            education_score,
            completeness_score,
            strengths,
            missing_skills,
            priority_skills,
            development_plan,
            subscores_json,
            metadata,
            model_version,
            updated_at,
        ) = row

        return {
            "cv_id": int(cv_id_v),
            "total_score": float(total_score),
            "grade": str(grade),
            "benchmark_role": str(benchmark_role),
            "skill_score": float(skill_score),
            "experience_score": float(experience_score),
            "project_score": float(project_score),
            "education_score": float(education_score),
            "completeness_score": float(completeness_score),
            "strengths": _to_list(strengths),
            "missing_skills": _to_list(missing_skills),
            "priority_skills": _to_list(priority_skills),
            "development_plan_30_60_90": _to_dict(development_plan),
            "subscores_json": _to_dict(subscores_json),
            "metadata": _to_dict(metadata),
            "model_version": str(model_version),
            "updated_at": str(updated_at),
        }

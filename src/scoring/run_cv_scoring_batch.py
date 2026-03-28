from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.infrastructure.db.postgres_client import PostgresClient, PostgresConfig
from src.matching.gap_analysis import ROLE_PROFILE_PATH, load_json
from src.scoring.cv_scoring import score_cv_record
from src.scoring.role_benchmark import build_role_benchmarks


def _to_dict_json(value: Any) -> dict[str, Any]:
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


def _to_list_json(value: Any) -> list[Any]:
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


def _fetch_detail_counts(client: PostgresClient, table_name: str) -> dict[int, int]:
    rows = client.fetch_all(f"SELECT cv_id, COUNT(*)::int FROM {table_name} GROUP BY cv_id")
    return {int(cv_id): int(cnt) for cv_id, cnt in rows}


def _upsert_role_benchmark(
    client: PostgresClient,
    *,
    role_name: str,
    role_data: dict[str, Any],
    model_version: str,
) -> None:
    client.execute(
        """
        INSERT INTO role_skill_benchmarks (
            role_name, top_market_skills, top_profile_skills, metadata, model_version, generated_at, updated_at
        )
        VALUES (%s, %s::jsonb, %s::jsonb, %s::jsonb, %s, NOW(), NOW())
        ON CONFLICT (role_name) DO UPDATE
        SET top_market_skills = EXCLUDED.top_market_skills,
            top_profile_skills = EXCLUDED.top_profile_skills,
            metadata = EXCLUDED.metadata,
            model_version = EXCLUDED.model_version,
            updated_at = NOW()
        """,
        (
            role_name,
            json.dumps(role_data.get("top_market_skills", []), ensure_ascii=False),
            json.dumps(role_data.get("top_profile_skills", []), ensure_ascii=False),
            json.dumps({"job_count": role_data.get("job_count", 0)}, ensure_ascii=False),
            model_version,
        ),
    )


def _upsert_cv_score(
    client: PostgresClient,
    *,
    cv_id: int,
    result: dict[str, Any],
    model_version: str,
) -> None:
    client.execute(
        """
        INSERT INTO cv_scoring_results (
            cv_id, total_score, skill_score, experience_score, project_score, education_score, completeness_score,
            grade, benchmark_role, strengths, missing_skills, priority_skills, development_plan_30_60_90,
            subscores_json, metadata, model_version, created_at, updated_at
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb,
            %s::jsonb, %s::jsonb, %s, NOW(), NOW()
        )
        ON CONFLICT (cv_id) DO UPDATE
        SET total_score = EXCLUDED.total_score,
            skill_score = EXCLUDED.skill_score,
            experience_score = EXCLUDED.experience_score,
            project_score = EXCLUDED.project_score,
            education_score = EXCLUDED.education_score,
            completeness_score = EXCLUDED.completeness_score,
            grade = EXCLUDED.grade,
            benchmark_role = EXCLUDED.benchmark_role,
            strengths = EXCLUDED.strengths,
            missing_skills = EXCLUDED.missing_skills,
            priority_skills = EXCLUDED.priority_skills,
            development_plan_30_60_90 = EXCLUDED.development_plan_30_60_90,
            subscores_json = EXCLUDED.subscores_json,
            metadata = EXCLUDED.metadata,
            model_version = EXCLUDED.model_version,
            updated_at = NOW()
        """,
        (
            cv_id,
            result["total_score"],
            result["skill_score"],
            result["experience_score"],
            result["project_score"],
            result["education_score"],
            result["completeness_score"],
            result["grade"],
            result["benchmark_role"],
            json.dumps(result.get("strengths", []), ensure_ascii=False),
            json.dumps(result.get("missing_skills", []), ensure_ascii=False),
            json.dumps(result.get("priority_skills", []), ensure_ascii=False),
            json.dumps(result.get("development_plan_30_60_90", {}), ensure_ascii=False),
            json.dumps(result.get("subscores_json", {}), ensure_ascii=False),
            json.dumps(result.get("metadata", {}), ensure_ascii=False),
            model_version,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch CV scoring with rubric + benchmark roles.")
    parser.add_argument(
        "--postgres_config",
        default=str(BASE_DIR / "configs" / "db" / "postgres.yaml"),
        help="Path to postgres.yaml config.",
    )
    parser.add_argument(
        "--role_profiles",
        default=str(ROLE_PROFILE_PATH),
        help="Path to role_profiles.json.",
    )
    parser.add_argument(
        "--model_version",
        default="cv_scoring_v1",
        help="Scoring model version for traceability.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pg_conf = PostgresConfig.from_yaml(args.postgres_config)
    role_profiles = load_json(Path(args.role_profiles))

    with PostgresClient(pg_conf) as client:
        role_benchmarks = build_role_benchmarks(client, role_profiles=role_profiles)
        for role_name, role_data in role_benchmarks.items():
            _upsert_role_benchmark(
                client,
                role_name=role_name,
                role_data=role_data,
                model_version=args.model_version,
            )

        edu_counts = _fetch_detail_counts(client, "cv_educations")
        exp_counts = _fetch_detail_counts(client, "cv_experiences")
        project_counts = _fetch_detail_counts(client, "cv_projects")
        language_counts = _fetch_detail_counts(client, "cv_languages")
        cert_counts = _fetch_detail_counts(client, "cv_certifications")
        link_counts = _fetch_detail_counts(client, "cv_links")

        gap_rows = client.fetch_all(
            """
            SELECT DISTINCT ON (cv_id)
                   cv_id, domain_fit, target_role_from_cv, best_fit_roles, strengths,
                   missing_skills, top_role_result, role_ranking
            FROM cv_gap_reports
            ORDER BY cv_id, created_at DESC
            """
        )
        gap_map: dict[int, dict[str, Any]] = {}
        for cv_id, domain_fit, target_role_from_cv, best_fit_roles, strengths, missing_skills, top_role_result, role_ranking in gap_rows:
            gap_map[int(cv_id)] = {
                "domain_fit": str(domain_fit or ""),
                "target_role_from_cv": str(target_role_from_cv or ""),
                "best_fit_roles": _to_list_json(best_fit_roles),
                "strengths": _to_list_json(strengths),
                "missing_skills": _to_list_json(missing_skills),
                "top_role_result": _to_dict_json(top_role_result),
                "role_ranking": _to_list_json(role_ranking),
            }

        cv_rows = client.fetch_all(
            """
            SELECT cv_id, parsed_json
            FROM cv_profiles
            ORDER BY cv_id
            """
        )

        processed = 0
        score_sum = 0.0
        for cv_id, parsed_json in cv_rows:
            rec = _to_dict_json(parsed_json)
            if not rec:
                continue
            key = int(cv_id)
            detail_counts = {
                "educations": edu_counts.get(key, 0),
                "experiences": exp_counts.get(key, 0),
                "projects": project_counts.get(key, 0),
                "languages": language_counts.get(key, 0),
                "certifications": cert_counts.get(key, 0),
                "links": link_counts.get(key, 0),
            }
            result = score_cv_record(
                cv_record=rec,
                detail_counts=detail_counts,
                gap_data=gap_map.get(key),
                role_benchmarks=role_benchmarks,
            )
            _upsert_cv_score(client, cv_id=key, result=result, model_version=args.model_version)
            processed += 1
            score_sum += float(result["total_score"])

    avg_score = round(score_sum / processed, 2) if processed else 0.0
    print(
        json.dumps(
            {
                "processed": processed,
                "avg_score": avg_score,
                "model_version": args.model_version,
                "role_benchmarks": len(role_profiles),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

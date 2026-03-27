from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.infrastructure.db.postgres_client import PostgresClient, PostgresConfig


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _to_optional_text(value: Any) -> str | None:
    txt = _to_text(value)
    return txt if txt else None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def _to_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    if hasattr(value, "tolist"):
        vals = value.tolist()
        if isinstance(vals, list):
            return [str(x).strip() for x in vals if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    return [text]


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def ensure_etl_run(client: PostgresClient, source_path: str) -> int:
    row = client.fetch_one(
        """
        INSERT INTO etl_runs (pipeline_name, data_version, source_path, status, metadata)
        VALUES (%s, %s, %s, 'running', %s::jsonb)
        RETURNING etl_run_id
        """,
        (
            "load_core_tables",
            "v1",
            source_path,
            _json_dumps({"note": "Load core tables from artifacts and processed JSON"}),
        ),
    )
    return int(row[0])


def finalize_etl_run(client: PostgresClient, etl_run_id: int, status: str, metadata: dict[str, Any]) -> None:
    client.execute(
        """
        UPDATE etl_runs
        SET status = %s,
            metadata = %s::jsonb,
            finished_at = NOW()
        WHERE etl_run_id = %s
        """,
        (status, _json_dumps(metadata), etl_run_id),
    )


def upsert_user_and_cv(
    client: PostgresClient,
    *,
    resume_json_path: Path,
    etl_run_id: int,
) -> tuple[int, int]:
    data = json.loads(resume_json_path.read_text(encoding="utf-8"))
    email = _to_text(data.get("email")) or "demo_user@local"
    full_name = _to_text(data.get("full_name")) or "Demo User"
    phone = _to_text(data.get("phone"))

    user_row = client.fetch_one(
        """
        INSERT INTO users (email, full_name, phone)
        VALUES (%s, %s, %s)
        ON CONFLICT (email) DO UPDATE
        SET full_name = EXCLUDED.full_name,
            phone = EXCLUDED.phone
        RETURNING user_id
        """,
        (email, full_name, phone),
    )
    user_id = int(user_row[0])

    target_role = _to_text(data.get("target_role"))
    exp_years_raw = _to_text(data.get("experience_years"))
    exp_years = None
    if exp_years_raw and exp_years_raw.lower() != "unknown":
        try:
            exp_years = float(exp_years_raw)
        except Exception:
            exp_years = None

    cv_row = client.fetch_one(
        """
        INSERT INTO cv_profiles (
            user_id, file_name, source_type, raw_text, parsed_json, target_role,
            experience_years, education_signals, etl_run_id
        )
        VALUES (%s, %s, 'upload', %s, %s::jsonb, %s, %s, %s::jsonb, %s)
        RETURNING cv_id
        """,
        (
            user_id,
            _to_text(data.get("file_name")),
            _to_text(data.get("raw_text_preview")),
            _json_dumps(data),
            target_role,
            exp_years,
            _json_dumps(data.get("education_signals", [])),
            etl_run_id,
        ),
    )
    return user_id, int(cv_row[0])


def ensure_skill_id(client: PostgresClient, skill_name: str) -> int:
    skill_name = _to_text(skill_name).lower()
    row = client.fetch_one(
        """
        INSERT INTO skills (canonical_name)
        VALUES (%s)
        ON CONFLICT (canonical_name) DO UPDATE
        SET canonical_name = EXCLUDED.canonical_name
        RETURNING skill_id
        """,
        (skill_name,),
    )
    return int(row[0])


def insert_cv_skills(
    client: PostgresClient,
    *,
    cv_id: int,
    resume_json_path: Path,
) -> int:
    data = json.loads(resume_json_path.read_text(encoding="utf-8"))
    count = 0
    for skill in data.get("skills", []):
        skill_name = _to_text(skill)
        if not skill_name:
            continue
        skill_id = ensure_skill_id(client, skill_name)
        client.execute(
            """
            INSERT INTO cv_skills (cv_id, skill_id, source, confidence)
            VALUES (%s, %s, 'extractor', %s)
            ON CONFLICT (cv_id, skill_id, source) DO NOTHING
            """,
            (cv_id, skill_id, 0.95),
        )
        count += 1
    return count


def upsert_jobs(
    client: PostgresClient,
    *,
    jobs_parquet_path: Path,
    etl_run_id: int,
) -> tuple[int, dict[str, int]]:
    df = pd.read_parquet(jobs_parquet_path)
    inserted = 0
    job_url_to_id: dict[str, int] = {}
    for _, row in df.iterrows():
        job_url = _to_text(row.get("job_url"))
        title = _to_text(row.get("job_title_display"))
        if not job_url or not title:
            continue

        ret = client.fetch_one(
            """
            INSERT INTO jobs (
                external_job_id, job_url, title, title_canonical, company_name, location,
                work_mode, job_family, experience_min_years, experience_max_years,
                employment_type_norm, education_level_norm, job_level_norm,
                description, requirements, metadata, etl_run_id
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
            ON CONFLICT (job_url) DO UPDATE
            SET title = EXCLUDED.title,
                title_canonical = EXCLUDED.title_canonical,
                company_name = EXCLUDED.company_name,
                location = EXCLUDED.location,
                work_mode = EXCLUDED.work_mode,
                job_family = EXCLUDED.job_family,
                experience_min_years = EXCLUDED.experience_min_years,
                experience_max_years = EXCLUDED.experience_max_years,
                employment_type_norm = EXCLUDED.employment_type_norm,
                education_level_norm = EXCLUDED.education_level_norm,
                job_level_norm = EXCLUDED.job_level_norm,
                description = EXCLUDED.description,
                requirements = EXCLUDED.requirements,
                metadata = EXCLUDED.metadata,
                updated_at = NOW(),
                etl_run_id = EXCLUDED.etl_run_id
            RETURNING job_id
            """,
            (
                _to_optional_text(row.get("job_id")),
                job_url,
                title,
                _to_text(row.get("job_title_canonical")),
                _to_text(row.get("company_name_raw")),
                _to_text(row.get("location_norm")),
                _to_text(row.get("work_mode")),
                _to_text(row.get("job_family")),
                _to_float(row.get("experience_min_years")),
                _to_float(row.get("experience_max_years")),
                _to_text(row.get("employment_type_norm")),
                _to_text(row.get("education_level_norm")),
                _to_text(row.get("job_level_norm")),
                _to_text(row.get("job_text_sparse")),
                _to_text(row.get("job_text_phobert_chatbot")),
                _json_dumps(
                    {
                        "salary_min_vnd_month": _to_float(row.get("salary_min_vnd_month")),
                        "salary_max_vnd_month": _to_float(row.get("salary_max_vnd_month")),
                        "salary_is_negotiable": bool(row.get("salary_is_negotiable", False)),
                    }
                ),
                etl_run_id,
            ),
        )
        job_id = int(ret[0])
        job_url_to_id[job_url] = job_id
        inserted += 1
    return inserted, job_url_to_id


def insert_job_skills(
    client: PostgresClient,
    *,
    job_skill_map_path: Path,
    job_url_to_id: dict[str, int],
) -> int:
    df = pd.read_parquet(job_skill_map_path)
    inserted = 0
    for _, row in df.iterrows():
        job_url = _to_text(row.get("job_url"))
        job_id = job_url_to_id.get(job_url)
        if not job_id:
            continue
        skill_name = _to_text(row.get("skill"))
        if not skill_name:
            continue
        skill_id = ensure_skill_id(client, skill_name)
        client.execute(
            """
            INSERT INTO job_skills (job_id, skill_id, source_field, importance, excerpt)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (job_id, skill_id, source_field, importance) DO NOTHING
            """,
            (
                job_id,
                skill_id,
                _to_text(row.get("source_field")),
                _to_text(row.get("importance")) or "mentioned",
                _to_text(row.get("excerpt")),
            ),
        )
        inserted += 1
    return inserted


def insert_gap_report(
    client: PostgresClient,
    *,
    cv_id: int,
    gap_json_path: Path,
) -> int:
    data = json.loads(gap_json_path.read_text(encoding="utf-8"))
    row = client.fetch_one(
        """
        INSERT INTO cv_gap_reports (
            cv_id, domain_fit, target_role_from_cv, best_fit_roles, strengths,
            missing_skills, top_role_result, role_ranking, market_gap_json, model_name
        )
        VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s)
        RETURNING gap_report_id
        """,
        (
            cv_id,
            _to_text(data.get("domain_fit")),
            _to_text(data.get("target_role_from_cv")),
            _json_dumps(data.get("best_fit_roles", [])),
            _json_dumps(data.get("strengths", [])),
            _json_dumps(data.get("missing_skills", [])),
            _json_dumps(data.get("top_role_result", {})),
            _json_dumps(data.get("role_ranking", [])),
            _json_dumps({}),
            "gap_analysis_v1",
        ),
    )
    return int(row[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load core PostgreSQL tables from current artifacts.")
    parser.add_argument(
        "--postgres_config",
        default=str(BASE_DIR / "configs" / "db" / "postgres.yaml"),
    )
    parser.add_argument(
        "--jobs_parquet",
        default=str(BASE_DIR / "artifacts" / "matching" / "jobs_matching_ready_v3.parquet"),
    )
    parser.add_argument(
        "--job_skill_map",
        default=str(BASE_DIR / "artifacts" / "matching" / "job_skill_map_v3.parquet"),
    )
    parser.add_argument(
        "--resume_json",
        default=str(BASE_DIR / "data" / "processed" / "resume_extracted.json"),
    )
    parser.add_argument(
        "--gap_json",
        default=str(BASE_DIR / "data" / "processed" / "gap_analysis_result.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pg_cfg = PostgresConfig.from_yaml(args.postgres_config)

    with PostgresClient(pg_cfg) as client:
        etl_run_id = ensure_etl_run(client, source_path=str(args.jobs_parquet))
        metadata: dict[str, Any] = {}
        try:
            user_id, cv_id = upsert_user_and_cv(
                client,
                resume_json_path=Path(args.resume_json),
                etl_run_id=etl_run_id,
            )
            cv_skill_count = insert_cv_skills(
                client,
                cv_id=cv_id,
                resume_json_path=Path(args.resume_json),
            )
            job_count, url_to_id = upsert_jobs(
                client,
                jobs_parquet_path=Path(args.jobs_parquet),
                etl_run_id=etl_run_id,
            )
            job_skill_count = insert_job_skills(
                client,
                job_skill_map_path=Path(args.job_skill_map),
                job_url_to_id=url_to_id,
            )
            gap_report_id = insert_gap_report(
                client,
                cv_id=cv_id,
                gap_json_path=Path(args.gap_json),
            )
            metadata = {
                "user_id": user_id,
                "cv_id": cv_id,
                "jobs_upserted": job_count,
                "job_skills_inserted": job_skill_count,
                "cv_skills_inserted": cv_skill_count,
                "gap_report_id": gap_report_id,
            }
            finalize_etl_run(client, etl_run_id, "success", metadata)
            print(f"[DONE] {metadata}")
        except Exception as exc:
            client.rollback()
            finalize_etl_run(client, etl_run_id, "failed", {"error": str(exc)})
            client.commit()
            raise


if __name__ == "__main__":
    main()

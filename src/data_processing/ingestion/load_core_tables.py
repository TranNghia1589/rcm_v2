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

from src.utils.infrastructure.db.postgres_client import PostgresClient, PostgresConfig


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


def _safe_cv_key(record: dict[str, Any]) -> str:
    cv_key = _to_text(record.get("cv_key") or record.get("cv_id"))
    if cv_key:
        return cv_key
    fn = _to_text(record.get("file_name"))
    if fn:
        return fn
    src = _to_text(record.get("source_path"))
    if src:
        return src
    return f"cv_{abs(hash(_json_dumps(record))) % 10_000_000}"


def ensure_etl_run(client: PostgresClient, source_path: str) -> int:
    row = client.fetch_one(
        """
        INSERT INTO etl_runs (pipeline_name, data_version, source_path, status, metadata)
        VALUES (%s, %s, %s, 'running', %s::jsonb)
        RETURNING etl_run_id
        """,
        (
            "load_core_tables",
            "v2",
            source_path,
            _json_dumps({"note": "Load jobs + CV batch + gaps to PostgreSQL"}),
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


def ensure_user(client: PostgresClient, *, email: str, full_name: str, phone: str) -> int:
    email_use = email or f"auto_{abs(hash(full_name + phone)) % 10_000_000}@local"
    row = client.fetch_one(
        """
        INSERT INTO users (email, full_name, phone)
        VALUES (%s, %s, %s)
        ON CONFLICT (email) DO UPDATE
        SET full_name = EXCLUDED.full_name,
            phone = EXCLUDED.phone
        RETURNING user_id
        """,
        (email_use, full_name or "Candidate", phone),
    )
    return int(row[0])


def upsert_cv_profile(client: PostgresClient, *, record: dict[str, Any], user_id: int, etl_run_id: int) -> int:
    cv_key = _safe_cv_key(record)
    exp_years_raw = _to_text(record.get("experience_years"))
    exp_years = None
    if exp_years_raw and exp_years_raw.lower() != "unknown":
        exp_years = _to_float(exp_years_raw)

    row = client.fetch_one(
        """
        INSERT INTO cv_profiles (
            cv_key, user_id, file_name, source_path, source_type, schema_version,
            address, career_objective, seniority_level, location_preference, work_mode_preference,
            raw_text, parsed_json, target_role, experience_years, education_signals, etl_run_id
        )
        VALUES (%s, %s, %s, %s, 'upload', %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s::jsonb, %s)
        ON CONFLICT (cv_key) DO UPDATE
        SET user_id = EXCLUDED.user_id,
            file_name = EXCLUDED.file_name,
            source_path = EXCLUDED.source_path,
            schema_version = EXCLUDED.schema_version,
            address = EXCLUDED.address,
            career_objective = EXCLUDED.career_objective,
            seniority_level = EXCLUDED.seniority_level,
            location_preference = EXCLUDED.location_preference,
            work_mode_preference = EXCLUDED.work_mode_preference,
            raw_text = EXCLUDED.raw_text,
            parsed_json = EXCLUDED.parsed_json,
            target_role = EXCLUDED.target_role,
            experience_years = EXCLUDED.experience_years,
            education_signals = EXCLUDED.education_signals,
            updated_at = NOW(),
            etl_run_id = EXCLUDED.etl_run_id
        RETURNING cv_id
        """,
        (
            cv_key,
            user_id,
            _to_text(record.get("file_name")),
            _to_text(record.get("source_path")),
            _to_text(record.get("schema_version")) or "cv_extracted.v1",
            _to_text(record.get("address")),
            _to_text(record.get("career_objective")),
            _to_text(record.get("seniority_level")),
            _to_text(record.get("location_preference")),
            _to_text(record.get("work_mode_preference")),
            _to_text(record.get("raw_text_preview")),
            _json_dumps(record),
            _to_text(record.get("target_role")),
            exp_years,
            _json_dumps(record.get("education_signals", [])),
            etl_run_id,
        ),
    )
    return int(row[0])


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


def replace_cv_detail_tables(client: PostgresClient, *, cv_id: int, record: dict[str, Any]) -> dict[str, int]:
    # reset detail tables for idempotent reload
    for table in [
        "cv_educations",
        "cv_experiences",
        "cv_projects",
        "cv_languages",
        "cv_certifications",
        "cv_links",
    ]:
        client.execute(f"DELETE FROM {table} WHERE cv_id = %s", (cv_id,))

    # CV skills
    client.execute("DELETE FROM cv_skills WHERE cv_id = %s", (cv_id,))
    cv_skill_count = 0
    for skill in _to_list(record.get("skills", [])):
        skill_id = ensure_skill_id(client, skill)
        client.execute(
            """
            INSERT INTO cv_skills (cv_id, skill_id, source, confidence)
            VALUES (%s, %s, 'extractor', %s)
            ON CONFLICT (cv_id, skill_id, source) DO NOTHING
            """,
            (cv_id, skill_id, 0.95),
        )
        cv_skill_count += 1

    # Education
    institutions = _to_list(record.get("educational_institution_name", []))
    degrees = _to_list(record.get("degree_names", []))
    majors = _to_list(record.get("major_field_of_studies", []))
    years = _to_list(record.get("passing_years", []))
    results = _to_list(record.get("educational_results", []))
    result_types = _to_list(record.get("result_types", []))
    n_edu = max(len(institutions), len(degrees), len(majors), len(years), len(results), len(result_types), 0)
    edu_count = 0
    for i in range(n_edu):
        client.execute(
            """
            INSERT INTO cv_educations (cv_id, institution_name, degree_name, major_field, passing_year, educational_result, result_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                cv_id,
                institutions[i] if i < len(institutions) else None,
                degrees[i] if i < len(degrees) else None,
                majors[i] if i < len(majors) else None,
                years[i] if i < len(years) else None,
                results[i] if i < len(results) else None,
                result_types[i] if i < len(result_types) else None,
            ),
        )
        edu_count += 1

    # Experience
    companies = _to_list(record.get("professional_company_names", []))
    positions = _to_list(record.get("positions", []))
    starts = _to_list(record.get("start_dates", []))
    ends = _to_list(record.get("end_dates", []))
    locs = _to_list(record.get("locations", []))
    related = _to_list(record.get("related_skils_in_job", []))
    responsibilities = _to_list(record.get("responsibilities", []))
    n_exp = max(len(companies), len(positions), len(starts), len(ends), len(locs), len(related), 0)
    exp_count = 0
    for i in range(n_exp):
        client.execute(
            """
            INSERT INTO cv_experiences (cv_id, company_name, position_name, start_date, end_date, location, responsibilities, related_skills)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
            """,
            (
                cv_id,
                companies[i] if i < len(companies) else None,
                positions[i] if i < len(positions) else None,
                starts[i] if i < len(starts) else None,
                ends[i] if i < len(ends) else None,
                locs[i] if i < len(locs) else None,
                _json_dumps([responsibilities[i]]) if i < len(responsibilities) else _json_dumps([]),
                _json_dumps([related[i]]) if i < len(related) else _json_dumps([]),
            ),
        )
        exp_count += 1

    # Projects
    projects = _to_list(record.get("projects", []))
    project_count = 0
    for p in projects:
        client.execute(
            """
            INSERT INTO cv_projects (cv_id, project_name, project_description, tech_stack, impact_summary)
            VALUES (%s, %s, %s, %s::jsonb, %s)
            """,
            (cv_id, p[:180], p, _json_dumps([]), ""),
        )
        project_count += 1

    # Certifications
    providers = _to_list(record.get("certification_providers", []))
    cert_skills = _to_list(record.get("certification_skills", []))
    issue_dates = _to_list(record.get("issue_dates", []))
    expiry_dates = _to_list(record.get("expiry_dates", []))
    n_cert = max(len(providers), len(cert_skills), len(issue_dates), len(expiry_dates), 0)
    cert_count = 0
    for i in range(n_cert):
        client.execute(
            """
            INSERT INTO cv_certifications (cv_id, certification_name, provider, issue_date, expiry_date, certification_skills)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb)
            """,
            (
                cv_id,
                (cert_skills[i] if i < len(cert_skills) else f"Certification {i+1}"),
                providers[i] if i < len(providers) else None,
                issue_dates[i] if i < len(issue_dates) else None,
                expiry_dates[i] if i < len(expiry_dates) else None,
                _json_dumps([cert_skills[i]]) if i < len(cert_skills) else _json_dumps([]),
            ),
        )
        cert_count += 1

    # Languages
    langs = _to_list(record.get("languages", []))
    profs = _to_list(record.get("proficiency_levels", []))
    lang_count = 0
    for i, lang in enumerate(langs):
        client.execute(
            """
            INSERT INTO cv_languages (cv_id, language_name, proficiency_level)
            VALUES (%s, %s, %s)
            """,
            (cv_id, lang, profs[i] if i < len(profs) else None),
        )
        lang_count += 1

    # Online links
    links = _to_list(record.get("online_links", []))
    link_count = 0
    for link in links:
        client.execute(
            """
            INSERT INTO cv_links (cv_id, link_url, link_type)
            VALUES (%s, %s, %s)
            """,
            (cv_id, link, "portfolio"),
        )
        link_count += 1

    return {
        "cv_skills": cv_skill_count,
        "educations": edu_count,
        "experiences": exp_count,
        "projects": project_count,
        "certifications": cert_count,
        "languages": lang_count,
        "links": link_count,
    }


def upsert_jobs(client: PostgresClient, *, jobs_parquet_path: Path, etl_run_id: int) -> tuple[int, dict[str, int]]:
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


def insert_job_skills(client: PostgresClient, *, job_skill_map_path: Path, job_url_to_id: dict[str, int]) -> int:
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


def _load_cv_records(cv_dataset_path: Path) -> list[dict[str, Any]]:
    def _decode_json_like(v: Any) -> Any:
        if not isinstance(v, str):
            return v
        text = v.strip()
        if not text:
            return v
        if (text.startswith("[") and text.endswith("]")) or (text.startswith("{") and text.endswith("}")):
            try:
                return json.loads(text)
            except Exception:
                return v
        return v

    if not cv_dataset_path.exists():
        raise FileNotFoundError(f"CV dataset not found: {cv_dataset_path}")

    suffix = cv_dataset_path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(cv_dataset_path)
        rows = [dict(r) for r in df.to_dict(orient="records")]
        return [{k: _decode_json_like(v) for k, v in row.items()} for row in rows]
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with open(cv_dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        data = json.loads(cv_dataset_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [dict(x) for x in data]
        if isinstance(data, dict):
            return [data]
    raise ValueError(f"Unsupported CV dataset format: {cv_dataset_path}")


def _load_gap_mapping(gap_dir: Path) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    if not gap_dir.exists():
        return mapping
    for path in gap_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        key = _to_text(data.get("cv_key"))
        if not key:
            stem = path.stem
            key = stem.replace("_gap", "")
        if key:
            mapping[key] = data
    return mapping


def insert_gap_report(client: PostgresClient, *, cv_id: int, gap_data: dict[str, Any]) -> int:
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
            _to_text(gap_data.get("domain_fit")),
            _to_text(gap_data.get("target_role_from_cv")),
            _json_dumps(gap_data.get("best_fit_roles", [])),
            _json_dumps(gap_data.get("strengths", [])),
            _json_dumps(gap_data.get("missing_skills", [])),
            _json_dumps(gap_data.get("top_role_result", {})),
            _json_dumps(gap_data.get("role_ranking", [])),
            _json_dumps(gap_data.get("market_gap_json", {})),
            _to_text(gap_data.get("model_name")) or "gap_analysis_v1",
        ),
    )
    return int(row[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load core PostgreSQL tables from jobs + CV batch artifacts.")
    parser.add_argument("--postgres_config", default=str(BASE_DIR / "config" / "db" / "postgres.yaml"))
    parser.add_argument("--jobs_parquet", default=str(BASE_DIR / "experiments" / "artifacts" / "matching" / "jobs_matching_ready.parquet"))
    parser.add_argument("--job_skill_map", default=str(BASE_DIR / "experiments" / "artifacts" / "matching" / "job_skill_map.parquet"))
    parser.add_argument(
        "--cv_dataset",
        default=str(BASE_DIR / "artifacts" / "cv" / "cv_extracted_dataset.parquet"),
        help="Parquet/JSONL/JSON dataset for extracted CVs.",
    )
    parser.add_argument(
        "--gap_dir",
        default=str(BASE_DIR / "data" / "processed" / "cv_gap_reports"),
        help="Directory containing per-CV gap JSON files (optional).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pg_cfg = PostgresConfig.from_yaml(args.postgres_config)

    cv_records = _load_cv_records(Path(args.cv_dataset))
    gap_map = _load_gap_mapping(Path(args.gap_dir))

    with PostgresClient(pg_cfg) as client:
        etl_run_id = ensure_etl_run(client, source_path=str(args.jobs_parquet))
        metadata: dict[str, Any] = {}
        try:
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

            cv_count = 0
            cv_skill_count = 0
            edu_count = 0
            exp_count = 0
            project_count = 0
            cert_count = 0
            lang_count = 0
            link_count = 0
            gap_count = 0

            for record in cv_records:
                email = _to_text(record.get("email"))
                full_name = _to_text(record.get("full_name")) or Path(_to_text(record.get("file_name")) or "Candidate").stem
                phone = _to_text(record.get("phone"))
                user_id = ensure_user(client, email=email, full_name=full_name, phone=phone)
                cv_id = upsert_cv_profile(client, record=record, user_id=user_id, etl_run_id=etl_run_id)
                counts = replace_cv_detail_tables(client, cv_id=cv_id, record=record)

                cv_count += 1
                cv_skill_count += counts["cv_skills"]
                edu_count += counts["educations"]
                exp_count += counts["experiences"]
                project_count += counts["projects"]
                cert_count += counts["certifications"]
                lang_count += counts["languages"]
                link_count += counts["links"]

                cv_key = _safe_cv_key(record)
                gap = gap_map.get(cv_key)
                if gap:
                    _ = insert_gap_report(client, cv_id=cv_id, gap_data=gap)
                    gap_count += 1

            metadata = {
                "jobs_upserted": job_count,
                "job_skills_inserted": job_skill_count,
                "cvs_upserted": cv_count,
                "cv_skills_inserted": cv_skill_count,
                "cv_educations_inserted": edu_count,
                "cv_experiences_inserted": exp_count,
                "cv_projects_inserted": project_count,
                "cv_certifications_inserted": cert_count,
                "cv_languages_inserted": lang_count,
                "cv_links_inserted": link_count,
                "cv_gap_reports_inserted": gap_count,
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



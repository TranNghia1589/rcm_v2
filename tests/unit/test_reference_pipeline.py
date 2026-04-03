from __future__ import annotations

import json
from pathlib import Path

from src.data_processing.reference.build_evaluation_cases import build_evaluation_cases
from src.data_processing.reference.build_role_profiles import build_role_profiles
from src.data_processing.reference.build_skill_catalog import build_skill_catalog
from src.data_processing.reference.run_reference_pipeline import run_reference_pipeline
from src.data_processing.reference.validate_and_promote import validate_reference_bundle


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        vals = []
        for h in headers:
            txt = str(row.get(h, ""))
            txt = txt.replace('"', '""')
            vals.append(f'"{txt}"')
        lines.append(",".join(vals))
    path.write_text("\n".join(lines), encoding="utf-8")


def test_build_skill_catalog_from_jobs_csv(tmp_path: Path) -> None:
    jobs_path = tmp_path / "jobs.csv"
    _write_csv(
        jobs_path,
        [
            {
                "source_field_name": "Data Analyst",
                "title": "Data Analyst",
                "detail_title": "Data Analyst",
                "tags": "SQL, Python, Dashboard",
                "desc_mota": "Build dashboards with Power BI and SQL.",
                "desc_yeucau": "Need Python and Statistics.",
                "desc_quyenloi": "",
                "detail_experience": "2 nam",
            },
            {
                "source_field_name": "Data Engineer",
                "title": "Data Engineer",
                "detail_title": "Data Engineer",
                "tags": "Python, ETL, Airflow",
                "desc_mota": "Build ETL with Python and SQL.",
                "desc_yeucau": "Need Airflow and SQL.",
                "desc_quyenloi": "",
                "detail_experience": "3 nam",
            },
        ],
    )
    base_catalog = tmp_path / "base_skill_catalog.json"
    base_catalog.write_text(
        json.dumps(
            {
                "Python": ["python"],
                "SQL": ["sql"],
                "Power BI": ["power bi"],
                "Airflow": ["airflow"],
                "Kubernetes": ["kubernetes"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    out_skill = tmp_path / "staging" / "skill_catalog.json"
    out_report = tmp_path / "staging" / "skill_report.json"
    report = build_skill_catalog(
        jobs_path=jobs_path,
        output_path=out_skill,
        report_path=out_report,
        base_catalog_path=base_catalog,
        seed_catalog_path=None,
        min_mentions=1,
        min_roles=1,
    )

    payload = json.loads(out_skill.read_text(encoding="utf-8"))
    assert "Python" in payload
    assert "SQL" in payload
    assert report["retained_skill_count"] >= 3


def test_build_role_profiles_and_validate(tmp_path: Path) -> None:
    jobs_path = tmp_path / "jobs.csv"
    _write_csv(
        jobs_path,
        [
            {
                "source_field_name": "Data Analyst",
                "title": "Data Analyst",
                "detail_title": "Data Analyst",
                "tags": "SQL, Python, Power BI",
                "desc_mota": "Analyze KPI and build dashboard.",
                "desc_yeucau": "Need SQL and Python",
                "desc_quyenloi": "Bonus",
                "detail_experience": "2 nam",
            },
            {
                "source_field_name": "Data Analyst",
                "title": "BI Analyst",
                "detail_title": "BI Analyst",
                "tags": "Excel, SQL, Power BI",
                "desc_mota": "Report and data analysis.",
                "desc_yeucau": "Power BI",
                "desc_quyenloi": "Hybrid",
                "detail_experience": "1 nam",
            },
        ],
    )
    staging_dir = tmp_path / "data" / "reference" / "staging"
    final_dir = tmp_path / "data" / "reference" / "final"
    staging_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    skill_catalog = staging_dir / "skill_catalog.json"
    skill_catalog.write_text(
        json.dumps(
            {
                "Python": ["python"],
                "SQL": ["sql"],
                "Power BI": ["power bi"],
                "Excel": ["excel"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (final_dir / "skill_catalog.json").write_text("{}", encoding="utf-8")
    (final_dir / "role_profiles.json").write_text("{}", encoding="utf-8")

    role_profiles = staging_dir / "role_profiles.json"
    role_report = staging_dir / "role_report.json"
    build_role_profiles(
        jobs_path=jobs_path,
        skill_catalog_path=skill_catalog,
        output_path=role_profiles,
        report_path=role_report,
        min_jobs_per_role=1,
        top_skills=6,
        top_keywords=6,
    )

    payload = json.loads(role_profiles.read_text(encoding="utf-8"))
    assert "Data Analyst" in payload
    assert "common_skills" in payload["Data Analyst"]

    result = validate_reference_bundle(
        staging_dir=staging_dir,
        final_dir=final_dir,
        drift_limit=1.0,
    )
    assert result.ok is True


def test_build_evaluation_cases(tmp_path: Path) -> None:
    role_profiles = tmp_path / "role_profiles.json"
    role_profiles.write_text(
        json.dumps(
            {
                "Backend Developer": {
                    "role_name": "Backend Developer",
                    "job_count": 10,
                    "common_skills": ["Node.js", "SQL"],
                    "common_keywords": ["api", "backend"],
                    "common_experience_patterns": ["1 years"],
                    "recommended_next_skills": ["Docker", "Kubernetes"],
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    cv_root = tmp_path / "cv_samples"
    synthetic = cv_root / "SYNTHETIC_EVAL"
    synthetic.mkdir(parents=True, exist_ok=True)
    (synthetic / "backend_developer_cv.txt").write_text("Node.js SQL API", encoding="utf-8")

    out_cases = tmp_path / "staging" / "evaluation_cases.json"
    out_report = tmp_path / "staging" / "evaluation_cases_report.json"
    report = build_evaluation_cases(
        role_profiles_path=role_profiles,
        output_path=out_cases,
        report_path=out_report,
        cv_root=cv_root,
        synthetic_folder="SYNTHETIC_EVAL",
        include_manual_cases=False,
    )
    payload = json.loads(out_cases.read_text(encoding="utf-8"))
    assert report["total_cases"] == 1
    assert payload[0]["expected_best_fit_roles_contains"] == ["Backend Developer"]


def test_run_reference_pipeline_end_to_end(tmp_path: Path) -> None:
    jobs_path = tmp_path / "jobs.csv"
    _write_csv(
        jobs_path,
        [
            {
                "source_field_name": "Data Analyst",
                "title": "Data Analyst",
                "detail_title": "Data Analyst",
                "tags": "SQL, Python, Power BI",
                "desc_mota": "Build dashboards and insights.",
                "desc_yeucau": "Need SQL and Python",
                "desc_quyenloi": "",
                "detail_experience": "2 nam",
            }
        ],
    )
    base_catalog = tmp_path / "base_skill_catalog.json"
    base_catalog.write_text(
        json.dumps(
            {
                "Python": ["python"],
                "SQL": ["sql"],
                "Power BI": ["power bi"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    cv_root = tmp_path / "cv_samples"
    syn = cv_root / "SYNTHETIC_EVAL"
    syn.mkdir(parents=True, exist_ok=True)
    (syn / "data_analyst_cv.txt").write_text("SQL Python Power BI", encoding="utf-8")

    staging_dir = tmp_path / "data/reference/staging"
    final_dir = tmp_path / "data/reference/final"
    archive_dir = tmp_path / "data/reference/archive"
    review_report = tmp_path / "data/reference/review/validation_report.json"
    final_dir.mkdir(parents=True, exist_ok=True)
    (final_dir / "skill_catalog.json").write_text("{}", encoding="utf-8")
    (final_dir / "role_profiles.json").write_text("{}", encoding="utf-8")

    result = run_reference_pipeline(
        jobs_path=jobs_path,
        base_catalog_path=base_catalog,
        seed_catalog_path=None,
        staging_dir=staging_dir,
        final_dir=final_dir,
        archive_dir=archive_dir,
        review_report=review_report,
        cv_root=cv_root,
        synthetic_folder="SYNTHETIC_EVAL",
        min_mentions=1,
        min_roles=1,
        min_jobs_per_role=1,
        top_skills=6,
        top_keywords=6,
        drift_limit=1.0,
        promote=True,
        include_eval_cases=True,
        include_manual_cases=False,
    )
    assert result["ok"] is True
    assert (final_dir / "skill_catalog.json").exists()
    assert (final_dir / "role_profiles.json").exists()
    assert (final_dir / "evaluation_cases.json").exists()

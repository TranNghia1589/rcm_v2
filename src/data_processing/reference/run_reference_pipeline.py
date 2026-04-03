from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .build_evaluation_cases import build_evaluation_cases
from .build_role_profiles import build_role_profiles
from .build_skill_catalog import build_skill_catalog
from .common import BASE_DIR, latest_job_raw_path
from .validate_and_promote import promote_bundle, validate_reference_bundle


def run_reference_pipeline(
    *,
    jobs_path: Path,
    base_catalog_path: Path,
    seed_catalog_path: Path | None,
    staging_dir: Path,
    final_dir: Path,
    archive_dir: Path,
    review_report: Path,
    cv_root: Path,
    synthetic_folder: str,
    min_mentions: int,
    min_roles: int,
    min_jobs_per_role: int,
    top_skills: int,
    top_keywords: int,
    drift_limit: float,
    promote: bool,
    include_eval_cases: bool,
    include_manual_cases: bool,
) -> dict[str, Any]:
    staging_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    review_report.parent.mkdir(parents=True, exist_ok=True)

    skill_out = staging_dir / "skill_catalog.json"
    skill_report = staging_dir / "skill_catalog_support_report.json"
    role_out = staging_dir / "role_profiles.json"
    role_report = staging_dir / "role_profiles_support_report.json"
    eval_out = staging_dir / "evaluation_cases.json"
    eval_report = staging_dir / "evaluation_cases_support_report.json"

    step_skill = build_skill_catalog(
        jobs_path=jobs_path,
        output_path=skill_out,
        report_path=skill_report,
        base_catalog_path=base_catalog_path,
        seed_catalog_path=seed_catalog_path,
        min_mentions=min_mentions,
        min_roles=min_roles,
    )
    step_role = build_role_profiles(
        jobs_path=jobs_path,
        skill_catalog_path=skill_out,
        output_path=role_out,
        report_path=role_report,
        min_jobs_per_role=min_jobs_per_role,
        top_skills=top_skills,
        top_keywords=top_keywords,
    )
    step_eval = build_evaluation_cases(
        role_profiles_path=role_out,
        output_path=eval_out,
        report_path=eval_report,
        cv_root=cv_root,
        synthetic_folder=synthetic_folder,
        include_manual_cases=include_manual_cases,
    )

    validation = validate_reference_bundle(
        staging_dir=staging_dir,
        final_dir=final_dir,
        drift_limit=drift_limit,
    )
    payload: dict[str, Any] = {
        "ok": validation.ok,
        "errors": validation.errors,
        "warnings": validation.warnings,
        "steps": {
            "build_skill_catalog": step_skill,
            "build_role_profiles": step_role,
            "build_evaluation_cases": step_eval,
            "validate_bundle": validation.report,
        },
    }
    if validation.ok and promote:
        promoted = promote_bundle(
            staging_dir=staging_dir,
            final_dir=final_dir,
            archive_dir=archive_dir,
            include_eval_cases=include_eval_cases,
        )
        payload["promoted"] = promoted

    review_report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-shot reference pipeline: build -> validate -> promote.")
    parser.add_argument(
        "--jobs_path",
        default="",
        help="Path to jobs raw dataset. Defaults to latest file under data/raw/jobs.",
    )
    parser.add_argument(
        "--base_catalog",
        default="data/reference/final/skill_catalog.json",
        help="Base canonical skill catalog.",
    )
    parser.add_argument(
        "--seed_catalog",
        default="data/reference/seed/skill_catalog_seed.json",
        help="Optional seed catalog path. Use empty string to disable.",
    )
    parser.add_argument("--staging_dir", default="data/reference/staging")
    parser.add_argument("--final_dir", default="data/reference/final")
    parser.add_argument("--archive_dir", default="data/reference/archive")
    parser.add_argument("--review_report", default="data/reference/review/validation_report.json")
    parser.add_argument("--cv_root", default="data/raw/cv_samples")
    parser.add_argument("--synthetic_folder", default="SYNTHETIC_EVAL")
    parser.add_argument("--min_mentions", type=int, default=3)
    parser.add_argument("--min_roles", type=int, default=1)
    parser.add_argument("--min_jobs_per_role", type=int, default=1)
    parser.add_argument("--top_skills", type=int, default=8)
    parser.add_argument("--top_keywords", type=int, default=8)
    parser.add_argument("--drift_limit", type=float, default=0.35)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--include_eval_cases", action="store_true")
    parser.add_argument("--no_manual_cases", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs_path = Path(args.jobs_path) if args.jobs_path else latest_job_raw_path(BASE_DIR)
    seed_path = Path(args.seed_catalog) if args.seed_catalog.strip() else None
    if seed_path is not None:
        seed_path = BASE_DIR / seed_path
        if not seed_path.exists():
            seed_path = None

    payload = run_reference_pipeline(
        jobs_path=jobs_path,
        base_catalog_path=BASE_DIR / args.base_catalog,
        seed_catalog_path=seed_path,
        staging_dir=BASE_DIR / args.staging_dir,
        final_dir=BASE_DIR / args.final_dir,
        archive_dir=BASE_DIR / args.archive_dir,
        review_report=BASE_DIR / args.review_report,
        cv_root=BASE_DIR / args.cv_root,
        synthetic_folder=args.synthetic_folder,
        min_mentions=int(args.min_mentions),
        min_roles=int(args.min_roles),
        min_jobs_per_role=int(args.min_jobs_per_role),
        top_skills=int(args.top_skills),
        top_keywords=int(args.top_keywords),
        drift_limit=float(args.drift_limit),
        promote=bool(args.promote),
        include_eval_cases=bool(args.include_eval_cases),
        include_manual_cases=not bool(args.no_manual_cases),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload.get("ok", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import BASE_DIR


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]
    report: dict[str, Any]


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_skill_catalog(payload: Any) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(payload, dict):
        return ["skill_catalog must be an object"], warnings
    if not payload:
        return ["skill_catalog is empty"], warnings

    for canonical, aliases in payload.items():
        if not str(canonical).strip():
            errors.append("skill_catalog has empty canonical key")
            continue
        if not isinstance(aliases, list) or not aliases:
            errors.append(f"{canonical}: aliases must be a non-empty list")
            continue
        normalized = {str(a).strip().lower() for a in aliases if str(a).strip()}
        if not normalized:
            errors.append(f"{canonical}: aliases list has no valid values")
        if str(canonical).lower() not in normalized:
            warnings.append(f"{canonical}: canonical token not present in aliases")
    return errors, warnings


def _validate_role_profiles(payload: Any) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    required = {
        "role_name",
        "job_count",
        "common_skills",
        "common_keywords",
        "common_experience_patterns",
        "recommended_next_skills",
    }
    if not isinstance(payload, dict):
        return ["role_profiles must be an object"], warnings
    if not payload:
        return ["role_profiles is empty"], warnings

    for role, data in payload.items():
        if not isinstance(data, dict):
            errors.append(f"{role}: role payload must be object")
            continue
        missing = [k for k in required if k not in data]
        if missing:
            errors.append(f"{role}: missing required keys {missing}")
        if int(data.get("job_count", 0)) <= 0:
            warnings.append(f"{role}: job_count <= 0")
        for list_key in [
            "common_skills",
            "common_keywords",
            "common_experience_patterns",
            "recommended_next_skills",
        ]:
            val = data.get(list_key, [])
            if not isinstance(val, list):
                errors.append(f"{role}: {list_key} must be a list")
            elif not val:
                warnings.append(f"{role}: {list_key} is empty")
    return errors, warnings


def _validate_eval_cases(payload: Any) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(payload, list):
        return ["evaluation_cases must be a list"], warnings
    if not payload:
        warnings.append("evaluation_cases is empty")
        return errors, warnings
    for i, case in enumerate(payload, start=1):
        if not isinstance(case, dict):
            errors.append(f"case[{i}] must be object")
            continue
        for key in ["case_id", "cv_file", "expected_domain_fit"]:
            if not str(case.get(key, "")).strip():
                errors.append(f"case[{i}] missing {key}")
    return errors, warnings


def _drift_ratio(new_count: int, old_count: int) -> float:
    if old_count <= 0:
        return 0.0 if new_count <= 0 else 1.0
    return abs(new_count - old_count) / float(old_count)


def validate_reference_bundle(
    *,
    staging_dir: Path,
    final_dir: Path,
    drift_limit: float = 0.35,
) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    staging_skill = staging_dir / "skill_catalog.json"
    staging_role = staging_dir / "role_profiles.json"
    staging_eval = staging_dir / "evaluation_cases.json"
    final_skill = final_dir / "skill_catalog.json"
    final_role = final_dir / "role_profiles.json"

    required_paths = [staging_skill, staging_role]
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        errors.append(f"Missing required staging files: {missing}")
        return ValidationResult(ok=False, errors=errors, warnings=warnings, report={})

    skill_payload = _load_json(staging_skill)
    role_payload = _load_json(staging_role)
    eval_payload = _load_json(staging_eval) if staging_eval.exists() else []

    e, w = _validate_skill_catalog(skill_payload)
    errors.extend(e)
    warnings.extend(w)
    e, w = _validate_role_profiles(role_payload)
    errors.extend(e)
    warnings.extend(w)
    e, w = _validate_eval_cases(eval_payload)
    errors.extend(e)
    warnings.extend(w)

    old_skill_count = len(_load_json(final_skill)) if final_skill.exists() else 0
    old_role_count = len(_load_json(final_role)) if final_role.exists() else 0
    new_skill_count = len(skill_payload)
    new_role_count = len(role_payload)

    skill_drift = _drift_ratio(new_skill_count, old_skill_count)
    role_drift = _drift_ratio(new_role_count, old_role_count)
    if skill_drift > drift_limit:
        warnings.append(
            f"Skill catalog drift is high: {skill_drift:.2%} (old={old_skill_count}, new={new_skill_count})"
        )
    if role_drift > drift_limit:
        warnings.append(
            f"Role profile drift is high: {role_drift:.2%} (old={old_role_count}, new={new_role_count})"
        )

    report = {
        "staging_dir": str(staging_dir),
        "final_dir": str(final_dir),
        "counts": {
            "skill_catalog_new": new_skill_count,
            "skill_catalog_old": old_skill_count,
            "role_profiles_new": new_role_count,
            "role_profiles_old": old_role_count,
            "evaluation_cases_new": len(eval_payload) if isinstance(eval_payload, list) else 0,
        },
        "drift": {
            "skill_catalog_ratio": skill_drift,
            "role_profiles_ratio": role_drift,
            "limit": drift_limit,
        },
    }
    return ValidationResult(ok=len(errors) == 0, errors=errors, warnings=warnings, report=report)


def promote_bundle(
    *,
    staging_dir: Path,
    final_dir: Path,
    archive_dir: Path,
    include_eval_cases: bool = False,
) -> dict[str, str]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot = archive_dir / f"reference_snapshot_{timestamp}"
    snapshot.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    promoted: dict[str, str] = {}
    files = ["skill_catalog.json", "role_profiles.json"]
    if include_eval_cases:
        files.append("evaluation_cases.json")

    for name in files:
        source = staging_dir / name
        if not source.exists():
            continue
        target = final_dir / name
        backup_target = snapshot / name
        if target.exists():
            shutil.copy2(target, backup_target)
        shutil.copy2(source, target)
        promoted[name] = str(target)
    promoted["archive_snapshot"] = str(snapshot)
    return promoted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and promote reference data from staging to final.")
    parser.add_argument("--staging_dir", default="data/reference/staging")
    parser.add_argument("--final_dir", default="data/reference/final")
    parser.add_argument("--archive_dir", default="data/reference/archive")
    parser.add_argument("--review_report", default="data/reference/review/validation_report.json")
    parser.add_argument("--drift_limit", type=float, default=0.35)
    parser.add_argument("--promote", action="store_true", help="Promote staging files to final if validation passes.")
    parser.add_argument(
        "--include_eval_cases",
        action="store_true",
        help="Also promote evaluation_cases.json if present in staging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    staging_dir = BASE_DIR / args.staging_dir
    final_dir = BASE_DIR / args.final_dir
    archive_dir = BASE_DIR / args.archive_dir
    review_report = BASE_DIR / args.review_report

    result = validate_reference_bundle(
        staging_dir=staging_dir,
        final_dir=final_dir,
        drift_limit=float(args.drift_limit),
    )

    payload: dict[str, Any] = {
        "ok": result.ok,
        "errors": result.errors,
        "warnings": result.warnings,
        "report": result.report,
    }
    if result.ok and args.promote:
        promoted = promote_bundle(
            staging_dir=staging_dir,
            final_dir=final_dir,
            archive_dir=archive_dir,
            include_eval_cases=bool(args.include_eval_cases),
        )
        payload["promoted"] = promoted

    review_report.parent.mkdir(parents=True, exist_ok=True)
    with open(review_report, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not result.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


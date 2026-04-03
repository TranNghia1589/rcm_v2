from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .common import BASE_DIR


MANUAL_CASES = [
    {
        "case_id": "cv_data_manual_high_fit",
        "description": "CV data manual in-domain for Data Analyst baseline.",
        "cv_file": "../data/raw/cv_samples/cv_data_manual.txt",
        "expected_target_role": "Data Analyst",
        "expected_domain_fit": "high",
        "expected_best_fit_roles_contains": ["Data Analyst"],
        "expected_missing_skills_contains": ["Power BI"],
    },
    {
        "case_id": "cv_semi_data_manual_medium_fit",
        "description": "CV semi-data profile, expected medium fit for Data Analyst.",
        "cv_file": "../data/raw/cv_samples/cv_semi_data_manual.txt",
        "expected_target_role": "Data Analyst",
        "expected_domain_fit": "medium",
        "expected_best_fit_roles_contains": ["Data Analyst"],
        "expected_missing_skills_contains": ["SQL", "Power BI"],
    },
]


def _slugify(value: str) -> str:
    txt = value.lower().strip()
    txt = re.sub(r"[^a-z0-9]+", "_", txt)
    return txt.strip("_")


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _guess_expected_missing(profile: dict[str, Any]) -> list[str]:
    next_skills = profile.get("recommended_next_skills", [])
    common_skills = profile.get("common_skills", [])
    out = []
    for skill in next_skills:
        if str(skill).strip():
            out.append(str(skill))
        if len(out) >= 2:
            return out
    for skill in common_skills:
        if str(skill).strip():
            out.append(str(skill))
        if len(out) >= 2:
            break
    return out


def _build_synthetic_case(
    *,
    cv_path: Path,
    role_profiles: dict[str, Any],
) -> dict[str, Any] | None:
    stem = cv_path.stem
    role_guess = stem.replace("_cv", "").replace("_", " ").title()

    matched_role = None
    for role_name in role_profiles.keys():
        role_slug = _slugify(role_name)
        stem_slug = _slugify(stem)
        if role_slug in stem_slug or stem_slug in role_slug:
            matched_role = role_name
            break
    if matched_role is None:
        for role_name in role_profiles.keys():
            role_slug = _slugify(role_name)
            stem_slug = _slugify(stem)
            if any(chunk in role_slug for chunk in stem_slug.split("_")):
                matched_role = role_name
                break
    if matched_role is None:
        return None

    profile = role_profiles.get(matched_role, {})
    return {
        "case_id": f"synthetic_{_slugify(matched_role)}",
        "description": f"Synthetic CV case aligned to role {matched_role}.",
        "cv_file": f"../data/raw/cv_samples/SYNTHETIC_EVAL/{cv_path.name}",
        "expected_target_role": "Unknown",
        "expected_domain_fit": "high",
        "expected_best_fit_roles_contains": [matched_role],
        "expected_missing_skills_contains": _guess_expected_missing(profile),
        "_meta": {
            "source": "synthetic_eval_folder",
            "role_guess": role_guess,
        },
    }


def _validate_cases(cases: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    required = {
        "case_id",
        "description",
        "cv_file",
        "expected_target_role",
        "expected_domain_fit",
        "expected_best_fit_roles_contains",
        "expected_missing_skills_contains",
    }
    seen: set[str] = set()
    for i, case in enumerate(cases, start=1):
        missing = [k for k in required if k not in case]
        if missing:
            errors.append(f"case[{i}] missing fields: {missing}")
            continue
        case_id = str(case["case_id"]).strip()
        if not case_id:
            errors.append(f"case[{i}] empty case_id")
            continue
        if case_id in seen:
            errors.append(f"duplicate case_id: {case_id}")
        seen.add(case_id)
    return errors


def _dedupe_case_ids(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, int] = {}
    out: list[dict[str, Any]] = []
    for case in cases:
        base_id = str(case.get("case_id", "")).strip()
        if not base_id:
            out.append(case)
            continue
        count = seen.get(base_id, 0)
        if count == 0:
            seen[base_id] = 1
            out.append(case)
            continue
        seen[base_id] = count + 1
        updated = dict(case)
        updated["case_id"] = f"{base_id}_{count+1}"
        out.append(updated)
    return out


def build_evaluation_cases(
    *,
    role_profiles_path: Path,
    output_path: Path,
    report_path: Path,
    cv_root: Path,
    synthetic_folder: str = "SYNTHETIC_EVAL",
    include_manual_cases: bool = True,
) -> dict[str, Any]:
    role_profiles = _load_json(role_profiles_path)
    if not isinstance(role_profiles, dict):
        raise ValueError(f"Invalid role profiles at {role_profiles_path}")

    cases: list[dict[str, Any]] = []
    if include_manual_cases:
        for case in MANUAL_CASES:
            cv_path = BASE_DIR / case["cv_file"]
            if cv_path.exists():
                cases.append(dict(case))

    synthetic_dir = cv_root / synthetic_folder
    matched = 0
    skipped = 0
    if synthetic_dir.exists():
        for cv_path in sorted(synthetic_dir.glob("*.txt")):
            built = _build_synthetic_case(cv_path=cv_path, role_profiles=role_profiles)
            if built is None:
                skipped += 1
                continue
            cases.append(built)
            matched += 1

    cases = _dedupe_case_ids(cases)
    errors = _validate_cases(cases)
    if errors:
        raise ValueError("Invalid evaluation cases: " + "; ".join(errors))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    report = {
        "role_profiles_path": str(role_profiles_path),
        "cv_root": str(cv_root),
        "synthetic_folder": str(synthetic_dir),
        "total_cases": len(cases),
        "manual_cases_included": include_manual_cases,
        "synthetic_matched": matched,
        "synthetic_skipped": skipped,
        "case_ids": [c["case_id"] for c in cases],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build staging evaluation_cases from CV sample files.")
    parser.add_argument(
        "--role_profiles_path",
        default="data/reference/staging/role_profiles.json",
        help="Path to role profiles JSON used to infer expected roles/skills.",
    )
    parser.add_argument(
        "--cv_root",
        default="data/raw/cv_samples",
        help="Root directory containing manual and synthetic CV text files.",
    )
    parser.add_argument(
        "--synthetic_folder",
        default="SYNTHETIC_EVAL",
        help="Folder under cv_root containing synthetic eval text files.",
    )
    parser.add_argument(
        "--output_path",
        default="data/reference/staging/evaluation_cases.json",
        help="Staging output for generated evaluation cases.",
    )
    parser.add_argument(
        "--report_path",
        default="data/reference/staging/evaluation_cases_support_report.json",
        help="Output support report path.",
    )
    parser.add_argument(
        "--no_manual_cases",
        action="store_true",
        help="Disable inclusion of manual baseline cases.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_evaluation_cases(
        role_profiles_path=BASE_DIR / args.role_profiles_path,
        output_path=BASE_DIR / args.output_path,
        report_path=BASE_DIR / args.report_path,
        cv_root=BASE_DIR / args.cv_root,
        synthetic_folder=args.synthetic_folder,
        include_manual_cases=not bool(args.no_manual_cases),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import (
    BASE_DIR,
    EXP_COLUMN_CANDIDATES,
    ROLE_COLUMN_CANDIDATES,
    build_skill_text,
    extract_keywords,
    latest_job_raw_path,
    load_records,
    normalize_alias,
    normalize_role_name,
    pick_first_key,
)


def _load_skill_catalog(path: Path) -> dict[str, list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: dict[str, list[str]] = {}
    for canonical, aliases in data.items():
        if isinstance(aliases, list) and aliases:
            out[str(canonical)] = [normalize_alias(a) for a in aliases if str(a).strip()]
    return out


def _extract_present_skills(text: str, catalog: dict[str, list[str]]) -> set[str]:
    lowered = text.lower()
    found: set[str] = set()
    for canonical, aliases in catalog.items():
        for alias in aliases:
            pattern = r"\b" + re.escape(alias) + r"\b"
            if re.search(pattern, lowered):
                found.add(canonical)
                break
    return found


def _normalize_experience(raw: str) -> str:
    txt = str(raw).strip()
    if not txt:
        return "Unknown"
    lowered = txt.lower()
    if "khong" in lowered and "yeu cau" in lowered:
        return "No requirement"
    match = re.search(r"(\d+)\s*nam", lowered)
    if match:
        return f"{match.group(1)} years"
    if "duoi" in lowered and "1" in lowered:
        return "<1 year"
    return txt[:32]


def build_role_profiles(
    *,
    jobs_path: Path,
    skill_catalog_path: Path,
    output_path: Path,
    report_path: Path,
    min_jobs_per_role: int = 1,
    top_skills: int = 8,
    top_keywords: int = 8,
) -> dict[str, Any]:
    rows = load_records(jobs_path)
    if not rows:
        raise ValueError(f"No records found in {jobs_path}")
    catalog = _load_skill_catalog(skill_catalog_path)

    role_job_count: dict[str, int] = defaultdict(int)
    role_skill_count: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    role_text_chunks: dict[str, list[str]] = defaultdict(list)
    role_exp_count: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in rows:
        role_raw = pick_first_key(row, ROLE_COLUMN_CANDIDATES)
        role = normalize_role_name(role_raw)
        role_job_count[role] += 1
        text = build_skill_text(row)
        role_text_chunks[role].append(text)

        for skill in _extract_present_skills(text, catalog):
            role_skill_count[role][skill] += 1

        exp_text = pick_first_key(row, EXP_COLUMN_CANDIDATES)
        role_exp_count[role][_normalize_experience(exp_text)] += 1

    global_skill_count: dict[str, int] = defaultdict(int)
    for role in role_skill_count:
        for skill, count in role_skill_count[role].items():
            global_skill_count[skill] += count

    profiles: dict[str, Any] = {}
    for role, jobs in sorted(role_job_count.items(), key=lambda kv: (-kv[1], kv[0])):
        if jobs < min_jobs_per_role:
            continue
        sorted_skills = sorted(role_skill_count[role].items(), key=lambda kv: (-kv[1], kv[0]))
        common_skills = [name for name, _ in sorted_skills[:top_skills]]
        next_skills = [name for name, _ in sorted_skills[top_skills : top_skills + 4]]

        if len(next_skills) < 4:
            fallback = sorted(global_skill_count.items(), key=lambda kv: (-kv[1], kv[0]))
            for skill, _ in fallback:
                if skill in common_skills or skill in next_skills:
                    continue
                next_skills.append(skill)
                if len(next_skills) >= 4:
                    break

        exp_patterns = sorted(
            role_exp_count[role].items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
        common_exp = [name for name, _ in exp_patterns[:5]]

        keywords = extract_keywords(role_text_chunks[role], top_k=top_keywords)

        profiles[role] = {
            "role_name": role,
            "job_count": jobs,
            "common_skills": common_skills,
            "common_keywords": keywords,
            "common_experience_patterns": common_exp,
            "recommended_next_skills": next_skills[:4],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

    report = {
        "jobs_path": str(jobs_path),
        "skill_catalog_path": str(skill_catalog_path),
        "total_jobs": len(rows),
        "role_count": len(profiles),
        "roles": sorted(profiles.keys()),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build staging role_profiles from jobs raw data.")
    parser.add_argument(
        "--jobs_path",
        default="",
        help="Path to jobs raw dataset (csv/parquet/jsonl/xlsx). Defaults to latest file under data/raw/jobs.",
    )
    parser.add_argument(
        "--skill_catalog_path",
        default="data/reference/staging/skill_catalog.json",
        help="Path to input skill catalog.",
    )
    parser.add_argument(
        "--output_path",
        default="data/reference/staging/role_profiles.json",
        help="Staging output for generated role profiles.",
    )
    parser.add_argument(
        "--report_path",
        default="data/reference/staging/role_profiles_support_report.json",
        help="Output support report path.",
    )
    parser.add_argument("--min_jobs_per_role", type=int, default=1)
    parser.add_argument("--top_skills", type=int, default=8)
    parser.add_argument("--top_keywords", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs_path = Path(args.jobs_path) if args.jobs_path else latest_job_raw_path(BASE_DIR)
    report = build_role_profiles(
        jobs_path=jobs_path,
        skill_catalog_path=BASE_DIR / args.skill_catalog_path,
        output_path=BASE_DIR / args.output_path,
        report_path=BASE_DIR / args.report_path,
        min_jobs_per_role=int(args.min_jobs_per_role),
        top_skills=int(args.top_skills),
        top_keywords=int(args.top_keywords),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


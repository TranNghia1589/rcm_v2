from __future__ import annotations

from typing import Any


WEIGHTS = {
    "skills": 35.0,
    "experience": 25.0,
    "projects": 20.0,
    "education": 10.0,
    "completeness": 10.0,
}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        txt = str(value).strip().lower()
        if not txt or txt == "unknown":
            return default
        return float(txt)
    except Exception:
        return default


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, tuple):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def _grade(total_score: float) -> str:
    if total_score >= 85:
        return "A"
    if total_score >= 70:
        return "B"
    if total_score >= 55:
        return "C"
    if total_score >= 40:
        return "D"
    return "E"


def _build_plan(priority_skills: list[str], has_projects: bool) -> dict[str, list[str]]:
    day30 = []
    day60 = []
    day90 = []

    for s in priority_skills[:2]:
        day30.append(f"Onboard and practice core {s} fundamentals.")
    for s in priority_skills[2:5]:
        day60.append(f"Build one mini project applying {s}.")
    for s in priority_skills[5:8]:
        day90.append(f"Add production-level proof using {s} in portfolio.")

    if not has_projects:
        day30.append("Draft one project scope aligned with target role.")
        day60.append("Complete and publish the first portfolio project.")
        day90.append("Refine project impact metrics and CV bullet points.")

    return {"day_30": day30[:4], "day_60": day60[:4], "day_90": day90[:4]}


def score_cv_record(
    *,
    cv_record: dict[str, Any],
    detail_counts: dict[str, int],
    gap_data: dict[str, Any] | None,
    role_benchmarks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    skills = {s.lower() for s in _as_list(cv_record.get("skills"))}
    projects = _as_list(cv_record.get("projects"))
    target_role = str(cv_record.get("target_role", "")).strip() or "Unknown"

    gap = gap_data or {}
    top_role_result = gap.get("top_role_result", {}) if isinstance(gap, dict) else {}
    benchmark_role = str(top_role_result.get("role") or target_role or "Unknown").strip()
    benchmark = role_benchmarks.get(benchmark_role, {})

    matched_skills = _as_list(top_role_result.get("matched_skills"))
    missing_skills = _as_list(top_role_result.get("missing_skills"))
    if not missing_skills:
        bm = [s.lower() for s in _as_list(benchmark.get("top_market_skills"))]
        missing_skills = [s for s in bm if s not in skills][:10]
    if not matched_skills:
        bm = [s.lower() for s in _as_list(benchmark.get("top_market_skills"))]
        matched_skills = [s for s in bm if s in skills]

    denom = max(len(matched_skills) + len(missing_skills), 1)
    skill_ratio = min(len(matched_skills) / denom, 1.0)
    skill_score = round(skill_ratio * WEIGHTS["skills"], 2)

    years = _to_float(cv_record.get("experience_years"))
    exp_rows = int(detail_counts.get("experiences", 0))
    exp_ratio = min(0.7 * min(years / 4.0, 1.0) + 0.3 * min(exp_rows / 3.0, 1.0), 1.0)
    experience_score = round(exp_ratio * WEIGHTS["experience"], 2)

    proj_rows = int(detail_counts.get("projects", 0))
    project_ratio = min(0.7 * min(max(len(projects), proj_rows) / 3.0, 1.0) + 0.3 * min(len(skills) / 8.0, 1.0), 1.0)
    project_score = round(project_ratio * WEIGHTS["projects"], 2)

    edu_rows = int(detail_counts.get("educations", 0))
    degree_names = " ".join(_as_list(cv_record.get("degree_names"))).lower()
    degree_bonus = 0.3 if any(x in degree_names for x in ["bachelor", "master", "phd", "engineer"]) else 0.0
    education_ratio = min(0.7 * min(edu_rows / 1.0, 1.0) + degree_bonus, 1.0)
    education_score = round(education_ratio * WEIGHTS["education"], 2)

    required_fields = [
        _as_list(cv_record.get("skills")),
        _as_list(cv_record.get("projects")),
        _as_list(cv_record.get("professional_company_names")),
        _as_list(cv_record.get("positions")),
        _as_list(cv_record.get("educational_institution_name")),
        _as_list(cv_record.get("degree_names")),
        _as_list(cv_record.get("start_dates")),
        _as_list(cv_record.get("end_dates")),
        str(cv_record.get("target_role", "")).strip(),
        str(cv_record.get("experience_years", "")).strip(),
    ]
    filled = 0
    for item in required_fields:
        if isinstance(item, list):
            filled += 1 if item else 0
        else:
            filled += 1 if item and item.lower() != "unknown" else 0
    completeness_ratio = filled / len(required_fields)
    completeness_score = round(completeness_ratio * WEIGHTS["completeness"], 2)

    total_score = round(
        skill_score + experience_score + project_score + education_score + completeness_score,
        2,
    )
    priority_skills = missing_skills[:8]
    plan = _build_plan(priority_skills, has_projects=bool(projects))

    return {
        "total_score": total_score,
        "skill_score": skill_score,
        "experience_score": experience_score,
        "project_score": project_score,
        "education_score": education_score,
        "completeness_score": completeness_score,
        "grade": _grade(total_score),
        "benchmark_role": benchmark_role,
        "strengths": matched_skills[:10],
        "missing_skills": missing_skills[:10],
        "priority_skills": priority_skills,
        "development_plan_30_60_90": plan,
        "subscores_json": {
            "weights": WEIGHTS,
            "ratios": {
                "skills": round(skill_ratio, 4),
                "experience": round(exp_ratio, 4),
                "projects": round(project_ratio, 4),
                "education": round(education_ratio, 4),
                "completeness": round(completeness_ratio, 4),
            },
        },
        "metadata": {
            "target_role": target_role,
            "experience_years": years,
            "detail_counts": detail_counts,
        },
    }

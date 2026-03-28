from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from src.infrastructure.db.postgres_client import PostgresClient


def _norm_skill(skill: str) -> str:
    return str(skill).strip().lower()


def _role_match(text: str, role_name: str) -> bool:
    t = text.lower()
    r = role_name.lower()
    if r in t:
        return True
    aliases = {
        "Data Analyst": ["bi analyst", "business intelligence analyst"],
        "Data Engineer": ["etl engineer", "pipeline engineer"],
        "Data Scientist": ["ml scientist"],
        "AI Engineer": ["machine learning engineer", "ml engineer"],
        "AI Researcher": ["research scientist", "ai scientist"],
    }
    return any(alias in t for alias in aliases.get(role_name, []))


def build_role_benchmarks(
    client: PostgresClient,
    *,
    role_profiles: dict[str, dict[str, Any]],
    top_k: int = 15,
) -> dict[str, dict[str, Any]]:
    rows = client.fetch_all(
        """
        SELECT j.title, COALESCE(j.title_canonical, ''), COALESCE(j.job_family, ''), s.canonical_name
        FROM jobs j
        JOIN job_skills js ON js.job_id = j.job_id
        JOIN skills s ON s.skill_id = js.skill_id
        """
    )

    role_skill_counter: dict[str, Counter[str]] = defaultdict(Counter)
    role_job_ids: dict[str, set[str]] = defaultdict(set)

    for title, title_canonical, job_family, skill in rows:
        title_txt = f"{title} {title_canonical} {job_family}".strip()
        skill_norm = _norm_skill(skill)
        if not skill_norm:
            continue
        for role_name in role_profiles.keys():
            if _role_match(title_txt, role_name):
                role_skill_counter[role_name][skill_norm] += 1
                role_job_ids[role_name].add(title_txt.lower())

    out: dict[str, dict[str, Any]] = {}
    for role_name, role_data in role_profiles.items():
        market_skills = [s for s, _ in role_skill_counter[role_name].most_common(top_k)]
        profile_skills = [_norm_skill(s) for s in role_data.get("common_skills", []) if _norm_skill(s)]

        # Fallback if job-title matching for a role is weak.
        if not market_skills:
            market_skills = profile_skills[:top_k]

        out[role_name] = {
            "role_name": role_name,
            "job_count": len(role_job_ids[role_name]),
            "top_market_skills": market_skills,
            "top_profile_skills": profile_skills[:top_k],
        }
    return out

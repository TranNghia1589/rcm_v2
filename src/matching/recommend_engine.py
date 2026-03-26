from __future__ import annotations

import ast
import glob
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_skill(skill: str) -> str:
    s = _normalize_text(skill).lower()
    alias_map = {
        "powerbi": "power bi",
        "scikit-learn": "sklearn",
        "postgresql": "postgres",
    }
    return alias_map.get(s, s)


def _parse_skills_from_row(row: pd.Series) -> List[str]:
    for key in ("skills_extracted", "skills_required", "skills_preferred"):
        raw = row.get(key, None)
        if isinstance(raw, list):
            vals = sorted({_normalize_skill(str(x)) for x in raw if str(x).strip()})
            if vals:
                return vals
        raw_s = _normalize_text(raw)
        if raw_s.startswith("[") and raw_s.endswith("]"):
            try:
                parsed = ast.literal_eval(raw_s)
                if isinstance(parsed, list):
                    vals = sorted({_normalize_skill(str(x)) for x in parsed if str(x).strip()})
                    if vals:
                        return vals
            except Exception:
                pass

    raw_pipe = _normalize_text(row.get("skills_normalized_str", ""))
    if raw_pipe:
        return sorted(
            {
                _normalize_skill(x)
                for x in [p.strip() for p in raw_pipe.split("|")]
                if x.strip()
            }
        )

    raw_list = row.get("skills_normalized", None)
    if isinstance(raw_list, list):
        return sorted({_normalize_skill(str(x)) for x in raw_list if str(x).strip()})

    raw_list_str = _normalize_text(raw_list)
    if raw_list_str.startswith("[") and raw_list_str.endswith("]"):
        try:
            parsed = ast.literal_eval(raw_list_str)
            if isinstance(parsed, list):
                return sorted({_normalize_skill(str(x)) for x in parsed if str(x).strip()})
        except Exception:
            pass

    return []


def _extract_cv_skills(cv_info: Dict[str, Any] | None, gap_result: Dict[str, Any] | None) -> List[str]:
    cv_skills: List[str] = []

    if cv_info:
        cv_skills.extend(cv_info.get("skills", []))
    if gap_result:
        cv_skills.extend(gap_result.get("strengths", []))

    seen = set()
    result = []
    for s in cv_skills:
        norm = _normalize_skill(str(s))
        if norm and norm not in seen:
            seen.add(norm)
            result.append(norm)
    return result


def find_latest_jobs_file(base_dir: Path) -> Path:
    parquet_default = base_dir / "artifacts" / "matching" / "jobs_matching_ready_v3.parquet"
    if parquet_default.exists():
        return parquet_default

    pattern = str(base_dir / "data" / "processed" / "jobs_nlp_ready_*.csv")
    candidates = glob.glob(pattern)
    if candidates:
        candidates.sort(key=os.path.getmtime, reverse=True)
        return Path(candidates[0])

    raise FileNotFoundError(
        "Khong tim thay jobs source. Da tim tai "
        f"{parquet_default} va {base_dir / 'data' / 'processed'}"
    )


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    cleaned = re.sub(r"[^a-zA-Z0-9\s]+", " ", text.lower())
    return [x for x in cleaned.split() if x]


def _score_role_alignment(title: str, best_fit_roles: List[str]) -> float:
    title_low = title.lower()
    role_keywords = {
        "Data Analyst": ["data analyst", "bi analyst", "business intelligence", "analytics"],
        "Data Engineer": ["data engineer", "etl", "pipeline", "warehouse"],
        "Data Scientist": ["data scientist", "machine learning", "ml"],
        "AI Engineer": ["ai engineer", "ml engineer", "llm", "deep learning"],
        "AI Researcher": ["ai researcher", "research"],
    }

    bonus = 0.0
    for idx, role in enumerate(best_fit_roles[:3]):
        kws = role_keywords.get(role, [])
        if any(kw in title_low for kw in kws):
            bonus = max(bonus, 0.18 - idx * 0.04)
    return max(0.0, bonus)


def get_top_job_recommendations(
    *,
    cv_info: Dict[str, Any] | None,
    gap_result: Dict[str, Any] | None,
    jobs_path: str | Path,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    jobs_path = Path(jobs_path)
    if jobs_path.suffix.lower() == ".parquet":
        jobs_df = pd.read_parquet(jobs_path)
    else:
        jobs_df = pd.read_csv(jobs_path)
    cv_skill_set = set(_extract_cv_skills(cv_info, gap_result))
    best_fit_roles = gap_result.get("best_fit_roles", []) if gap_result else []

    ranked: List[Dict[str, Any]] = []

    for _, row in jobs_df.iterrows():
        job_skills = _parse_skills_from_row(row)
        job_skill_set = set(job_skills)
        matched = sorted(cv_skill_set & job_skill_set)
        missing = sorted(job_skill_set - cv_skill_set)

        if job_skill_set:
            skill_score = len(matched) / len(job_skill_set)
        else:
            # fallback using text token overlap when no extracted skill
            text_fields = " ".join(
                [
                    _normalize_text(row.get("job_title_clean", "")),
                    _normalize_text(row.get("requirements_clean", "")),
                    _normalize_text(row.get("description_clean", "")),
                ]
            )
            tokens = set(_tokenize(text_fields))
            skill_score = len(tokens & cv_skill_set) / max(len(cv_skill_set), 1)

        role_bonus = _score_role_alignment(_normalize_text(row.get("job_title_clean", "")), best_fit_roles)
        final_score = min(1.0, 0.82 * skill_score + role_bonus)

        job_title = (
            _normalize_text(row.get("job_title_display", ""))
            or _normalize_text(row.get("job_title_raw", ""))
            or _normalize_text(row.get("job_title_clean", ""))
            or "Unknown"
        )
        company_name = _normalize_text(row.get("company_name_raw", ""))
        location = (
            _normalize_text(row.get("location_norm", ""))
            or _normalize_text(row.get("location_normalized", ""))
            or _normalize_text(row.get("location_raw", ""))
        )
        requirements_clean = (
            _normalize_text(row.get("job_text_phobert_chatbot", ""))
            or _normalize_text(row.get("requirements_clean", ""))
        )
        description_clean = (
            _normalize_text(row.get("job_text_sparse", ""))
            or _normalize_text(row.get("description_clean", ""))
        )

        ranked.append(
            {
                "job_title": job_title,
                "company_name": company_name,
                "location": location,
                "salary": _normalize_text(row.get("salary_raw", "")),
                "job_url": _normalize_text(row.get("job_url", "")),
                "score": round(final_score, 4),
                "matched_skills": matched[:8],
                "missing_skills": missing[:8],
                "job_skills": sorted(job_skill_set),
                "requirements_clean": requirements_clean,
                "description_clean": description_clean,
                "experience_min_years": row.get("experience_min_years", None),
                "experience_max_years": row.get("experience_max_years", None),
                "experience_raw": _normalize_text(row.get("experience_raw", "")),
            }
        )

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]

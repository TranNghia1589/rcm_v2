from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


BASE_DIR = Path(__file__).resolve().parents[2]
ROLE_PROFILE_PATH = BASE_DIR / "data" / "role_profiles" / "role_profiles.json"
DEFAULT_OUTPUT_PATH = BASE_DIR / "data" / "processed" / "gap_analysis_result.json"
DEFAULT_CV_JSON = BASE_DIR / "data" / "processed" / "resume_extracted.json"


ROLE_ALIASES = {
    "sql": "SQL",
    "excel": "Excel",
    "power bi": "Power BI",
    "powerbi": "Power BI",
    "tableau": "Tableau",
    "python": "Python",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "statistics": "Statistics",
    "machine learning": "Machine Learning",
    "deep learning": "Deep Learning",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "scikit-learn": "Scikit-learn",
    "sklearn": "Scikit-learn",
    "spark": "Spark",
    "airflow": "Airflow",
    "etl": "ETL",
    "data warehouse": "Data Warehouse",
    "nlp": "NLP",
    "computer vision": "Computer Vision",
    "dashboard": "Dashboarding",
    "data visualization": "Data Visualization",
    "git": "Git",
    "docker": "Docker",
    "linux": "Linux",
    "mysql": "MySQL",
    "postgresql": "PostgreSQL",
    "postgres": "PostgreSQL",
    "mongodb": "MongoDB",
    "aws": "AWS",
    "azure": "Azure",
    "gcp": "GCP",
    "llm": "LLM",
    "rag": "RAG",
    "langchain": "LangChain",
    "streamlit": "Streamlit",
    "flask": "Flask",
    "fastapi": "FastAPI",
}


ROLE_KEYWORD_HINTS = {
    "Data Analyst": ["sql", "excel", "power bi", "tableau", "statistics", "dashboard"],
    "Data Engineer": ["python", "sql", "etl", "airflow", "spark", "data warehouse"],
    "AI Engineer": ["python", "machine learning", "deep learning", "pytorch", "tensorflow", "llm"],
    "AI Researcher": ["machine learning", "deep learning", "research", "experiment", "paper", "statistics"],
    "Data Scientist": ["python", "machine learning", "statistics", "pandas", "numpy", "modeling"],
    "Data Labeling": ["annotation", "data labeling", "quality control", "review"],
}


def resolve_path(path: str) -> Path:
    """Resolve a user-provided path.

    - If the path exists as given, return it.
    - Otherwise try resolving it relative to BASE_DIR.
    """
    p = Path(path)
    if p.exists():
        return p

    p_from_base = BASE_DIR / path
    if p_from_base.exists():
        return p_from_base

    return p


def load_json(path: Path) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not find JSON file at: {path} (tried both as-is and relative to {BASE_DIR})"
        ) from e


def normalize_skill(skill: str) -> str:
    s = str(skill).strip().lower()
    return ROLE_ALIASES.get(s, skill.strip())


def normalize_skill_list(skills: List[str]) -> List[str]:
    result = []
    seen = set()
    for skill in skills:
        normalized = normalize_skill(skill)
        key = normalized.lower()
        if key and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def safe_float_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_skill_overlap(cv_skills: List[str], role_skills: List[str]) -> Tuple[List[str], List[str], float]:
    cv_norm = normalize_skill_list(cv_skills)
    role_norm = normalize_skill_list(role_skills)

    cv_set = {s.lower(): s for s in cv_norm}
    role_set = {s.lower(): s for s in role_norm}

    matched_keys = sorted(set(cv_set.keys()) & set(role_set.keys()))
    missing_keys = sorted(set(role_set.keys()) - set(cv_set.keys()))

    matched = [role_set[k] for k in matched_keys]
    missing = [role_set[k] for k in missing_keys]

    overlap_score = safe_float_div(len(matched), max(len(role_norm), 1))
    return matched, missing, overlap_score


def compute_keyword_match(cv_info: Dict, role_name: str, role_keywords: List[str]) -> float:
    score = 0.0

    text_parts = []
    text_parts.extend(cv_info.get("skills", []))
    text_parts.extend(cv_info.get("projects", []))
    text_parts.extend(cv_info.get("education_signals", []))
    text_parts.append(cv_info.get("target_role", ""))

    joined = " ".join([str(x).lower() for x in text_parts if x])

    keyword_hits = 0
    for kw in role_keywords:
        if kw.lower() in joined:
            keyword_hits += 1

    score += safe_float_div(keyword_hits, max(len(role_keywords), 1))

    target_role = str(cv_info.get("target_role", "")).strip().lower()
    if target_role and target_role != "unknown" and role_name.lower() == target_role:
        score += 0.5

    return min(score, 1.0)


def compute_experience_match(cv_experience_years: str, role_name: str, role_profile: Dict) -> float:
    value = str(cv_experience_years).strip().lower()

    if value == "unknown":
        return 0.4

    try:
        years = int(value)
    except ValueError:
        return 0.4

    common_patterns = " ".join(role_profile.get("common_experience_patterns", [])).lower()

    if role_name in ["Data Analyst", "Data Scientist"] and years <= 2:
        return 1.0
    if role_name == "Data Engineer" and years <= 1:
        return 0.6
    if role_name == "AI Engineer" and years <= 1:
        return 0.6
    if role_name == "AI Researcher" and years <= 1:
        return 0.6
    if role_name == "Data Labeling" and years <= 1:
        return 1.0

    if "fresher" in common_patterns or "không yêu cầu" in common_patterns:
        return 1.0

    if years <= 3:
        return 0.8
    return 0.7

def analyze_cv_against_roles(cv_info, role_profiles):
    cv_skills = set(cv_info.get("skills", []))
    cv_target = cv_info.get("target_role", "Unknown")

    experience_years = safe_parse_experience(cv_info.get("experience_years", 0))

    role_priority = {
        "Data Analyst": 5,
        "Data Scientist": 5,
        "Data Engineer": 5,
        "AI Engineer": 4,
        "AI Researcher": 4,
        "Data Labeling": 1
    }

    role_scores = []

    for role_name, role_data in role_profiles.items():
        role_skills = set(role_data.get("common_skills", []))

        matched_skills = cv_skills & role_skills
        missing_skills = role_skills - cv_skills

        # 1. Skill overlap
        if len(role_skills) > 0:
            skill_overlap_score = len(matched_skills) / len(role_skills)
        else:
            skill_overlap_score = 0.0

        # 2. Target role match
        target_role_match_score = 0.0
        if cv_target != "Unknown" and cv_target == role_name:
            target_role_match_score = 1.0

        # 3. Experience score
        if experience_years >= 3:
            experience_score = 1.0
        elif experience_years >= 1:
            experience_score = 0.6
        else:
            experience_score = 0.3

        # 4. Keyword-like score from matched skill count
        if len(matched_skills) >= 3:
            keyword_score = 1.0
        elif len(matched_skills) >= 1:
            keyword_score = 0.5
        else:
            keyword_score = 0.0

        priority_bonus = role_priority.get(role_name, 1) * 0.01

        final_score = (
            0.5 * skill_overlap_score +
            0.2 * target_role_match_score +
            0.2 * keyword_score +
            0.1 * experience_score +
            priority_bonus
        )

        role_scores.append({
            "role": role_name,
            "score": round(final_score, 3),
            "skill_overlap_score": round(skill_overlap_score, 3),
            "target_role_match_score": round(target_role_match_score, 3),
            "keyword_match_score": round(keyword_score, 3),
            "experience_score": round(experience_score, 3),
            "matched_skills": sorted(list(matched_skills)),
            "missing_skills": sorted(list(missing_skills))
        })

    # Nếu role_profiles rỗng thì trả kết quả an toàn
    if not role_scores:
        return {
            "target_role_from_cv": cv_target,
            "domain_fit": "low",
            "best_fit_roles": [],
            "strengths": [],
            "missing_skills": [],
            "top_role_result": {},
            "role_ranking": []
        }

    role_scores.sort(key=lambda x: x["score"], reverse=True)

    top_role = role_scores[0]
    best_roles = [r["role"] for r in role_scores[:3]]

    matched_count = len(top_role["matched_skills"])
    top_score = top_role["score"]

    # Domain fit logic
    if cv_target == "Unknown" and matched_count < 2:
        domain_fit = "low"
    elif top_score >= 0.6:
        domain_fit = "high"
    elif top_score >= 0.35:
        domain_fit = "medium"
    else:
        domain_fit = "low"

    # Nếu ngoài domain thì fallback missing skills về Data Analyst
    if domain_fit == "low" and "Data Analyst" in role_profiles:
        da_skills = set(role_profiles["Data Analyst"].get("common_skills", []))
        missing_skills = sorted(list(da_skills - cv_skills))
    else:
        missing_skills = top_role["missing_skills"]

    result = {
        "target_role_from_cv": cv_target,
        "domain_fit": domain_fit,
        "best_fit_roles": best_roles,
        "strengths": top_role["matched_skills"],
        "missing_skills": missing_skills[:10],
        "top_role_result": top_role,
        "role_ranking": role_scores[:5]
    }

    return result
def score_role(cv_info: Dict, role_name: str, role_profile: Dict) -> Dict:
    cv_skills = cv_info.get("skills", [])
    role_skills = role_profile.get("common_skills", [])

    matched_skills, missing_skills, skill_overlap_score = compute_skill_overlap(cv_skills, role_skills)

    role_keywords = role_profile.get("common_keywords", [])
    if not role_keywords:
        role_keywords = ROLE_KEYWORD_HINTS.get(role_name, [])

    keyword_match_score = compute_keyword_match(cv_info, role_name, role_keywords)
    experience_match_score = compute_experience_match(
        cv_info.get("experience_years", "Unknown"),
        role_name,
        role_profile,
    )

    target_role_match_score = 0.0
    target_role = str(cv_info.get("target_role", "")).strip().lower()
    if target_role and target_role != "unknown" and role_name.lower() == target_role:
        target_role_match_score = 1.0
    role_priority = {
            "Data Analyst": 5,
            "Data Scientist": 5,
            "Data Engineer": 5,
            "AI Engineer": 4,
            "AI Researcher": 4,
            "Data Labeling": 1
        }

    priority_bonus = role_priority.get(role_name, 1) * 0.01
    final_score = (
        0.5 * skill_overlap_score
        + 0.2 * keyword_match_score
        + 0.2 * experience_match_score
        + 0.1 * target_role_match_score
        + priority_bonus
    )

    return {
        "role_name": role_name,
        "score": round(final_score, 4),
        "skill_overlap_score": round(skill_overlap_score, 4),
        "keyword_match_score": round(keyword_match_score, 4),
        "experience_match_score": round(experience_match_score, 4),
        "target_role_match_score": round(target_role_match_score, 4),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "recommended_next_skills": role_profile.get("recommended_next_skills", []),
        "common_skills": role_profile.get("common_skills", []),
    }


def build_development_plan(best_role_result: Dict, cv_info: Dict) -> List[str]:
    plan = []

    missing_skills = best_role_result.get("missing_skills", [])
    recommended = best_role_result.get("recommended_next_skills", [])

    for skill in missing_skills[:3]:
        plan.append(f"Học hoặc củng cố {skill}")

    for skill in recommended:
        if len(plan) >= 5:
            break
        if all(skill.lower() not in p.lower() for p in plan):
            plan.append(f"Phát triển thêm {skill}")

    projects = cv_info.get("projects", [])
    if not projects:
        plan.append("Bổ sung ít nhất 1 project thực tế vào CV")

    return plan[:5]

def safe_parse_experience(value):
    try:
        if value is None:
            return 0
        text = str(value).strip().lower()
        if text in ["unknown", "", "none", "null"]:
            return 0
        return int(text)
    except Exception:
        return 0



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cv_json",
        default=str(DEFAULT_CV_JSON),
        help=f"Path to extracted CV JSON (default: {DEFAULT_CV_JSON})",
    )
    parser.add_argument(
        "--role_profiles",
        default=str(ROLE_PROFILE_PATH),
        help="Path to role_profiles.json",
    )
    parser.add_argument(
        "--output_path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to save result JSON",
    )
    args = parser.parse_args()

    cv_json_path = resolve_path(args.cv_json)
    role_profile_path = resolve_path(args.role_profiles)
    output_path = resolve_path(args.output_path)

    cv_info = load_json(cv_json_path)
    role_profiles = load_json(role_profile_path)

    result = analyze_cv_against_roles(cv_info, role_profiles)

    print(json.dumps(result, ensure_ascii=True, indent=2))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nSaved gap analysis result to: {output_path}")


if __name__ == "__main__":
    main()
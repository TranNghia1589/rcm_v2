from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Dict, List


PROJECT_SIGNAL_RULES = {
    "dashboard_bi": ["dashboard", "power bi", "tableau", "report", "kpi"],
    "etl_pipeline": ["etl", "pipeline", "airflow", "data warehouse", "spark"],
    "ml_modeling": ["machine learning", "model", "prediction", "classification", "regression"],
    "llm_rag": ["llm", "rag", "prompt", "langchain"],
    "experimentation": ["a/b test", "experiment", "hypothesis"],
}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _parse_experience_years(value: Any) -> int | None:
    raw = _normalize_text(value).lower()
    if not raw or raw in {"unknown", "none", "null"}:
        return None
    nums = re.findall(r"\d+", raw)
    if not nums:
        return None
    return max(int(x) for x in nums)


def _detect_project_signals(text: str) -> List[str]:
    low = text.lower()
    found = []
    for signal, patterns in PROJECT_SIGNAL_RULES.items():
        if any(p in low for p in patterns):
            found.append(signal)
    return found


def build_market_gap_report(
    *,
    cv_info: Dict[str, Any] | None,
    gap_result: Dict[str, Any] | None,
    recommended_jobs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    cv_info = cv_info or {}
    gap_result = gap_result or {}

    cv_skills = {str(x).strip().lower() for x in cv_info.get("skills", []) if str(x).strip()}
    cv_projects_text = " ".join(cv_info.get("projects", []))
    cv_project_signals = set(_detect_project_signals(cv_projects_text))
    cv_exp_years = _parse_experience_years(cv_info.get("experience_years", "Unknown"))

    market_skill_counter = Counter()
    market_exp_values: List[int] = []
    market_project_counter = Counter()

    for job in recommended_jobs:
        for s in job.get("job_skills", []):
            market_skill_counter[str(s).strip().lower()] += 1

        min_exp = job.get("experience_min_years")
        max_exp = job.get("experience_max_years")
        if isinstance(min_exp, (int, float)):
            market_exp_values.append(int(min_exp))
        elif isinstance(max_exp, (int, float)):
            market_exp_values.append(int(max_exp))

        text = " ".join(
            [
                _normalize_text(job.get("requirements_clean", "")),
                _normalize_text(job.get("description_clean", "")),
            ]
        )
        for signal in _detect_project_signals(text):
            market_project_counter[signal] += 1

    missing_skills_ranked = []
    for skill, freq in market_skill_counter.most_common(20):
        if skill not in cv_skills:
            missing_skills_ranked.append({"skill": skill, "frequency_in_top_jobs": freq})

    market_project_signals = [k for k, _ in market_project_counter.most_common(10)]
    missing_project_signals = [s for s in market_project_signals if s not in cv_project_signals]

    expected_exp = round(median(market_exp_values), 1) if market_exp_values else None
    experience_gap_years = None
    if expected_exp is not None and cv_exp_years is not None and cv_exp_years < expected_exp:
        experience_gap_years = round(expected_exp - cv_exp_years, 1)

    roadmap_30_60_90 = {
        "day_1_30": [
            f"Hoc va luyen 2 ky nang uu tien: {', '.join([x['skill'] for x in missing_skills_ranked[:2]])}"
            if missing_skills_ranked
            else "On lai cac ky nang hien co va chuan hoa CV."
        ],
        "day_31_60": [
            f"Xay dung mini-project theo huong: {', '.join(missing_project_signals[:2])}"
            if missing_project_signals
            else "Xay dung 1 project datacenter theo role muc tieu.",
        ],
        "day_61_90": [
            "Nang cap portfolio/CV va ung tuyen vao top job matching.",
            "On bo cau hoi phong van theo ky nang dang thieu.",
        ],
    }

    return {
        "target_role": (gap_result.get("best_fit_roles") or ["Unknown"])[0],
        "top_missing_skills": missing_skills_ranked[:10],
        "market_project_signals": market_project_signals,
        "missing_project_signals": missing_project_signals[:5],
        "experience": {
            "cv_experience_years": cv_exp_years,
            "expected_years_from_top_jobs": expected_exp,
            "experience_gap_years": experience_gap_years,
        },
        "roadmap_30_60_90": roadmap_30_60_90,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv_json", required=True)
    parser.add_argument("--gap_json", required=True)
    parser.add_argument("--recommended_jobs_json", required=True)
    parser.add_argument("--output_path", default="")
    args = parser.parse_args()

    with open(args.cv_json, "r", encoding="utf-8") as f:
        cv_info = json.load(f)
    with open(args.gap_json, "r", encoding="utf-8") as f:
        gap_result = json.load(f)
    with open(args.recommended_jobs_json, "r", encoding="utf-8") as f:
        recommended_jobs = json.load(f)

    report = build_market_gap_report(
        cv_info=cv_info,
        gap_result=gap_result,
        recommended_jobs=recommended_jobs,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.output_path:
        out = Path(args.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Saved gap report to: {out}")


if __name__ == "__main__":
    main()

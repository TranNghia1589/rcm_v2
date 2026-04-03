#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import requests

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.models.matching.gap_engine import build_market_gap_report
from src.models.matching.recommend_engine import find_latest_jobs_file, get_top_job_recommendations


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ask_ollama(prompt: str) -> str:
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json().get("response", "").strip()


def to_internal_context(
    *,
    cv_info: Dict[str, Any],
    gap_result: Dict[str, Any],
    top_jobs: List[Dict[str, Any]],
    gap_report: Dict[str, Any],
) -> str:
    return json.dumps(
        {
            "cv_info": cv_info,
            "gap_result": gap_result,
            "top_jobs": top_jobs,
            "gap_report": gap_report,
        },
        ensure_ascii=False,
        indent=2,
    )


def build_strict_prompt(question: str, context_json: str) -> str:
    return f"""
Báº¡n lÃ  chatbot tÆ° váº¥n CV/job matching.
Chá»‰ Ä‘Æ°á»£c dÃ¹ng dá»¯ liá»‡u trong CONTEXT ná»™i bá»™ dÆ°á»›i Ä‘Ã¢y.
KhÃ´ng Ä‘Æ°á»£c dÃ¹ng kiáº¿n thá»©c ngoÃ i context.
Náº¿u cÃ¢u há»i khÃ´ng cÃ³ dá»¯ liá»‡u Ä‘á»ƒ tráº£ lá»i thÃ¬ nÃ³i rÃµ "KhÃ´ng cÃ³ dá»¯ liá»‡u ná»™i bá»™ phÃ¹ há»£p".
Tráº£ lá»i báº±ng tiáº¿ng Viá»‡t, ngáº¯n gá»n, cÃ³ bullet náº¿u cáº§n.

CONTEXT:
{context_json}

CÃ‚U Há»ŽI:
{question}
""".strip()


def detect_internal_intent(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ["job", "viá»‡c", "vá»‹ trÃ­", "á»©ng tuyá»ƒn", "top"]):
        return "job_match"
    if any(k in q for k in ["thiáº¿u", "gap", "cÃ²n thiáº¿u"]):
        return "gap_focus"
    if any(k in q for k in ["cáº£i thiá»‡n cv", "sá»­a cv", "review cv", "nÃªn há»c", "roadmap", "lá»™ trÃ¬nh"]):
        return "improve_cv"
    if any(k in q for k in ["kinh nghiá»‡m", "experience"]):
        return "experience_focus"
    return "summary"


def internal_fallback_answer(question: str, gap_result: Dict[str, Any], top_jobs: List[Dict[str, Any]], gap_report: Dict[str, Any]) -> str:
    role = (gap_result.get("best_fit_roles") or ["Unknown"])[0]
    domain_fit = gap_result.get("domain_fit", "unknown")
    strengths = gap_result.get("strengths", [])
    top_missing = [x.get("skill", "") for x in gap_report.get("top_missing_skills", []) if x.get("skill")]
    missing_projects = gap_report.get("missing_project_signals", [])
    exp = gap_report.get("experience", {})
    intent = detect_internal_intent(question)

    lines = [f"Dá»±a trÃªn dá»¯ liá»‡u ná»™i bá»™, mÃ¬nh phÃ¢n tÃ­ch cho cÃ¢u há»i: {question}"]
    lines.append(f"Hiá»‡n táº¡i CV phÃ¹ há»£p nháº¥t vá»›i vai trÃ² {role} (domain_fit={domain_fit}).")

    if intent == "job_match":
        lines.append("\nTop job phÃ¹ há»£p Ä‘á»ƒ á»©ng tuyá»ƒn:")
        for idx, job in enumerate(top_jobs[:10], 1):
            lines.append(
                f"{idx}. {job.get('job_title', 'Unknown')} | {job.get('company_name', 'N/A')} | "
                f"{job.get('location', 'N/A')} | score={job.get('score', 0)}"
            )

    elif intent == "gap_focus":
        lines.append("\nCÃ¡c Ä‘iá»ƒm cÃ²n thiáº¿u quan trá»ng:")
        if top_missing:
            lines.extend([f"- {x}" for x in top_missing[:8]])
        else:
            lines.append("- ChÆ°a phÃ¡t hiá»‡n thiáº¿u há»¥t rÃµ rÃ ng tá»« top jobs.")

    elif intent == "experience_focus":
        lines.append("\nSo sÃ¡nh kinh nghiá»‡m vá»›i thá»‹ trÆ°á»ng:")
        lines.append(f"- Kinh nghiá»‡m trong CV: {exp.get('cv_experience_years', 'Unknown')} nÄƒm")
        lines.append(f"- Má»©c ká»³ vá»ng tá»« top jobs: {exp.get('expected_years_from_top_jobs', 'Unknown')} nÄƒm")
        lines.append(f"- Khoáº£ng cÃ¡ch: {exp.get('experience_gap_years', 0)} nÄƒm")

    else:
        lines.append("\nÄiá»ƒm máº¡nh hiá»‡n táº¡i:")
        if strengths:
            lines.extend([f"- {x}" for x in strengths[:6]])
        else:
            lines.append("- ChÆ°a cÃ³ nhiá»u tÃ­n hiá»‡u ká»¹ nÄƒng máº¡nh.")

        lines.append("\nÄiá»ƒm cáº§n cáº£i thiá»‡n:")
        if top_missing:
            lines.extend([f"- {x}" for x in top_missing[:6]])
        else:
            lines.append("- ChÆ°a phÃ¡t hiá»‡n thiáº¿u há»¥t rÃµ rÃ ng.")

        if missing_projects:
            lines.append("\nLoáº¡i project nÃªn bá»• sung:")
            lines.extend([f"- {x}" for x in missing_projects[:4]])

        roadmap = gap_report.get("roadmap_30_60_90", {})
        if roadmap:
            lines.append("\nGá»£i Ã½ lá»™ trÃ¬nh 30-60-90 ngÃ y:")
            for phase in ["day_1_30", "day_31_60", "day_61_90"]:
                tasks = roadmap.get(phase, [])
                if tasks:
                    lines.append(f"- {phase}: {tasks[0]}")

    lines.append("\nGhi chÃº: cÃ¢u tráº£ lá»i nÃ y Ä‘Æ°á»£c táº¡o hoÃ n toÃ n tá»« dá»¯ liá»‡u ná»™i bá»™ cá»§a báº¡n.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv_json", required=True)
    parser.add_argument("--gap_result", required=True)
    parser.add_argument("--jobs_path", default="")
    parser.add_argument("--top_k_jobs", type=int, default=10)
    parser.add_argument("--allow_external_llm", action="store_true")
    args = parser.parse_args()

    base_dir = BASE_DIR
    cv_info = load_json(args.cv_json)
    gap_result = load_json(args.gap_result)
    jobs_path = Path(args.jobs_path) if args.jobs_path else find_latest_jobs_file(base_dir)

    top_jobs = get_top_job_recommendations(
        cv_info=cv_info,
        gap_result=gap_result,
        jobs_path=jobs_path,
        top_k=args.top_k_jobs,
    )
    gap_report = build_market_gap_report(
        cv_info=cv_info,
        gap_result=gap_result,
        recommended_jobs=top_jobs,
    )
    context_json = to_internal_context(
        cv_info=cv_info,
        gap_result=gap_result,
        top_jobs=top_jobs,
        gap_report=gap_report,
    )

    print("\n=== CHATBOT 1:1 (type 'exit' to quit) ===")
    while True:
        question = input("\nBáº¡n: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Káº¿t thÃºc phiÃªn chat.")
            break

        if args.allow_external_llm:
            try:
                answer = ask_ollama(build_strict_prompt(question, context_json))
                if not answer:
                    answer = internal_fallback_answer(question, gap_result, top_jobs, gap_report)
            except Exception:
                answer = internal_fallback_answer(question, gap_result, top_jobs, gap_report)
        else:
            answer = internal_fallback_answer(question, gap_result, top_jobs, gap_report)

        print(f"\nBot: {answer}")


if __name__ == "__main__":
    main()



import argparse
import json
from pathlib import Path

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_non_data_ai_background(gap_result: dict) -> str:
    """
    Suy Ä‘oÃ¡n xem CV hiá»‡n táº¡i cÃ³ Ä‘ang ngoÃ i domain Data/AI hay khÃ´ng.
    """
    domain_fit = gap_result.get("domain_fit", "unknown")
    strengths = [str(x).lower() for x in gap_result.get("strengths", [])]
    target_role = str(gap_result.get("target_role_from_cv", "Unknown")).lower()

    data_ai_keywords = [
        "python", "sql", "excel", "power bi", "tableau", "statistics",
        "machine learning", "deep learning", "pytorch", "tensorflow",
        "etl", "airflow", "spark", "dashboarding", "data analysis",
        "nlp", "computer vision", "rag", "llm"
    ]

    matched = sum(1 for s in strengths if s.lower() in data_ai_keywords)

    if domain_fit == "low" and matched == 0:
        if target_role and target_role != "unknown":
            return f"Há»“ sÆ¡ hiá»‡n táº¡i cá»§a báº¡n chÆ°a thá»ƒ hiá»‡n rÃµ Ä‘á»‹nh hÆ°á»›ng Data/AI vÃ  Ä‘ang nghiÃªng nhiá»u hÆ¡n vá» hÆ°á»›ng {target_role}."
        return "Há»“ sÆ¡ hiá»‡n táº¡i cá»§a báº¡n chÆ°a thá»ƒ hiá»‡n rÃµ Ä‘á»‹nh hÆ°á»›ng Data/AI vÃ  cÃ³ xu hÆ°á»›ng thuá»™c lÄ©nh vá»±c khÃ¡c."

    return ""

ROLE_ALIASES = {
    "da": "Data Analyst",
    "data analyst": "Data Analyst",

    "de": "Data Engineer",
    "data engineer": "Data Engineer",

    "ds": "Data Scientist",
    "data scientist": "Data Scientist",

    "ai": "AI Engineer",
    "ai engineer": "AI Engineer",

    "mle": "AI Engineer",
    "ml engineer": "AI Engineer",

    "bi": "BI Analyst",
    "bi analyst": "BI Analyst",

    "ba": "Business Analyst",
    "business analyst": "Business Analyst"
}
def normalize_role_from_question(question: str):
    q = question.lower()

    for alias, role in ROLE_ALIASES.items():
        if alias in q:
            return role

    return None

def build_structured_context(gap_result: dict) -> dict:
    top_role_result = gap_result.get("top_role_result", {})

    return {
        "domain_fit": gap_result.get("domain_fit", "unknown"),
        "target_role_from_cv": gap_result.get("target_role_from_cv", "Unknown"),
        "best_fit_roles": gap_result.get("best_fit_roles", []),
        "strengths": gap_result.get("strengths", []),
        "missing_skills": gap_result.get("missing_skills", []),
        "development_plan": gap_result.get("development_plan", []),
        "top_role_result": {
            "role": top_role_result.get("role", ""),
            "score": top_role_result.get("score", 0),
            "skill_overlap_score": top_role_result.get("skill_overlap_score", 0),
            "keyword_match_score": top_role_result.get("keyword_match_score", 0),
            "experience_score": top_role_result.get("experience_score", 0),
            "target_role_match_score": top_role_result.get("target_role_match_score", 0),
            "matched_skills": top_role_result.get("matched_skills", []),
            "missing_skills": top_role_result.get("missing_skills", [])
        },
        "role_ranking": gap_result.get("role_ranking", [])
    }

def build_prompt(gap_result: dict, user_question: str) -> str:
    structured_context = build_structured_context(gap_result)
    domain_hint = detect_non_data_ai_background(gap_result)

    return f"""
Báº¡n lÃ  chatbot tÆ° váº¥n nghá» nghiá»‡p vÃ  phÃ¢n tÃ­ch CV cho nhÃ³m ngÃ nh Data/AI.

NGUYÃŠN Táº®C:
- Tráº£ lá»i báº±ng tiáº¿ng Viá»‡t.
- KhÃ´ng bá»‹a thÃ´ng tin ngoÃ i dá»¯ liá»‡u Ä‘Æ°á»£c cung cáº¥p.
- KhÃ´ng dÃ¹ng lá»i khuyÃªn chung chung náº¿u dá»¯ liá»‡u Ä‘Ã£ cÃ³ thÃ´ng tin cá»¥ thá»ƒ.
- Pháº£i bÃ¡m sÃ¡t vÃ o: best_fit_roles, strengths, missing_skills, development_plan, domain_fit.
- Náº¿u CV chÆ°a phÃ¹ há»£p vá»›i nhÃ³m Data/AI, hÃ£y nÃ³i rÃµ nhÆ°ng lá»‹ch sá»± vÃ  thá»±c táº¿.
- Náº¿u cÃ³ thá»ƒ, hÃ£y nÃªu thá»© tá»± Æ°u tiÃªn há»c ká»¹ nÄƒng.

Gá»¢I Ã NGá»® Cáº¢NH NGOÃ€I DOMAIN:
{domain_hint if domain_hint else "KhÃ´ng cÃ³ ghi chÃº Ä‘áº·c biá»‡t."}

Báº®T BUá»˜C tráº£ lá»i Ä‘Ãºng theo 5 má»¥c sau vÃ  giá»¯ nguyÃªn thá»© tá»±:

1. Má»©c Ä‘á»™ phÃ¹ há»£p
- NÃªu CV hiá»‡n táº¡i phÃ¹ há»£p á»Ÿ má»©c nÃ o vá»›i role má»¥c tiÃªu.
- Náº¿u domain_fit lÃ  low, pháº£i nÃ³i rÃµ CV hiá»‡n chÆ°a thuá»™c nhÃ³m Data/AI rÃµ rÃ ng.

2. Äiá»ƒm máº¡nh hiá»‡n táº¡i
- Chá»‰ nÃªu cÃ¡c Ä‘iá»ƒm máº¡nh tháº­t sá»± cÃ³ trong strengths hoáº·c matched_skills.
- Náº¿u strengths Ã­t hoáº·c rá»—ng, nÃ³i ngáº¯n gá»n ráº±ng CV chÆ°a cÃ³ nhiá»u tÃ­n hiá»‡u máº¡nh cho role má»¥c tiÃªu.

3. Äiá»ƒm cÃ²n thiáº¿u
- Chá»‰ nÃªu cÃ¡c thiáº¿u há»¥t cá»¥ thá»ƒ tá»« missing_skills.
- KhÃ´ng biáº¿n ngÃ nh/lÄ©nh vá»±c thÃ nh ká»¹ nÄƒng.

4. Ká»¹ nÄƒng nÃªn phÃ¡t triá»ƒn tiáº¿p
- Æ¯u tiÃªn theo development_plan vÃ  recommended_next_skills.
- NÃªu theo thá»© tá»± há»c há»£p lÃ½ náº¿u cÃ³ thá»ƒ.

5. HÃ nh Ä‘á»™ng Ä‘á» xuáº¥t trong 1â€“3 thÃ¡ng
- ÄÆ°a ra hÃ nh Ä‘á»™ng cá»¥ thá»ƒ, thá»±c táº¿, ngáº¯n gá»n.
- VÃ­ dá»¥: há»c cÃ´ng cá»¥, lÃ m project, bá»• sung CV, thá»±c táº­p, portfolio.

Dá»® LIá»†U PHÃ‚N TÃCH:
{json.dumps(structured_context, ensure_ascii=False, indent=2)}

CÃ‚U Há»ŽI NGÆ¯á»œI DÃ™NG:
{user_question}
""".strip()


def ask_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()


def fallback_response(gap_result: dict, user_question: str) -> str:
    roles = gap_result.get("best_fit_roles", [])
    strengths = gap_result.get("strengths", [])
    missing = gap_result.get("missing_skills", [])
    plan = gap_result.get("development_plan", [])
    domain_fit = gap_result.get("domain_fit", "unknown")
    domain_hint = detect_non_data_ai_background(gap_result)

    lines = []

    # 1. Má»©c Ä‘á»™ phÃ¹ há»£p
    lines.append("1. Má»©c Ä‘á»™ phÃ¹ há»£p")
    if roles:
        if domain_fit == "low":
            if domain_hint:
                lines.append(domain_hint)
            lines.append(f"Role gáº§n nháº¥t hiá»‡n táº¡i trong nhÃ³m Data/AI lÃ  {roles[0]}, nhÆ°ng má»©c Ä‘á»™ phÃ¹ há»£p cÃ²n tháº¥p.")
        elif domain_fit == "medium":
            lines.append(f"CV cá»§a báº¡n hiá»‡n phÃ¹ há»£p á»Ÿ má»©c trung bÃ¬nh vá»›i vá»‹ trÃ­ {roles[0]}.")
        else:
            lines.append(f"CV cá»§a báº¡n hiá»‡n phÃ¹ há»£p khÃ¡ tá»‘t vá»›i vá»‹ trÃ­ {roles[0]}.")
    else:
        lines.append("ChÆ°a xÃ¡c Ä‘á»‹nh Ä‘Æ°á»£c role phÃ¹ há»£p rÃµ rÃ ng.")

    # 2. Äiá»ƒm máº¡nh hiá»‡n táº¡i
    lines.append("\n2. Äiá»ƒm máº¡nh hiá»‡n táº¡i")
    if strengths:
        for s in strengths[:5]:
            lines.append(f"- {s}")
    else:
        lines.append("- CV hiá»‡n chÆ°a cÃ³ nhiá»u tÃ­n hiá»‡u máº¡nh khá»›p vá»›i role Data/AI má»¥c tiÃªu.")

    # 3. Äiá»ƒm cÃ²n thiáº¿u
    lines.append("\n3. Äiá»ƒm cÃ²n thiáº¿u")
    if missing:
        for m in missing[:5]:
            lines.append(f"- {m}")
    else:
        lines.append("- ChÆ°a phÃ¡t hiá»‡n thiáº¿u há»¥t cá»¥ thá»ƒ ná»•i báº­t.")

    # 4. Ká»¹ nÄƒng nÃªn phÃ¡t triá»ƒn tiáº¿p
    lines.append("\n4. Ká»¹ nÄƒng nÃªn phÃ¡t triá»ƒn tiáº¿p")
    if plan:
        for p in plan[:5]:
            lines.append(f"- {p}")
    else:
        lines.append("- NÃªn Æ°u tiÃªn há»c thÃªm ká»¹ nÄƒng ná»n táº£ng theo role má»¥c tiÃªu.")

    # 5. HÃ nh Ä‘á»™ng Ä‘á» xuáº¥t trong 1-3 thÃ¡ng
    lines.append("\n5. HÃ nh Ä‘á»™ng Ä‘á» xuáº¥t trong 1â€“3 thÃ¡ng")
    suggested_actions = []

    if missing:
        if any("excel" in m.lower() for m in missing):
            suggested_actions.append("- Ã”n hoáº·c há»c Excel nÃ¢ng cao vÃ  thá»±c hÃ nh vá»›i dá»¯ liá»‡u thá»±c táº¿.")
        if any("sql" in m.lower() for m in missing):
            suggested_actions.append("- Há»c SQL cÆ¡ báº£n Ä‘áº¿n trung cáº¥p vÃ  luyá»‡n truy váº¥n trÃªn dataset tháº­t.")
        if any("power bi" in m.lower() or "tableau" in m.lower() for m in missing):
            suggested_actions.append("- LÃ m Ã­t nháº¥t 1 project dashboard báº±ng Power BI hoáº·c Tableau.")
        if any("statistics" in m.lower() for m in missing):
            suggested_actions.append("- Ã”n láº¡i xÃ¡c suáº¥t thá»‘ng kÃª vÃ  cÃ¡c chá»‰ sá»‘ mÃ´ táº£ cÆ¡ báº£n.")

    suggested_actions.append("- Bá»• sung 1â€“2 project dá»¯ liá»‡u vÃ o CV náº¿u hiá»‡n chÆ°a cÃ³ project liÃªn quan.")
    suggested_actions.append("- Viáº¿t láº¡i CV theo hÆ°á»›ng nháº¥n máº¡nh ká»¹ nÄƒng vÃ  dá»± Ã¡n liÃªn quan Ä‘áº¿n role má»¥c tiÃªu.")

    seen = set()
    dedup_actions = []
    for action in suggested_actions:
        if action not in seen:
            seen.add(action)
            dedup_actions.append(action)

    for action in dedup_actions[:5]:
        lines.append(action)

    lines.append("\n[Ghi chÃº] Há»‡ thá»‘ng Ä‘ang dÃ¹ng cháº¿ Ä‘á»™ dá»± phÃ²ng vÃ¬ chÆ°a gá»i Ä‘Æ°á»£c Ollama/Llama 3.")
    return "\n".join(lines)
def fallback_cv_or_career_answer(gap_result: dict, intent: str) -> str:
    roles = gap_result.get("best_fit_roles", [])
    strengths = gap_result.get("strengths", [])
    missing = gap_result.get("missing_skills", [])
    plan = gap_result.get("development_plan", [])
    domain_fit = gap_result.get("domain_fit", "unknown")
    top_role = gap_result.get("top_role_result", {})
    score = top_role.get("score", 0)

    lines = []

    if intent == "cv_analysis":
        lines.append("1. Má»©c Ä‘á»™ phÃ¹ há»£p")
    else:
        lines.append("1. Má»¥c tiÃªu phÃ¹ há»£p hiá»‡n táº¡i")

    if roles:
        if domain_fit == "low":
            lines.append(
                f"CV hiá»‡n táº¡i chÆ°a phÃ¹ há»£p máº¡nh vá»›i nhÃ³m Data/AI. "
                f"Role gáº§n nháº¥t hiá»‡n táº¡i lÃ  {roles[0]} vá»›i má»©c phÃ¹ há»£p tháº¥p (score: {score})."
            )
        elif domain_fit == "medium":
            lines.append(
                f"CV hiá»‡n táº¡i phÃ¹ há»£p á»Ÿ má»©c trung bÃ¬nh vá»›i vá»‹ trÃ­ {roles[0]} "
                f"(score: {score}). Báº¡n Ä‘Ã£ cÃ³ má»™t pháº§n ná»n táº£ng nhÆ°ng váº«n cÃ²n thiáº¿u cÃ¡c ká»¹ nÄƒng quan trá»ng."
            )
        else:
            lines.append(
                f"CV hiá»‡n táº¡i phÃ¹ há»£p khÃ¡ tá»‘t vá»›i vá»‹ trÃ­ {roles[0]} "
                f"(score: {score})."
            )
    else:
        lines.append("ChÆ°a xÃ¡c Ä‘á»‹nh Ä‘Æ°á»£c role phÃ¹ há»£p rÃµ rÃ ng.")

    lines.append("\n2. Äiá»ƒm máº¡nh hiá»‡n táº¡i")
    if strengths:
        for s in strengths[:5]:
            lines.append(f"- {s}")
    else:
        lines.append("- CV hiá»‡n chÆ°a cÃ³ nhiá»u tÃ­n hiá»‡u máº¡nh khá»›p vá»›i role má»¥c tiÃªu.")

    lines.append("\n3. Äiá»ƒm cÃ²n thiáº¿u" if intent == "cv_analysis" else "\n3. Äiá»ƒm cáº§n bÃ¹ Ä‘áº¯p")
    if missing:
        for m in missing[:5]:
            lines.append(f"- {m}")
    else:
        lines.append("- ChÆ°a phÃ¡t hiá»‡n thiáº¿u há»¥t ná»•i báº­t.")

    lines.append("\n4. Ká»¹ nÄƒng nÃªn phÃ¡t triá»ƒn tiáº¿p" if intent == "cv_analysis" else "\n4. Káº¿ hoáº¡ch há»c ká»¹ nÄƒng")
    if plan:
        for p in plan[:5]:
            lines.append(f"- {p}")
    else:
        lines.append("- NÃªn Æ°u tiÃªn há»c thÃªm ká»¹ nÄƒng ná»n táº£ng theo role má»¥c tiÃªu.")

    lines.append("\n5. HÃ nh Ä‘á»™ng Ä‘á» xuáº¥t trong 1â€“3 thÃ¡ng")

    actions = []
    if "SQL" in missing:
        actions.append("- Há»c SQL cÆ¡ báº£n Ä‘áº¿n trung cáº¥p vÃ  luyá»‡n truy váº¥n vá»›i dataset tháº­t.")
    if "Excel" in missing:
        actions.append("- Ã”n hoáº·c há»c Excel nÃ¢ng cao, Ä‘áº·c biá»‡t lÃ  xá»­ lÃ½ vÃ  tá»•ng há»£p dá»¯ liá»‡u.")
    if "Power BI" in missing or "Tableau" in missing:
        actions.append("- LÃ m Ã­t nháº¥t 1 project dashboard báº±ng Power BI hoáº·c Tableau.")
    if "Statistics" in missing:
        actions.append("- Ã”n láº¡i xÃ¡c suáº¥t thá»‘ng kÃª vÃ  cÃ¡c chá»‰ sá»‘ mÃ´ táº£ cÆ¡ báº£n.")

    actions.append("- Bá»• sung 1â€“2 project dá»¯ liá»‡u vÃ o CV náº¿u hiá»‡n chÆ°a cÃ³ project liÃªn quan.")
    actions.append("- Viáº¿t láº¡i CV theo hÆ°á»›ng nháº¥n máº¡nh ká»¹ nÄƒng vÃ  dá»± Ã¡n liÃªn quan Ä‘áº¿n role má»¥c tiÃªu.")

    seen = set()
    for action in actions:
        if action not in seen:
            seen.add(action)
            lines.append(action)

    lines.append("\n[Ghi chÃº] Há»‡ thá»‘ng Ä‘ang dÃ¹ng cháº¿ Ä‘á»™ dá»± phÃ²ng vÃ¬ chÆ°a gá»i Ä‘Æ°á»£c Ollama/Llama 3.")
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap_result", required=True, help="Path to gap analysis result JSON")
    parser.add_argument("--question", required=True, help="User question for the chatbot")
    args = parser.parse_args()

    gap_result_path = Path(args.gap_result)
    if not gap_result_path.exists():
        raise FileNotFoundError(f"KhÃ´ng tÃ¬m tháº¥y file gap result: {gap_result_path}")

    gap_result = load_json(str(gap_result_path))
    prompt = build_prompt(gap_result, args.question)

    try:
        answer = ask_ollama(prompt)
        if not answer:
            answer = fallback_response(gap_result, args.question)
    except Exception as e:
        print(f"[Warning] KhÃ´ng gá»i Ä‘Æ°á»£c Ollama: {e}")
        answer = fallback_response(gap_result, args.question)

    print("\n===== CHATBOT ANSWER =====\n")
    print(answer)


if __name__ == "__main__":
    main()
"""Legacy chatbot script kept for reference only.

Active API chatbot flow lives under `apps/api/app/services/rag/`.
"""


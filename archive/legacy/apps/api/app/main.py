import argparse
import json
import sys
from pathlib import Path

import requests

# Fix encoding for Vietnamese text on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def classify_question(question: str) -> str:
    q = question.lower()

    cv_keywords = [
        "cv", "resume", "há»“ sÆ¡", "thiáº¿u gÃ¬", "thiáº¿u ká»¹ nÄƒng",
        "phÃ¹ há»£p nghá»", "há»£p nghá» nÃ o", "Ä‘iá»ƒm máº¡nh", "Ä‘iá»ƒm yáº¿u",
        "dá»±a trÃªn cv", "dá»±a trÃªn há»“ sÆ¡", "á»©ng tuyá»ƒn", "há»£p vá»›i data analyst khÃ´ng"
    ]

    career_keywords = [
        "nÃªn há»c gÃ¬", "roadmap", "lá»™ trÃ¬nh", "nÃªn phÃ¡t triá»ƒn gÃ¬",
        "nÃªn lÃ m project gÃ¬", "phÃ¡t triá»ƒn ká»¹ nÄƒng", "3 thÃ¡ng", "6 thÃ¡ng",
        "Ä‘á»ƒ trá»Ÿ thÃ nh", "Ä‘á»ƒ theo", "Ä‘á»‹nh hÆ°á»›ng nghá» nghiá»‡p", "nÃªn há»c trÆ°á»›c"
    ]

    for kw in cv_keywords:
        if kw in q:
            return "cv_analysis"

    for kw in career_keywords:
        if kw in q:
            return "career_advice"

    return "general_question"


def detect_non_data_ai_background(gap_result: dict) -> str:
    domain_fit = gap_result.get("domain_fit", "unknown")
    strengths = [str(x).lower() for x in gap_result.get("strengths", [])]
    target_role = str(gap_result.get("target_role_from_cv", "Unknown")).lower()

    data_ai_keywords = [
        "python", "sql", "excel", "power bi", "tableau", "statistics",
        "machine learning", "deep learning", "pytorch", "tensorflow",
        "etl", "airflow", "spark", "dashboarding", "data analysis",
        "nlp", "computer vision", "rag", "llm"
    ]

    matched = sum(1 for s in strengths if s in data_ai_keywords)

    if domain_fit == "low" and matched == 0:
        if target_role and target_role != "unknown":
            return (
                f"Há»“ sÆ¡ hiá»‡n táº¡i cá»§a ngÆ°á»i dÃ¹ng chÆ°a thá»ƒ hiá»‡n rÃµ Ä‘á»‹nh hÆ°á»›ng Data/AI "
                f"vÃ  Ä‘ang nghiÃªng nhiá»u hÆ¡n vá» hÆ°á»›ng {target_role}."
            )
        return (
            "Há»“ sÆ¡ hiá»‡n táº¡i cá»§a ngÆ°á»i dÃ¹ng chÆ°a thá»ƒ hiá»‡n rÃµ Ä‘á»‹nh hÆ°á»›ng Data/AI "
            "vÃ  cÃ³ xu hÆ°á»›ng thuá»™c lÄ©nh vá»±c khÃ¡c."
        )

    return ""


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
            "role_name": top_role_result.get("role_name", ""),
            "score": top_role_result.get("score", 0),
            "matched_skills": top_role_result.get("matched_skills", []),
            "missing_skills": top_role_result.get("missing_skills", []),
            "recommended_next_skills": top_role_result.get("recommended_next_skills", []),
        },
    }


def build_cv_prompt(gap_result: dict, user_question: str) -> str:
    structured_context = build_structured_context(gap_result)
    domain_hint = detect_non_data_ai_background(gap_result)

    return f"""
Báº¡n lÃ  chatbot tÆ° váº¥n nghá» nghiá»‡p vÃ  phÃ¢n tÃ­ch CV cho nhÃ³m ngÃ nh Data/AI.

NGUYÃŠN Táº®C:
- Tráº£ lá»i báº±ng tiáº¿ng Viá»‡t.
- Chá»‰ sá»­ dá»¥ng thÃ´ng tin cÃ³ trong dá»¯ liá»‡u phÃ¢n tÃ­ch.
- KhÃ´ng bá»‹a thÃªm ká»¹ nÄƒng, kinh nghiá»‡m hoáº·c ngÃ nh nghá» ngoÃ i dá»¯ liá»‡u.
- KhÃ´ng láº·p Ã½ giá»¯a cÃ¡c má»¥c.
- Náº¿u CV chÆ°a phÃ¹ há»£p vá»›i Data/AI, hÃ£y nÃ³i rÃµ nhÆ°ng lá»‹ch sá»±.
- Æ¯u tiÃªn nÃªu ká»¹ nÄƒng cá»¥ thá»ƒ thay vÃ¬ nÃ³i chung chung.

Báº®T BUá»˜C tráº£ lá»i theo Ä‘Ãºng 5 má»¥c dÆ°á»›i Ä‘Ã¢y:

1. Má»©c Ä‘á»™ phÃ¹ há»£p
- NÃªu ngáº¯n gá»n CV hiá»‡n táº¡i phÃ¹ há»£p á»Ÿ má»©c nÃ o vá»›i role má»¥c tiÃªu.
- Náº¿u domain_fit lÃ  low, pháº£i nÃ³i rÃµ CV chÆ°a cÃ³ nhiá»u tÃ­n hiá»‡u Data/AI.

2. Äiá»ƒm máº¡nh hiá»‡n táº¡i
- Chá»‰ nÃªu cÃ¡c Ä‘iá»ƒm máº¡nh tháº­t sá»± cÃ³ trong strengths hoáº·c matched_skills.
- Náº¿u strengths Ã­t hoáº·c rá»—ng, nÃ³i ngáº¯n gá»n ráº±ng CV chÆ°a cÃ³ nhiá»u tÃ­n hiá»‡u máº¡nh cho role má»¥c tiÃªu.

3. Äiá»ƒm cÃ²n thiáº¿u
- Chá»‰ nÃªu cÃ¡c ká»¹ nÄƒng cÃ²n thiáº¿u tá»« missing_skills.
- KhÃ´ng biáº¿n lÄ©nh vá»±c hoáº·c ngÃ nh thÃ nh ká»¹ nÄƒng.

4. Ká»¹ nÄƒng nÃªn phÃ¡t triá»ƒn tiáº¿p
- Æ¯u tiÃªn theo development_plan.
- Náº¿u cÃ³ thá»ƒ, sáº¯p xáº¿p theo thá»© tá»± há»c há»£p lÃ½.

5. HÃ nh Ä‘á»™ng Ä‘á» xuáº¥t trong 1â€“3 thÃ¡ng
- ÄÆ°a ra hÃ nh Ä‘á»™ng cá»¥ thá»ƒ, thá»±c táº¿, ngáº¯n gá»n.
- VÃ­ dá»¥: há»c cÃ´ng cá»¥, lÃ m project, cáº­p nháº­t CV, thá»±c táº­p.

GHI CHÃš NGOÃ€I DOMAIN:
{domain_hint if domain_hint else "KhÃ´ng cÃ³ ghi chÃº Ä‘áº·c biá»‡t."}

Dá»® LIá»†U PHÃ‚N TÃCH:
{json.dumps(structured_context, ensure_ascii=False, indent=2)}

CÃ‚U Há»ŽI NGÆ¯á»œI DÃ™NG:
{user_question}
""".strip()


def build_career_prompt(gap_result: dict, user_question: str) -> str:
    structured_context = build_structured_context(gap_result)
    domain_hint = detect_non_data_ai_background(gap_result)

    return f"""
Báº¡n lÃ  chatbot tÆ° váº¥n Ä‘á»‹nh hÆ°á»›ng nghá» nghiá»‡p trong lÄ©nh vá»±c Data/AI.

NGUYÃŠN Táº®C:
- Tráº£ lá»i báº±ng tiáº¿ng Viá»‡t.
- KhÃ´ng bá»‹a thÃ´ng tin ngoÃ i dá»¯ liá»‡u.
- KhÃ´ng láº·p Ã½.
- Pháº£i cá»¥ thá»ƒ, thá»±c táº¿, cÃ³ thá»© tá»± Æ°u tiÃªn.
- Táº­p trung vÃ o ká»¹ nÄƒng cáº§n há»c, project nÃªn lÃ m vÃ  hÆ°á»›ng Ä‘i phÃ¹ há»£p hÆ¡n náº¿u cÃ³.

Báº®T BUá»˜C tráº£ lá»i theo Ä‘Ãºng 5 má»¥c:
1. Má»¥c tiÃªu phÃ¹ há»£p hiá»‡n táº¡i
2. Äiá»ƒm máº¡nh hiá»‡n táº¡i
3. Äiá»ƒm cáº§n bÃ¹ Ä‘áº¯p
4. Káº¿ hoáº¡ch há»c ká»¹ nÄƒng
5. HÃ nh Ä‘á»™ng cá»¥ thá»ƒ trong 1â€“3 thÃ¡ng

GHI CHÃš NGOÃ€I DOMAIN:
{domain_hint if domain_hint else "KhÃ´ng cÃ³ ghi chÃº Ä‘áº·c biá»‡t."}

Dá»® LIá»†U PHÃ‚N TÃCH:
{json.dumps(structured_context, ensure_ascii=False, indent=2)}

CÃ‚U Há»ŽI:
{user_question}
""".strip()


def build_general_prompt(user_question: str) -> str:
    return f"""
Báº¡n lÃ  trá»£ lÃ½ tÆ° váº¥n nghá» nghiá»‡p trong lÄ©nh vá»±c Data/AI.

YÃŠU Cáº¦U:
- Tráº£ lá»i báº±ng tiáº¿ng Viá»‡t.
- Dá»… hiá»ƒu, chÃ­nh xÃ¡c, sÃºc tÃ­ch.
- Náº¿u lÃ  cÃ¢u há»i ná»n táº£ng, hÃ£y giáº£i thÃ­ch nhÆ° cho ngÆ°á»i má»›i há»c.
- Náº¿u cÃ³ thá»ƒ, Ä‘Æ°a vÃ­ dá»¥ ngáº¯n.

CÃ‚U Há»ŽI:
{user_question}
""".strip()


def fallback_cv_or_career_answer(gap_result: dict, intent: str) -> str:
    roles = gap_result.get("best_fit_roles", [])
    strengths = gap_result.get("strengths", [])
    missing = gap_result.get("missing_skills", [])
    plan = gap_result.get("development_plan", [])
    domain_fit = gap_result.get("domain_fit", "unknown")
    domain_hint = detect_non_data_ai_background(gap_result)

    lines = []

    if intent == "cv_analysis":
        section5 = "5. HÃ nh Ä‘á»™ng Ä‘á» xuáº¥t trong 1â€“3 thÃ¡ng"
    else:
        section5 = "5. HÃ nh Ä‘á»™ng cá»¥ thá»ƒ trong 1â€“3 thÃ¡ng"

    # 1
    lines.append("1. Má»©c Ä‘á»™ phÃ¹ há»£p" if intent == "cv_analysis" else "1. Má»¥c tiÃªu phÃ¹ há»£p hiá»‡n táº¡i")
    if roles:
        if domain_fit == "low":
            if domain_hint:
                lines.append(domain_hint)
            lines.append(f"Vai trÃ² gáº§n nháº¥t hiá»‡n táº¡i trong nhÃ³m Data/AI lÃ  {roles[0]}, nhÆ°ng má»©c Ä‘á»™ phÃ¹ há»£p cÃ²n tháº¥p.")
        elif domain_fit == "medium":
            lines.append(f"CV hiá»‡n táº¡i phÃ¹ há»£p á»Ÿ má»©c trung bÃ¬nh vá»›i vá»‹ trÃ­ {roles[0]}.")
        else:
            lines.append(f"CV hiá»‡n táº¡i phÃ¹ há»£p khÃ¡ tá»‘t vá»›i vá»‹ trÃ­ {roles[0]}.")
    else:
        lines.append("ChÆ°a xÃ¡c Ä‘á»‹nh Ä‘Æ°á»£c vai trÃ² phÃ¹ há»£p rÃµ rÃ ng.")

    # 2
    lines.append("\n2. Äiá»ƒm máº¡nh hiá»‡n táº¡i")
    if strengths:
        for s in strengths[:5]:
            lines.append(f"- {s}")
    else:
        lines.append("- CV hiá»‡n chÆ°a cÃ³ nhiá»u tÃ­n hiá»‡u máº¡nh khá»›p vá»›i role Data/AI má»¥c tiÃªu.")

    # 3
    lines.append("\n3. Äiá»ƒm cÃ²n thiáº¿u" if intent == "cv_analysis" else "\n3. Äiá»ƒm cáº§n bÃ¹ Ä‘áº¯p")
    if missing:
        for m in missing[:5]:
            lines.append(f"- {m}")
    else:
        lines.append("- ChÆ°a phÃ¡t hiá»‡n thiáº¿u há»¥t cá»¥ thá»ƒ ná»•i báº­t.")

    # 4
    lines.append("\n4. Ká»¹ nÄƒng nÃªn phÃ¡t triá»ƒn tiáº¿p" if intent == "cv_analysis" else "\n4. Káº¿ hoáº¡ch há»c ká»¹ nÄƒng")
    if plan:
        for p in plan[:5]:
            lines.append(f"- {p}")
    else:
        lines.append("- NÃªn Æ°u tiÃªn há»c thÃªm ká»¹ nÄƒng ná»n táº£ng theo role má»¥c tiÃªu.")

    # 5
    lines.append(f"\n{section5}")
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
    dedup = []
    for action in suggested_actions:
        if action not in seen:
            seen.add(action)
            dedup.append(action)

    for action in dedup[:5]:
        lines.append(action)

    lines.append("\n[Ghi chÃº] Há»‡ thá»‘ng Ä‘ang dÃ¹ng cháº¿ Ä‘á»™ dá»± phÃ²ng vÃ¬ chÆ°a gá»i Ä‘Æ°á»£c Ollama/Llama 3.")
    return "\n".join(lines)


def fallback_general_answer() -> str:
    return (
        "Hiá»‡n chÆ°a gá»i Ä‘Æ°á»£c mÃ´ hÃ¬nh LLM. "
        "Báº¡n hÃ£y báº­t Ollama rá»“i thá»­ láº¡i Ä‘á»ƒ mÃ¬nh tráº£ lá»i cÃ¢u há»i kiáº¿n thá»©c chung tá»± nhiÃªn hÆ¡n."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True, help="CÃ¢u há»i ngÆ°á»i dÃ¹ng")
    parser.add_argument("--gap_result", default="", help="Optional path tá»›i gap_analysis_result.json")
    args = parser.parse_args()

    intent = classify_question(args.question)
    gap_result = None

    if args.gap_result:
        gap_path = Path(args.gap_result)
        if gap_path.exists():
            gap_result = load_json(str(gap_path))

    if intent == "cv_analysis":
        if not gap_result:
            raise ValueError("CÃ¢u há»i dáº¡ng cv_analysis cáº§n cung cáº¥p --gap_result")
        prompt = build_cv_prompt(gap_result, args.question)

    elif intent == "career_advice":
        if not gap_result:
            raise ValueError("CÃ¢u há»i dáº¡ng career_advice cáº§n cung cáº¥p --gap_result")
        prompt = build_career_prompt(gap_result, args.question)

    else:
        prompt = build_general_prompt(args.question)

    try:
        answer = ask_ollama(prompt)
        if not answer:
            if intent == "general_question":
                answer = fallback_general_answer()
            else:
                answer = fallback_cv_or_career_answer(gap_result, intent)
    except Exception as e:
        print(f"[Warning] KhÃ´ng gá»i Ä‘Æ°á»£c Ollama: {e}")
        if intent == "general_question":
            answer = fallback_general_answer()
        else:
            answer = fallback_cv_or_career_answer(gap_result, intent)

    print("\n===== INTENT =====")
    print(intent)
    print("\n===== CHATBOT ANSWER =====\n")
    print(answer)


if __name__ == "__main__":
    main()

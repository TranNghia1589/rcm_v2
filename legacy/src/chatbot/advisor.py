import argparse
import json
from pathlib import Path

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_non_data_ai_background(gap_result: dict) -> str:
    """
    Suy đoán xem CV hiện tại có đang ngoài domain Data/AI hay không.
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
            return f"Hồ sơ hiện tại của bạn chưa thể hiện rõ định hướng Data/AI và đang nghiêng nhiều hơn về hướng {target_role}."
        return "Hồ sơ hiện tại của bạn chưa thể hiện rõ định hướng Data/AI và có xu hướng thuộc lĩnh vực khác."

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
Bạn là chatbot tư vấn nghề nghiệp và phân tích CV cho nhóm ngành Data/AI.

NGUYÊN TẮC:
- Trả lời bằng tiếng Việt.
- Không bịa thông tin ngoài dữ liệu được cung cấp.
- Không dùng lời khuyên chung chung nếu dữ liệu đã có thông tin cụ thể.
- Phải bám sát vào: best_fit_roles, strengths, missing_skills, development_plan, domain_fit.
- Nếu CV chưa phù hợp với nhóm Data/AI, hãy nói rõ nhưng lịch sự và thực tế.
- Nếu có thể, hãy nêu thứ tự ưu tiên học kỹ năng.

GỢI Ý NGỮ CẢNH NGOÀI DOMAIN:
{domain_hint if domain_hint else "Không có ghi chú đặc biệt."}

BẮT BUỘC trả lời đúng theo 5 mục sau và giữ nguyên thứ tự:

1. Mức độ phù hợp
- Nêu CV hiện tại phù hợp ở mức nào với role mục tiêu.
- Nếu domain_fit là low, phải nói rõ CV hiện chưa thuộc nhóm Data/AI rõ ràng.

2. Điểm mạnh hiện tại
- Chỉ nêu các điểm mạnh thật sự có trong strengths hoặc matched_skills.
- Nếu strengths ít hoặc rỗng, nói ngắn gọn rằng CV chưa có nhiều tín hiệu mạnh cho role mục tiêu.

3. Điểm còn thiếu
- Chỉ nêu các thiếu hụt cụ thể từ missing_skills.
- Không biến ngành/lĩnh vực thành kỹ năng.

4. Kỹ năng nên phát triển tiếp
- Ưu tiên theo development_plan và recommended_next_skills.
- Nêu theo thứ tự học hợp lý nếu có thể.

5. Hành động đề xuất trong 1–3 tháng
- Đưa ra hành động cụ thể, thực tế, ngắn gọn.
- Ví dụ: học công cụ, làm project, bổ sung CV, thực tập, portfolio.

DỮ LIỆU PHÂN TÍCH:
{json.dumps(structured_context, ensure_ascii=False, indent=2)}

CÂU HỎI NGƯỜI DÙNG:
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

    # 1. Mức độ phù hợp
    lines.append("1. Mức độ phù hợp")
    if roles:
        if domain_fit == "low":
            if domain_hint:
                lines.append(domain_hint)
            lines.append(f"Role gần nhất hiện tại trong nhóm Data/AI là {roles[0]}, nhưng mức độ phù hợp còn thấp.")
        elif domain_fit == "medium":
            lines.append(f"CV của bạn hiện phù hợp ở mức trung bình với vị trí {roles[0]}.")
        else:
            lines.append(f"CV của bạn hiện phù hợp khá tốt với vị trí {roles[0]}.")
    else:
        lines.append("Chưa xác định được role phù hợp rõ ràng.")

    # 2. Điểm mạnh hiện tại
    lines.append("\n2. Điểm mạnh hiện tại")
    if strengths:
        for s in strengths[:5]:
            lines.append(f"- {s}")
    else:
        lines.append("- CV hiện chưa có nhiều tín hiệu mạnh khớp với role Data/AI mục tiêu.")

    # 3. Điểm còn thiếu
    lines.append("\n3. Điểm còn thiếu")
    if missing:
        for m in missing[:5]:
            lines.append(f"- {m}")
    else:
        lines.append("- Chưa phát hiện thiếu hụt cụ thể nổi bật.")

    # 4. Kỹ năng nên phát triển tiếp
    lines.append("\n4. Kỹ năng nên phát triển tiếp")
    if plan:
        for p in plan[:5]:
            lines.append(f"- {p}")
    else:
        lines.append("- Nên ưu tiên học thêm kỹ năng nền tảng theo role mục tiêu.")

    # 5. Hành động đề xuất trong 1-3 tháng
    lines.append("\n5. Hành động đề xuất trong 1–3 tháng")
    suggested_actions = []

    if missing:
        if any("excel" in m.lower() for m in missing):
            suggested_actions.append("- Ôn hoặc học Excel nâng cao và thực hành với dữ liệu thực tế.")
        if any("sql" in m.lower() for m in missing):
            suggested_actions.append("- Học SQL cơ bản đến trung cấp và luyện truy vấn trên dataset thật.")
        if any("power bi" in m.lower() or "tableau" in m.lower() for m in missing):
            suggested_actions.append("- Làm ít nhất 1 project dashboard bằng Power BI hoặc Tableau.")
        if any("statistics" in m.lower() for m in missing):
            suggested_actions.append("- Ôn lại xác suất thống kê và các chỉ số mô tả cơ bản.")

    suggested_actions.append("- Bổ sung 1–2 project dữ liệu vào CV nếu hiện chưa có project liên quan.")
    suggested_actions.append("- Viết lại CV theo hướng nhấn mạnh kỹ năng và dự án liên quan đến role mục tiêu.")

    seen = set()
    dedup_actions = []
    for action in suggested_actions:
        if action not in seen:
            seen.add(action)
            dedup_actions.append(action)

    for action in dedup_actions[:5]:
        lines.append(action)

    lines.append("\n[Ghi chú] Hệ thống đang dùng chế độ dự phòng vì chưa gọi được Ollama/Llama 3.")
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
        lines.append("1. Mức độ phù hợp")
    else:
        lines.append("1. Mục tiêu phù hợp hiện tại")

    if roles:
        if domain_fit == "low":
            lines.append(
                f"CV hiện tại chưa phù hợp mạnh với nhóm Data/AI. "
                f"Role gần nhất hiện tại là {roles[0]} với mức phù hợp thấp (score: {score})."
            )
        elif domain_fit == "medium":
            lines.append(
                f"CV hiện tại phù hợp ở mức trung bình với vị trí {roles[0]} "
                f"(score: {score}). Bạn đã có một phần nền tảng nhưng vẫn còn thiếu các kỹ năng quan trọng."
            )
        else:
            lines.append(
                f"CV hiện tại phù hợp khá tốt với vị trí {roles[0]} "
                f"(score: {score})."
            )
    else:
        lines.append("Chưa xác định được role phù hợp rõ ràng.")

    lines.append("\n2. Điểm mạnh hiện tại")
    if strengths:
        for s in strengths[:5]:
            lines.append(f"- {s}")
    else:
        lines.append("- CV hiện chưa có nhiều tín hiệu mạnh khớp với role mục tiêu.")

    lines.append("\n3. Điểm còn thiếu" if intent == "cv_analysis" else "\n3. Điểm cần bù đắp")
    if missing:
        for m in missing[:5]:
            lines.append(f"- {m}")
    else:
        lines.append("- Chưa phát hiện thiếu hụt nổi bật.")

    lines.append("\n4. Kỹ năng nên phát triển tiếp" if intent == "cv_analysis" else "\n4. Kế hoạch học kỹ năng")
    if plan:
        for p in plan[:5]:
            lines.append(f"- {p}")
    else:
        lines.append("- Nên ưu tiên học thêm kỹ năng nền tảng theo role mục tiêu.")

    lines.append("\n5. Hành động đề xuất trong 1–3 tháng")

    actions = []
    if "SQL" in missing:
        actions.append("- Học SQL cơ bản đến trung cấp và luyện truy vấn với dataset thật.")
    if "Excel" in missing:
        actions.append("- Ôn hoặc học Excel nâng cao, đặc biệt là xử lý và tổng hợp dữ liệu.")
    if "Power BI" in missing or "Tableau" in missing:
        actions.append("- Làm ít nhất 1 project dashboard bằng Power BI hoặc Tableau.")
    if "Statistics" in missing:
        actions.append("- Ôn lại xác suất thống kê và các chỉ số mô tả cơ bản.")

    actions.append("- Bổ sung 1–2 project dữ liệu vào CV nếu hiện chưa có project liên quan.")
    actions.append("- Viết lại CV theo hướng nhấn mạnh kỹ năng và dự án liên quan đến role mục tiêu.")

    seen = set()
    for action in actions:
        if action not in seen:
            seen.add(action)
            lines.append(action)

    lines.append("\n[Ghi chú] Hệ thống đang dùng chế độ dự phòng vì chưa gọi được Ollama/Llama 3.")
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap_result", required=True, help="Path to gap analysis result JSON")
    parser.add_argument("--question", required=True, help="User question for the chatbot")
    args = parser.parse_args()

    gap_result_path = Path(args.gap_result)
    if not gap_result_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file gap result: {gap_result_path}")

    gap_result = load_json(str(gap_result_path))
    prompt = build_prompt(gap_result, args.question)

    try:
        answer = ask_ollama(prompt)
        if not answer:
            answer = fallback_response(gap_result, args.question)
    except Exception as e:
        print(f"[Warning] Không gọi được Ollama: {e}")
        answer = fallback_response(gap_result, args.question)

    print("\n===== CHATBOT ANSWER =====\n")
    print(answer)


if __name__ == "__main__":
    main()
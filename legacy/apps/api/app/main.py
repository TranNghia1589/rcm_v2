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
MODEL_NAME = "llama3:latest"


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
        "cv", "resume", "hồ sơ", "thiếu gì", "thiếu kỹ năng",
        "phù hợp nghề", "hợp nghề nào", "điểm mạnh", "điểm yếu",
        "dựa trên cv", "dựa trên hồ sơ", "ứng tuyển", "hợp với data analyst không"
    ]

    career_keywords = [
        "nên học gì", "roadmap", "lộ trình", "nên phát triển gì",
        "nên làm project gì", "phát triển kỹ năng", "3 tháng", "6 tháng",
        "để trở thành", "để theo", "định hướng nghề nghiệp", "nên học trước"
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
                f"Hồ sơ hiện tại của người dùng chưa thể hiện rõ định hướng Data/AI "
                f"và đang nghiêng nhiều hơn về hướng {target_role}."
            )
        return (
            "Hồ sơ hiện tại của người dùng chưa thể hiện rõ định hướng Data/AI "
            "và có xu hướng thuộc lĩnh vực khác."
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
Bạn là chatbot tư vấn nghề nghiệp và phân tích CV cho nhóm ngành Data/AI.

NGUYÊN TẮC:
- Trả lời bằng tiếng Việt.
- Chỉ sử dụng thông tin có trong dữ liệu phân tích.
- Không bịa thêm kỹ năng, kinh nghiệm hoặc ngành nghề ngoài dữ liệu.
- Không lặp ý giữa các mục.
- Nếu CV chưa phù hợp với Data/AI, hãy nói rõ nhưng lịch sự.
- Ưu tiên nêu kỹ năng cụ thể thay vì nói chung chung.

BẮT BUỘC trả lời theo đúng 5 mục dưới đây:

1. Mức độ phù hợp
- Nêu ngắn gọn CV hiện tại phù hợp ở mức nào với role mục tiêu.
- Nếu domain_fit là low, phải nói rõ CV chưa có nhiều tín hiệu Data/AI.

2. Điểm mạnh hiện tại
- Chỉ nêu các điểm mạnh thật sự có trong strengths hoặc matched_skills.
- Nếu strengths ít hoặc rỗng, nói ngắn gọn rằng CV chưa có nhiều tín hiệu mạnh cho role mục tiêu.

3. Điểm còn thiếu
- Chỉ nêu các kỹ năng còn thiếu từ missing_skills.
- Không biến lĩnh vực hoặc ngành thành kỹ năng.

4. Kỹ năng nên phát triển tiếp
- Ưu tiên theo development_plan.
- Nếu có thể, sắp xếp theo thứ tự học hợp lý.

5. Hành động đề xuất trong 1–3 tháng
- Đưa ra hành động cụ thể, thực tế, ngắn gọn.
- Ví dụ: học công cụ, làm project, cập nhật CV, thực tập.

GHI CHÚ NGOÀI DOMAIN:
{domain_hint if domain_hint else "Không có ghi chú đặc biệt."}

DỮ LIỆU PHÂN TÍCH:
{json.dumps(structured_context, ensure_ascii=False, indent=2)}

CÂU HỎI NGƯỜI DÙNG:
{user_question}
""".strip()


def build_career_prompt(gap_result: dict, user_question: str) -> str:
    structured_context = build_structured_context(gap_result)
    domain_hint = detect_non_data_ai_background(gap_result)

    return f"""
Bạn là chatbot tư vấn định hướng nghề nghiệp trong lĩnh vực Data/AI.

NGUYÊN TẮC:
- Trả lời bằng tiếng Việt.
- Không bịa thông tin ngoài dữ liệu.
- Không lặp ý.
- Phải cụ thể, thực tế, có thứ tự ưu tiên.
- Tập trung vào kỹ năng cần học, project nên làm và hướng đi phù hợp hơn nếu có.

BẮT BUỘC trả lời theo đúng 5 mục:
1. Mục tiêu phù hợp hiện tại
2. Điểm mạnh hiện tại
3. Điểm cần bù đắp
4. Kế hoạch học kỹ năng
5. Hành động cụ thể trong 1–3 tháng

GHI CHÚ NGOÀI DOMAIN:
{domain_hint if domain_hint else "Không có ghi chú đặc biệt."}

DỮ LIỆU PHÂN TÍCH:
{json.dumps(structured_context, ensure_ascii=False, indent=2)}

CÂU HỎI:
{user_question}
""".strip()


def build_general_prompt(user_question: str) -> str:
    return f"""
Bạn là trợ lý tư vấn nghề nghiệp trong lĩnh vực Data/AI.

YÊU CẦU:
- Trả lời bằng tiếng Việt.
- Dễ hiểu, chính xác, súc tích.
- Nếu là câu hỏi nền tảng, hãy giải thích như cho người mới học.
- Nếu có thể, đưa ví dụ ngắn.

CÂU HỎI:
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
        section5 = "5. Hành động đề xuất trong 1–3 tháng"
    else:
        section5 = "5. Hành động cụ thể trong 1–3 tháng"

    # 1
    lines.append("1. Mức độ phù hợp" if intent == "cv_analysis" else "1. Mục tiêu phù hợp hiện tại")
    if roles:
        if domain_fit == "low":
            if domain_hint:
                lines.append(domain_hint)
            lines.append(f"Vai trò gần nhất hiện tại trong nhóm Data/AI là {roles[0]}, nhưng mức độ phù hợp còn thấp.")
        elif domain_fit == "medium":
            lines.append(f"CV hiện tại phù hợp ở mức trung bình với vị trí {roles[0]}.")
        else:
            lines.append(f"CV hiện tại phù hợp khá tốt với vị trí {roles[0]}.")
    else:
        lines.append("Chưa xác định được vai trò phù hợp rõ ràng.")

    # 2
    lines.append("\n2. Điểm mạnh hiện tại")
    if strengths:
        for s in strengths[:5]:
            lines.append(f"- {s}")
    else:
        lines.append("- CV hiện chưa có nhiều tín hiệu mạnh khớp với role Data/AI mục tiêu.")

    # 3
    lines.append("\n3. Điểm còn thiếu" if intent == "cv_analysis" else "\n3. Điểm cần bù đắp")
    if missing:
        for m in missing[:5]:
            lines.append(f"- {m}")
    else:
        lines.append("- Chưa phát hiện thiếu hụt cụ thể nổi bật.")

    # 4
    lines.append("\n4. Kỹ năng nên phát triển tiếp" if intent == "cv_analysis" else "\n4. Kế hoạch học kỹ năng")
    if plan:
        for p in plan[:5]:
            lines.append(f"- {p}")
    else:
        lines.append("- Nên ưu tiên học thêm kỹ năng nền tảng theo role mục tiêu.")

    # 5
    lines.append(f"\n{section5}")
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
    dedup = []
    for action in suggested_actions:
        if action not in seen:
            seen.add(action)
            dedup.append(action)

    for action in dedup[:5]:
        lines.append(action)

    lines.append("\n[Ghi chú] Hệ thống đang dùng chế độ dự phòng vì chưa gọi được Ollama/Llama 3.")
    return "\n".join(lines)


def fallback_general_answer() -> str:
    return (
        "Hiện chưa gọi được mô hình LLM. "
        "Bạn hãy bật Ollama rồi thử lại để mình trả lời câu hỏi kiến thức chung tự nhiên hơn."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True, help="Câu hỏi người dùng")
    parser.add_argument("--gap_result", default="", help="Optional path tới gap_analysis_result.json")
    args = parser.parse_args()

    intent = classify_question(args.question)
    gap_result = None

    if args.gap_result:
        gap_path = Path(args.gap_result)
        if gap_path.exists():
            gap_result = load_json(str(gap_path))

    if intent == "cv_analysis":
        if not gap_result:
            raise ValueError("Câu hỏi dạng cv_analysis cần cung cấp --gap_result")
        prompt = build_cv_prompt(gap_result, args.question)

    elif intent == "career_advice":
        if not gap_result:
            raise ValueError("Câu hỏi dạng career_advice cần cung cấp --gap_result")
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
        print(f"[Warning] Không gọi được Ollama: {e}")
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
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PromptingConfig:
    max_context_chunks: int = 4
    max_history_messages: int = 6
    language: str = "vi"
    style: str = "clear_and_grounded"
    ollama_timeout_sec: int = 180
    ollama_temperature: float = 0.25
    ollama_retries: int = 1
    ollama_url: str = "http://127.0.0.1:11434/api/generate"
    ollama_model: str = "llama3.1:8b"
    ollama_keep_alive: str = "10m"
    ollama_retry_backoff_sec: float = 0.35
    strict_no_fallback: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PromptingConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(
            max_context_chunks=int(raw.get("max_context_chunks", 4)),
            max_history_messages=int(raw.get("max_history_messages", 6)),
            language=str(raw.get("language", "vi")),
            style=str(raw.get("style", "clear_and_grounded")),
            ollama_timeout_sec=int(raw.get("ollama_timeout_sec", 180)),
            ollama_temperature=float(raw.get("ollama_temperature", 0.25)),
            ollama_retries=int(raw.get("ollama_retries", 1)),
            ollama_url=str(raw.get("ollama_url", "http://127.0.0.1:11434/api/generate")),
            ollama_model=str(raw.get("ollama_model", "llama3.1:8b")),
            ollama_keep_alive=str(raw.get("ollama_keep_alive", "10m")),
            ollama_retry_backoff_sec=float(raw.get("ollama_retry_backoff_sec", 0.35)),
            strict_no_fallback=bool(raw.get("strict_no_fallback", False)),
        )


def classify_intent(question: str) -> str:
    q = (question or "").lower()
    if any(
        k in q
        for k in [
            "hr",
            "recruiter",
            "tuyển dụng",
            "screen cv",
            "lọc cv",
            "jd",
            "job description",
            "phỏng vấn",
            "ứng viên",
            "talent acquisition",
            "headcount",
        ]
    ):
        return "hr_recruiter"
    if any(k in q for k in ["job", "việc", "vị trí", "ứng tuyển", "phù hợp job", "gợi ý job"]):
        return "job_recommendation"
    if any(
        k in q
        for k in [
            "chuyển",
            "trái ngành",
            "chuyển lĩnh vực",
            "career switch",
            "roadmap",
            "lộ trình",
            "nên học gì",
            "bắt đầu học",
            "1-3 tháng",
            "30-60-90",
        ]
    ):
        return "career_transition"
    if any(
        k in q
        for k in [
            "cv",
            "resume",
            "hồ sơ",
            "kỹ năng",
            "bo sung",
            "bổ sung",
            "cải thiện",
            "improve cv",
            "thiếu gì",
            "fit role",
            "phù hợp role",
            "gap",
        ]
    ):
        return "cv_improvement"
    if any(
        k in q
        for k in [
            "ngành",
            "thị trường",
            "career path",
            "xu hướng",
            "role nào",
            "data analyst",
            "data engineer",
            "data scientist",
            "business analyst",
            "mức lương",
            "job market",
        ]
    ):
        return "industry_insight"
    return "general"


def build_grounded_prompt(
    *,
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    config: PromptingConfig,
    intent: str = "general",
    conversation_history: list[dict[str, str]] | None = None,
    internal_cv_context: str = "",
) -> str:
    top_chunks = retrieved_chunks[: config.max_context_chunks]
    context_lines: list[str] = []
    for c in top_chunks:
        context_lines.append(
            f"[chunk_id={c.get('chunk_id')}] title={c.get('title', '')}\n{c.get('content', '')}"
        )
    context_text = "\n\n".join(context_lines) if context_lines else "Khong co context."
    history_lines: list[str] = []
    for item in conversation_history or []:
        role = str(item.get("role", "")).strip() or "unknown"
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        history_lines.append(f"{role.upper()}: {content}")
    history_text = "\n".join(history_lines) if history_lines else "Khong co lich su hoi thoai."
    citation_rule = (
        "Khi nêu thông tin rút trực tiếp từ RETRIEVED_CONTEXT, thêm trích dẫn dạng [chunk_id=...]. "
        "Không được bịa chunk_id."
        if top_chunks
        else "Hiện không có RETRIEVED_CONTEXT, tuyệt đối không chèn [chunk_id=...]."
    )

    intent_guide = ""
    if intent == "job_recommendation":
        intent_guide = (
            "Mục tiêu câu trả lời:\n"
            "1) Đưa ra tối đa 5 job gợi ý phù hợp từ CV context, history và RETRIEVED_CONTEXT nếu có.\n"
            "2) Mỗi job nêu lý do phù hợp ngắn gọn.\n"
            "3) Nêu kỹ năng cần bổ sung để tăng khả năng trúng tuyển.\n"
            "4) Nếu INTERNAL_CV_CONTEXT có domain fit low/medium/high thì phải giữ đúng mức đó, không tự nâng mức fit."
        )
    elif intent == "hr_recruiter":
        intent_guide = (
            "Mục tiêu câu trả lời:\n"
            "1) Phân tầng ứng viên theo mức fit: mạnh, trung bình, rủi ro hoặc cần xác minh thêm.\n"
            "2) Nêu tín hiệu tốt, tín hiệu thiếu, project liên quan và câu hỏi screening/interview nên hỏi.\n"
            "3) Nếu thiếu dữ liệu thì chỉ rõ mục nào HR cần xác minh thêm.\n"
            "4) Không nói 'fit cao' nếu INTERNAL_CV_CONTEXT thể hiện domain fit chỉ ở mức medium hoặc low.\n"
            "5) Ưu tiên cấu trúc: Mức fit, Lý do, Điểm cần xác minh, Câu hỏi screening."
        )
    elif intent in {"cv_improvement", "career_transition"}:
        intent_guide = (
            "Mục tiêu câu trả lời:\n"
            "1) Xác định mức fit hiện tại theo CV/gap context trước.\n"
            "2) Liệt kê kỹ năng, kinh nghiệm, project và tín hiệu education liên quan theo thứ tự ưu tiên.\n"
            "3) Nếu liên quan chuyển lĩnh vực, nêu lộ trình 30-60-90 ngày ngắn gọn và thực tế.\n"
            "4) Nếu INTERNAL_CV_CONTEXT có domain fit low/medium/high thì phải nói đúng mức đó."
        )
    elif intent == "industry_insight":
        intent_guide = (
            "Mục tiêu câu trả lời:\n"
            "1) Giải thích bối cảnh ngành hoặc khác biệt giữa các role dễ hiểu.\n"
            "2) Nêu tín hiệu tuyển dụng/đòi hỏi kỹ năng phổ biến nếu có RETRIEVED_CONTEXT.\n"
            "3) Nếu không có dữ liệu nội bộ, vẫn được phép đưa guidance nghề nghiệp phổ quát nhưng phải nói đó là gợi ý chung."
        )
    else:
        intent_guide = (
            "Trả lời đúng trọng tâm câu hỏi. "
            "Ưu tiên CV context, lịch sử hội thoại và RETRIEVED_CONTEXT; nếu thiếu thì vẫn có thể đưa gợi ý nghề nghiệp phổ quát."
        )

    return f"""
Bạn là trợ lý tư vấn nghề nghiệp cho nhóm Data, AI, Product, Business và các case HR tuyển dụng.
BẮT BUỘC trả lời chủ yếu bằng tiếng Việt tự nhiên.
Được phép giữ các thuật ngữ kỹ thuật như SQL, Python, Power BI, Machine Learning.
Không dùng tiếng Anh cho câu văn giải thích thông thường.
ƯU TIÊN sử dụng dữ liệu trong INTERNAL_CV_CONTEXT, CONVERSATION_HISTORY và RETRIEVED_CONTEXT.
Nếu RETRIEVED_CONTEXT trống, bạn vẫn được phép trả lời bằng hiểu biết nghề nghiệp/HR phổ quát, nhưng phải nói rõ đó là gợi ý chung chứ không phải dữ liệu nội bộ.
Không suy đoán quá mức từ dữ liệu ít.
Nếu người dùng hỏi follow-up kiểu "vậy", "còn thiếu gì", "thế HR nhìn thế nào", hãy suy luận chủ thể từ CONVERSATION_HISTORY trước khi trả lời.
Nếu có INTERNAL_CV_CONTEXT, hãy xem đó là hồ sơ đang được bàn tới.
Nếu INTERNAL_CV_CONTEXT ghi rõ domain fit hoặc mức phù hợp, phải bám đúng mức đó.
Ưu tiên dùng các trường CV đã xử lý như skills, projects, education signals, development plan, role ranking khi chúng có mặt trong INTERNAL_CV_CONTEXT.
Khi nhắc tới điểm mạnh hoặc điểm yếu, cố gắng nêu rõ chúng đến từ project nào, skill nào, hay gap signal nào trong CV đã xử lý.
{citation_rule}
{intent_guide}

INTERNAL_CV_CONTEXT:
{internal_cv_context or "Khong co CV context."}

CONVERSATION_HISTORY:
{history_text}

RETRIEVED_CONTEXT:
{context_text}

CÂU HỎI:
{question}
""".strip()


def build_vietnamese_rewrite_prompt(answer: str, question: str) -> str:
    return f"""
Hãy viết lại câu trả lời sau thành tiếng Việt tự nhiên, rõ ràng, đúng trọng tâm.
Giữ nguyên ý nghĩa, không bịa thêm dữ liệu mới.
Cho phép giữ các từ chuyên ngành tiếng Anh nếu cần.

Câu hỏi:
{question}

Câu trả lời cần viết lại:
{answer}
""".strip()


def is_english_heavy(text: str) -> bool:
    if not text or len(text.strip()) < 10:
        return False
    low = text.lower()
    markers = [
        "based on",
        "i recommend",
        "please note",
        "opportunities",
        "provided context",
        "assessment",
    ]
    if any(m in low for m in markers):
        return True

    latin_words = re.findall(r"\b[a-z]{3,}\b", low)
    vi_chars = re.findall(r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụýỳỷỹỵ]", low)
    if latin_words and not vi_chars and len(latin_words) >= 12:
        return True
    return False


def build_fallback_answer(question: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    if not retrieved_chunks:
        return (
            "Hiện chưa có ngữ cảnh phù hợp trong kho dữ liệu để trả lời câu hỏi này. "
            "Bạn thử diễn đạt cụ thể hơn về role, kỹ năng hoặc vị trí ứng tuyển."
        )

    top = retrieved_chunks[:3]
    lines = [f"Tóm tắt theo dữ liệu truy xuất cho câu hỏi: {question}"]
    for idx, c in enumerate(top, 1):
        content = str(c.get("content", "")).replace("\n", " ").strip()
        short = content[:180] + ("..." if len(content) > 180 else "")
        lines.append(f"{idx}. {short} [chunk_id={c.get('chunk_id')}]")
    return "\n".join(lines)

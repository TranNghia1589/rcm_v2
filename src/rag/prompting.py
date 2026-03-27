from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PromptingConfig:
    max_context_chunks: int = 4
    language: str = "vi"
    style: str = "clear_and_grounded"
    ollama_timeout_sec: int = 180
    ollama_temperature: float = 0.25
    ollama_retries: int = 1
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "llama3:latest"
    ollama_keep_alive: str = "10m"
    ollama_retry_backoff_sec: float = 0.35
    strict_no_fallback: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PromptingConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(
            max_context_chunks=int(raw.get("max_context_chunks", 4)),
            language=str(raw.get("language", "vi")),
            style=str(raw.get("style", "clear_and_grounded")),
            ollama_timeout_sec=int(raw.get("ollama_timeout_sec", 180)),
            ollama_temperature=float(raw.get("ollama_temperature", 0.25)),
            ollama_retries=int(raw.get("ollama_retries", 1)),
            ollama_url=str(raw.get("ollama_url", "http://localhost:11434/api/generate")),
            ollama_model=str(raw.get("ollama_model", "llama3:latest")),
            ollama_keep_alive=str(raw.get("ollama_keep_alive", "10m")),
            ollama_retry_backoff_sec=float(raw.get("ollama_retry_backoff_sec", 0.35)),
            strict_no_fallback=bool(raw.get("strict_no_fallback", False)),
        )


def classify_intent(question: str) -> str:
    q = (question or "").lower()
    if any(k in q for k in ["job", "việc", "vị trí", "ứng tuyển", "phù hợp job", "gợi ý job"]):
        return "job_recommendation"
    if any(k in q for k in ["chuyển", "trái ngành", "chuyển lĩnh vực", "career switch"]):
        return "career_transition"
    if any(k in q for k in ["cv", "kỹ năng", "bo sung", "bổ sung", "cải thiện", "improve cv"]):
        return "cv_improvement"
    return "general"


def build_grounded_prompt(
    *,
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    config: PromptingConfig,
    intent: str = "general",
) -> str:
    top_chunks = retrieved_chunks[: config.max_context_chunks]
    context_lines: list[str] = []
    for c in top_chunks:
        context_lines.append(
            f"[chunk_id={c.get('chunk_id')}] title={c.get('title', '')}\n{c.get('content', '')}"
        )
    context_text = "\n\n".join(context_lines) if context_lines else "Khong co context."

    intent_guide = ""
    if intent == "job_recommendation":
        intent_guide = (
            "Mục tiêu câu trả lời:\n"
            "1) Đưa ra tối đa 5 job gợi ý từ CONTEXT.\n"
            "2) Mỗi job nêu lý do phù hợp ngắn gọn.\n"
            "3) Nêu kỹ năng cần bổ sung để tăng khả năng trúng tuyển."
        )
    elif intent in {"cv_improvement", "career_transition"}:
        intent_guide = (
            "Mục tiêu câu trả lời:\n"
            "1) Liệt kê kỹ năng/kinh nghiệm thị trường đang yêu cầu trong CONTEXT.\n"
            "2) Gợi ý cải thiện CV theo thứ tự ưu tiên.\n"
            "3) Nếu liên quan chuyển lĩnh vực, nêu lộ trình 30-60-90 ngày."
        )
    else:
        intent_guide = "Trả lời đúng trọng tâm câu hỏi và dựa hoàn toàn vào CONTEXT."

    return f"""
Bạn là trợ lý tư vấn nghề nghiệp.
BẮT BUỘC trả lời chủ yếu bằng tiếng Việt tự nhiên.
Được phép giữ các thuật ngữ kỹ thuật như SQL, Python, Power BI, Machine Learning.
Không dùng tiếng Anh cho câu văn giải thích thông thường.
Chỉ sử dụng thông tin trong CONTEXT.
Không được dùng kiến thức bên ngoài CONTEXT.
Không suy đoán quá mức từ dữ liệu ít.
Nếu thiếu dữ liệu thì nói rõ: "Không đủ dữ liệu nội bộ để kết luận".
Khi nêu thông tin quan trọng, thêm trích dẫn dạng [chunk_id=...].
{intent_guide}

CONTEXT:
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

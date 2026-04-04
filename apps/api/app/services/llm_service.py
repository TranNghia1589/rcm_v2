"""
LLM Service for generating contextual improvement suggestions using Ollama.
"""
from __future__ import annotations

import os
import re
from typing import Any

import requests


class OllamaLLMService:
    """Service to interact with Ollama for LLM-based suggestions."""

    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "qwen"):
        """
        Initialize Ollama LLM service.

        Args:
            ollama_url: Base URL of Ollama server
            model: Model name (e.g., "qwen", "mistral", "neural-chat")
        """
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.api_endpoint = f"{self.ollama_url}/api/generate"

    def classify_user_intent(self, question: str) -> str:
        """
        Classify user intent for guest analysis routing.

        Returns one of: score | recommend | improve_cv | general
        """
        prompt = self._build_intent_prompt(question)
        try:
            raw = self._call_ollama(prompt, timeout=20, temperature=0.0)
            normalized = (raw or "").strip().lower()
            m = re.search(r"\b(score|recommend|improve_cv|general)\b", normalized)
            if m:
                return m.group(1)
            return "general"
        except Exception:
            return "general"

    def generate_improvement_suggestions(
        self,
        cv_context: str,
        target_domain: str,
        target_role: str,
        question: str,
    ) -> list[dict[str, str]]:
        """
        Generate improvement suggestions using LLM.

        Args:
            cv_context: Current CV skills and background
            target_domain: Target domain (DATA_AI, WEB, MOBILE, etc.)
            target_role: Target role/position
            question: User's question/intent

        Returns:
            List of skill suggestions with reasons
        """
        prompt = self._build_prompt(cv_context, target_domain, target_role, question)

        try:
            response = self._call_ollama(prompt)
            suggestions = self._parse_response(response)
            return suggestions
        except Exception as e:
            print(f"[LLM Error] {str(e)}")
            return []

    def _build_intent_prompt(self, question: str) -> str:
        return f"""Bạn là bộ phân loại ý định cho chatbot tư vấn CV.

Nhiệm vụ: Phân loại CÂU HỎI vào đúng 1 nhãn sau:
- score: người dùng muốn chấm điểm/đánh giá CV
- recommend: người dùng muốn gợi ý việc làm phù hợp
- improve_cv: người dùng muốn cải thiện CV/kỹ năng/lộ trình bổ sung
- general: trò chuyện tự nhiên, chưa rơi vào 3 nhóm trên

Chỉ trả về đúng 1 từ duy nhất trong 4 từ: score | recommend | improve_cv | general
Không giải thích.

Câu hỏi: {question}
""".strip()

    def _build_prompt(
        self,
        cv_context: str,
        target_domain: str,
        target_role: str,
        question: str,
    ) -> str:
        return f"""Bạn là một chuyên gia tư vấn nghề nghiệp giỏi. Phân tích CV và đưa ra gợi ý cải thiện.

THÔNG TIN CV HIỆN TẠI:
{cv_context}

MỤC TIÊU:
- Domain: {target_domain}
- Target Role: {target_role}
- User Question: {question}

NHIỆM VỤ: Đưa ra TOP 5 kỹ năng MỨC ƯU TIÊN mà người này cần học để đạt được mục tiêu.

Định dạng trả lời như sau (CHÍNH XÁC):
Skill 1: [skill_name]
Why: [lý do ngắn gọn tại sao cần kỹ năng này]

Skill 2: [skill_name]
Why: [lý do ngắn gọn]

(tiếp tục với skill 3, 4, 5)

QUAN TRỌNG:
- Chỉ liệt kê skills cụ thể, không dùng cụm chung chung
- Lý do phải liên quan đến domain {target_domain} và role {target_role}
- Ưu tiên skills mà user chưa có"""

    def _call_ollama(self, prompt: str, timeout: int = 60, temperature: float = 0.7) -> str:
        """Call Ollama API to generate response."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": float(temperature),
        }

        try:
            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Cannot connect to Ollama at {self.ollama_url}")
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Ollama request timeout after {timeout}s")
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {str(e)}")

    def _parse_response(self, response: str) -> list[dict[str, str]]:
        """Parse LLM response into structured suggestions."""
        suggestions = []
        lines = response.split("\n")
        current_skill = None
        current_why = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.lower().startswith("skill"):
                if current_skill and current_why:
                    suggestions.append({
                        "skill": current_skill,
                        "why": current_why,
                    })

                if ":" in line:
                    current_skill = line.split(":", 1)[1].strip()
                    current_why = None

            elif line.lower().startswith("why"):
                if ":" in line:
                    current_why = line.split(":", 1)[1].strip()

        if current_skill and current_why:
            suggestions.append({
                "skill": current_skill,
                "why": current_why,
            })

        return suggestions[:5]


def get_ollama_service(
    ollama_url: str | None = None,
    model: str | None = None,
) -> OllamaLLMService:
    """Get Ollama LLM service instance."""
    url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
    model_name = model or os.getenv("OLLAMA_MODEL", "qwen")
    return OllamaLLMService(ollama_url=url, model=model_name)

from __future__ import annotations

import json
from typing import Any

from src.rag.generate import OllamaGenerator


def build_explanation_prompt(question: str, items: list[dict[str, Any]]) -> str:
    return f"""
Ban la tro ly tu van nghe nghiep.
Giai thich ngan gon vi sao cac job sau phu hop voi ung vien dua tren score vector + graph.
Tra loi bang tieng Viet tu nhien, ro rang, khong biet them du lieu.

Cau hoi:
{question}

Danh sach job:
{json.dumps(items, ensure_ascii=False, indent=2)}
""".strip()


def generate_explanations(
    question: str,
    items: list[dict[str, Any]],
    *,
    generator: OllamaGenerator | None = None,
    timeout_sec: int = 90,
    temperature: float = 0.2,
    retries: int = 1,
) -> str:
    if not items:
        return "Khong co du du lieu de giai thich."

    prompt = build_explanation_prompt(question, items)
    active_generator = generator or OllamaGenerator()
    try:
        return active_generator.generate(
            prompt,
            timeout_sec=timeout_sec,
            temperature=temperature,
            retries=retries,
        )
    except Exception:
        lines = ["Giai thich du phong:"]
        for idx, item in enumerate(items[:5], 1):
            lines.append(
                f"{idx}. {item.get('title','')} phu hop vi graph_score={item.get('graph_score',0):.3f}, "
                f"vector_score={item.get('vector_score',0):.3f}, coverage={item.get('coverage',0):.3f}."
            )
        return "\n".join(lines)

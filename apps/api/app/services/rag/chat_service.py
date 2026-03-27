from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import re

from apps.api.app.services.rag.retrieval_service import RetrievalService
from src.rag.generate import OllamaConfig, OllamaGenerator
from src.rag.prompting import (
    PromptingConfig,
    build_fallback_answer,
    build_grounded_prompt,
    build_vietnamese_rewrite_prompt,
    classify_intent,
    is_english_heavy,
)


@dataclass
class ChatService:
    postgres_config_path: str | Path
    retrieval_config_path: str | Path
    prompting_config_path: str | Path
    embedding_config_path: str | Path

    def __post_init__(self) -> None:
        self._prompt_cfg = PromptingConfig.from_yaml(self.prompting_config_path)
        self._retrieval = RetrievalService(
            postgres_config_path=self.postgres_config_path,
            retrieval_config_path=self.retrieval_config_path,
            embedding_config_path=self.embedding_config_path,
        )
        self._generator = OllamaGenerator(
            OllamaConfig(
                url=self._prompt_cfg.ollama_url,
                model=self._prompt_cfg.ollama_model,
                timeout_sec=self._prompt_cfg.ollama_timeout_sec,
                temperature=self._prompt_cfg.ollama_temperature,
                retries=self._prompt_cfg.ollama_retries,
                retry_backoff_sec=self._prompt_cfg.ollama_retry_backoff_sec,
                keep_alive=self._prompt_cfg.ollama_keep_alive,
            )
        )

    def _rerank_chunks(self, question: str, chunks: list[dict], intent: str) -> list[dict]:
        q_tokens = set(re.findall(r"[a-zA-ZÀ-ỹ0-9]+", (question or "").lower()))
        intent_boost = {
            "job_recommendation": {"job", "việc", "vị", "tuyển", "position", "role"},
            "cv_improvement": {"cv", "kỹ", "năng", "skills", "improve"},
            "career_transition": {"chuyển", "trái", "ngành", "lộ", "trình", "roadmap"},
            "general": set(),
        }
        boosts = intent_boost.get(intent, set())

        def score(c: dict) -> float:
            text = f"{c.get('title', '')} {c.get('content', '')}".lower()
            t_tokens = set(re.findall(r"[a-zA-ZÀ-ỹ0-9]+", text))
            overlap = len(q_tokens & t_tokens)
            intent_overlap = len(boosts & t_tokens)
            dist = float(c.get("distance", 1.0))
            return overlap * 1.5 + intent_overlap * 2.0 - dist * 2.0

        ranked = sorted(chunks, key=score, reverse=True)
        return ranked

    def _generate_with_retry(self, prompt: str, *, timeout_sec: int, temperature: float, retries: int) -> str:
        return self._generator.generate(
            prompt,
            timeout_sec=timeout_sec,
            temperature=temperature,
            retries=retries,
        )

    def ask(self, *, question: str, top_k: int = 5) -> dict:
        t0 = perf_counter()
        intent = classify_intent(question)
        retrieval_query = question
        if intent in {"cv_improvement", "career_transition"}:
            retrieval_query = f"{question}. yeu cau ky nang kinh nghiem du an trong tin tuyen dung data ai"
        elif intent == "job_recommendation":
            retrieval_query = f"{question}. goi y vi tri tuyen dung phu hop ky nang"

        chunks = self._retrieval.search(question=retrieval_query, top_k=max(6, top_k + 2))
        chunks = self._rerank_chunks(question, chunks, intent)[:top_k]
        prompt = build_grounded_prompt(
            question=question,
            retrieved_chunks=chunks,
            config=self._prompt_cfg,
            intent=intent,
        )

        used_fallback = False
        fallback_reason = ""
        fallback_stage = ""
        answer = ""
        try:
            answer = self._generate_with_retry(
                prompt,
                timeout_sec=self._prompt_cfg.ollama_timeout_sec,
                temperature=self._prompt_cfg.ollama_temperature,
                retries=self._prompt_cfg.ollama_retries,
            )
            if is_english_heavy(answer):
                rewrite_prompt = build_vietnamese_rewrite_prompt(answer=answer, question=question)
                try:
                    answer = self._generate_with_retry(
                        rewrite_prompt,
                        timeout_sec=self._prompt_cfg.ollama_timeout_sec,
                        temperature=0.15,
                        retries=self._prompt_cfg.ollama_retries,
                    )
                except Exception:
                    # Keep the original generated answer if rewrite step fails.
                    pass
        except Exception as exc:
            fallback_reason = f"{exc.__class__.__name__}: {str(exc)}"[:500]
            fallback_stage = "generate"
            if self._prompt_cfg.strict_no_fallback:
                raise RuntimeError(f"LLM generation failed at {fallback_stage}: {fallback_reason}") from exc
            used_fallback = True
            answer = build_fallback_answer(question, chunks)

        sources = [
            {
                "chunk_id": int(c["chunk_id"]),
                "document_id": int(c["document_id"]),
                "title": str(c.get("title", "")),
                "distance": float(c["distance"]),
            }
            for c in chunks
        ]
        _ = (perf_counter() - t0) * 1000.0
        return {
            "answer": answer,
            "sources": sources,
            "retrieval_count": len(chunks),
            "used_fallback": used_fallback,
            "fallback_reason": fallback_reason,
            "fallback_stage": fallback_stage,
        }

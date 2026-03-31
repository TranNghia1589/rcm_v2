from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import re
from typing import Any

from apps.api.app.services.rag.retrieval_service import RetrievalService
from src.infrastructure.db.postgres_client import PostgresClient, PostgresConfig
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
        self._postgres_cfg = PostgresConfig.from_yaml(self.postgres_config_path)

    def _to_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        text = str(value).strip()
        if not text:
            return []
        text = text.strip("[]")
        if not text:
            return []
        parts = [x.strip(" '\"") for x in text.split(",")]
        return [x for x in parts if x]

    def _load_cv_profile_context(self, cv_id: int | None) -> str:
        if cv_id is None:
            return ""
        try:
            with PostgresClient(self._postgres_cfg) as client:
                profile_row = client.fetch_one(
                    """
                    SELECT
                        COALESCE(target_role, ''),
                        COALESCE(experience_years::text, ''),
                        COALESCE(parsed_json->'skills', '[]'::jsonb),
                        COALESCE(career_objective, '')
                    FROM cv_profiles
                    WHERE cv_id = %s
                    LIMIT 1
                    """,
                    (int(cv_id),),
                )
                score_row = client.fetch_one(
                    """
                    SELECT
                        COALESCE(benchmark_role, ''),
                        COALESCE(grade, ''),
                        COALESCE(total_score, 0),
                        COALESCE(strengths, '[]'::jsonb),
                        COALESCE(missing_skills, '[]'::jsonb),
                        COALESCE(priority_skills, '[]'::jsonb)
                    FROM cv_scoring_results
                    WHERE cv_id = %s
                    LIMIT 1
                    """,
                    (int(cv_id),),
                )
                gap_row = client.fetch_one(
                    """
                    SELECT
                        COALESCE(target_role_from_cv, ''),
                        COALESCE(best_fit_roles, '[]'::jsonb),
                        COALESCE(missing_skills, '[]'::jsonb)
                    FROM cv_gap_reports
                    WHERE cv_id = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (int(cv_id),),
                )
        except Exception:
            return ""

        lines = [f"cv_id={int(cv_id)}"]
        if profile_row:
            target_role, exp_years, skills, objective = profile_row
            skill_list = self._to_list(skills)[:15]
            if target_role:
                lines.append(f"target_role={target_role}")
            if exp_years:
                lines.append(f"experience_years={exp_years}")
            if skill_list:
                lines.append(f"cv_skills={', '.join(skill_list)}")
            if objective:
                lines.append(f"career_objective={str(objective)[:220]}")

        if score_row:
            benchmark_role, grade, total_score, strengths, missing, priority = score_row
            if benchmark_role:
                lines.append(f"benchmark_role={benchmark_role}")
            if grade:
                lines.append(f"cv_grade={grade}")
            lines.append(f"cv_total_score={float(total_score):.2f}")
            strengths_list = self._to_list(strengths)[:10]
            missing_list = self._to_list(missing)[:10]
            priority_list = self._to_list(priority)[:10]
            if strengths_list:
                lines.append(f"strengths={', '.join(strengths_list)}")
            if missing_list:
                lines.append(f"missing_skills={', '.join(missing_list)}")
            if priority_list:
                lines.append(f"priority_skills={', '.join(priority_list)}")

        if gap_row:
            target_from_cv, best_fit_roles, gap_missing = gap_row
            if target_from_cv:
                lines.append(f"gap_target_role={target_from_cv}")
            best_fit_list = self._to_list(best_fit_roles)[:5]
            gap_missing_list = self._to_list(gap_missing)[:10]
            if best_fit_list:
                lines.append(f"best_fit_roles={', '.join(best_fit_list)}")
            if gap_missing_list:
                lines.append(f"gap_missing_skills={', '.join(gap_missing_list)}")

        return "\n".join(lines)

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

    def ask(self, *, question: str, top_k: int = 5, cv_id: int | None = None) -> dict:
        t0 = perf_counter()
        intent = classify_intent(question)
        profile_context = self._load_cv_profile_context(cv_id)
        retrieval_query = question
        if intent in {"cv_improvement", "career_transition"}:
            retrieval_query = f"{question}. yeu cau ky nang kinh nghiem du an trong tin tuyen dung data ai"
        elif intent == "job_recommendation":
            retrieval_query = f"{question}. goi y vi tri tuyen dung phu hop ky nang"
        if profile_context:
            retrieval_query = f"{retrieval_query}. du lieu cv: {profile_context.replace(chr(10), '; ')}"

        chunks = self._retrieval.search(question=retrieval_query, top_k=max(6, top_k + 2))
        chunks = self._rerank_chunks(question, chunks, intent)[:top_k]
        prompt = build_grounded_prompt(
            question=question,
            retrieved_chunks=chunks,
            config=self._prompt_cfg,
            intent=intent,
            profile_context=profile_context,
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

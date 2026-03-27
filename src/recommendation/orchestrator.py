from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.rag.generate import OllamaConfig, OllamaGenerator
from src.recommendation.candidate_generation import vector_candidates
from src.recommendation.explanation import generate_explanations
from src.recommendation.graph_ranking import graph_candidates


@dataclass
class HybridConfig:
    vector_top_k: int = 30
    graph_top_k: int = 30
    final_top_k: int = 10
    weight_vector: float = 0.45
    weight_graph: float = 0.55
    enable_llm_explanation: bool = True
    explanation_max_items: int = 5
    llm_explanation_timeout_sec: int = 90
    llm_explanation_retries: int = 1
    llm_explanation_temperature: float = 0.2
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "llama3:latest"
    ollama_keep_alive: str = "10m"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "HybridConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        weights = raw.get("weights", {}) or {}
        return cls(
            vector_top_k=int(raw.get("vector_top_k", 30)),
            graph_top_k=int(raw.get("graph_top_k", 30)),
            final_top_k=int(raw.get("final_top_k", 10)),
            weight_vector=float(weights.get("vector", 0.45)),
            weight_graph=float(weights.get("graph", 0.55)),
            enable_llm_explanation=bool(raw.get("enable_llm_explanation", True)),
            explanation_max_items=int(raw.get("explanation_max_items", 5)),
            llm_explanation_timeout_sec=int(raw.get("llm_explanation_timeout_sec", 90)),
            llm_explanation_retries=int(raw.get("llm_explanation_retries", 1)),
            llm_explanation_temperature=float(raw.get("llm_explanation_temperature", 0.2)),
            ollama_url=str(raw.get("ollama_url", "http://localhost:11434/api/generate")),
            ollama_model=str(raw.get("ollama_model", "llama3:latest")),
            ollama_keep_alive=str(raw.get("ollama_keep_alive", "10m")),
        )


def _normalize_scores(items: list[dict[str, Any]], key: str, out_key: str) -> None:
    if not items:
        return
    vals = [float(x.get(key, 0.0)) for x in items]
    lo = min(vals)
    hi = max(vals)
    if hi - lo < 1e-9:
        for x in items:
            x[out_key] = 1.0 if hi > 0 else 0.0
        return
    for x in items:
        x[out_key] = (float(x.get(key, 0.0)) - lo) / (hi - lo)


def run_hybrid_recommendation(
    *,
    question: str,
    cv_id: int,
    postgres_config_path: str | Path,
    neo4j_config_path: str | Path,
    retrieval_config_path: str | Path,
    embedding_config_path: str | Path,
    hybrid_config_path: str | Path,
) -> dict[str, Any]:
    cfg = HybridConfig.from_yaml(hybrid_config_path)

    vec = vector_candidates(
        question=question,
        postgres_config_path=postgres_config_path,
        retrieval_config_path=retrieval_config_path,
        embedding_config_path=embedding_config_path,
        top_k=cfg.vector_top_k,
    )
    grp = graph_candidates(
        cv_id=cv_id,
        neo4j_config_path=neo4j_config_path,
        top_k=cfg.graph_top_k,
    )

    by_job: dict[int, dict[str, Any]] = {}
    for x in vec:
        by_job[int(x["job_id"])] = {
            **x,
            "graph_score": 0.0,
            "matched_skills": 0,
            "total_required": 0,
            "coverage": 0.0,
        }
    for g in grp:
        job_id = int(g["job_id"])
        if job_id not in by_job:
            by_job[job_id] = {
                "job_id": job_id,
                "title": g.get("title", ""),
                "company_name": g.get("company_name", ""),
                "location": g.get("location", ""),
                "vector_score": 0.0,
                "supporting_chunk_ids": [],
                **g,
            }
        else:
            by_job[job_id].update(
                {
                    "matched_skills": g.get("matched_skills", 0),
                    "total_required": g.get("total_required", 0),
                    "coverage": g.get("coverage", 0.0),
                    "graph_score": g.get("graph_score", 0.0),
                }
            )

    items = list(by_job.values())
    _normalize_scores(items, "vector_score", "vector_norm")
    _normalize_scores(items, "graph_score", "graph_norm")

    for x in items:
        x["hybrid_score"] = (
            cfg.weight_vector * float(x.get("vector_norm", 0.0))
            + cfg.weight_graph * float(x.get("graph_norm", 0.0))
        )

    items.sort(key=lambda r: float(r.get("hybrid_score", 0.0)), reverse=True)
    final_items = items[: cfg.final_top_k]

    explanation = ""
    if cfg.enable_llm_explanation:
        explanation_generator = OllamaGenerator(
            OllamaConfig(
                url=cfg.ollama_url,
                model=cfg.ollama_model,
                timeout_sec=cfg.llm_explanation_timeout_sec,
                temperature=cfg.llm_explanation_temperature,
                retries=cfg.llm_explanation_retries,
                keep_alive=cfg.ollama_keep_alive,
            )
        )
        explanation = generate_explanations(
            question,
            final_items[: cfg.explanation_max_items],
            generator=explanation_generator,
            timeout_sec=cfg.llm_explanation_timeout_sec,
            temperature=cfg.llm_explanation_temperature,
            retries=cfg.llm_explanation_retries,
        )

    return {
        "cv_id": cv_id,
        "question": question,
        "items": final_items,
        "explanation": explanation,
    }

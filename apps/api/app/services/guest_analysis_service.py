from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import yaml

from apps.api.app.services.role_inference_service import RoleInferenceService
from apps.api.app.services.rag.retrieval_service import RetrievalService
from src.models.cv.extract_cv_info import extract_cv_info
from src.models.rag.embed import create_embedder_from_yaml
from src.models.rag.embedding_service_client import EmbeddingServiceClient
from src.models.rag.retrieve import retrieve_chunks


def _safe_intent(intent: str, query: str) -> str:
    raw = (intent or "").strip().lower()
    if raw in {"score", "recommend", "improve_cv"}:
        return raw
    q = (query or "").strip().lower()
    if any(k in q for k in ["goi y", "recommend", "job", "cong viec"]):
        return "recommend"
    if any(k in q for k in ["cai thien", "improve", "roadmap", "gap"]):
        return "improve_cv"
    return "score"


def _to_float_years(value: Any) -> float:
    txt = str(value or "").strip().lower()
    if not txt or txt == "unknown":
        return 0.0
    m = re.search(r"(\d+(?:\.\d+)?)", txt)
    if not m:
        return 0.0
    try:
        return float(m.group(1))
    except ValueError:
        return 0.0


def _grade(score: float) -> str:
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    return "D"


def _mean_vectors(a: list[float], b: list[float], wa: float = 0.55, wb: float = 0.45) -> list[float]:
    if not a:
        return b
    if not b:
        return a
    n = min(len(a), len(b))
    return [wa * float(a[i]) + wb * float(b[i]) for i in range(n)]


@dataclass
class GuestAnalysisService:
    role_profiles_path: str | Path
    postgres_config_path: str | Path
    retrieval_config_path: str | Path
    embedding_config_path: str | Path
    embedding_service_config_path: str | Path | None = None

    def __post_init__(self) -> None:
        self._retrieval = RetrievalService(
            postgres_config_path=self.postgres_config_path,
            retrieval_config_path=self.retrieval_config_path,
            embedding_config_path=self.embedding_config_path,
            embedding_service_config_path=self.embedding_service_config_path,
        )
        self._role = RoleInferenceService(self.role_profiles_path)
        self._embedder = None

    def _extract(self, *, file_bytes: bytes, file_name: str) -> dict[str, Any]:
        suffix = Path(file_name).suffix.lower()
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            return extract_cv_info(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _score(self, record: dict[str, Any]) -> dict[str, Any]:
        skills_count = len(record.get("skills") or [])
        projects_count = len(record.get("projects") or [])
        cert_count = len(record.get("certification_skills") or [])
        exp_years = _to_float_years(record.get("experience_years"))
        has_edu = 1.0 if (record.get("degree_names") or record.get("education_signals")) else 0.0

        skills_sub = min(100.0, (skills_count / 12.0) * 100.0)
        exp_sub = min(100.0, (exp_years / 3.0) * 100.0)
        projects_sub = min(100.0, (projects_count / 4.0) * 100.0)
        cert_sub = min(100.0, (cert_count / 3.0) * 100.0)
        edu_sub = 100.0 if has_edu > 0 else 40.0

        overall = (
            0.35 * skills_sub
            + 0.25 * exp_sub
            + 0.2 * projects_sub
            + 0.1 * cert_sub
            + 0.1 * edu_sub
        )
        overall = round(float(min(100.0, max(0.0, overall))), 2)

        return {
            "overall_score": overall,
            "grade": _grade(overall),
            "subscores": {
                "skills": round(skills_sub, 2),
                "experience": round(exp_sub, 2),
                "projects": round(projects_sub, 2),
                "certifications": round(cert_sub, 2),
                "education": round(edu_sub, 2),
            },
        }

    def _resolve_embedding_service_url(self) -> str:
        if not self.embedding_service_config_path:
            return ""
        try:
            raw = yaml.safe_load(Path(self.embedding_service_config_path).read_text(encoding="utf-8")) or {}
            return str(raw.get("url", "")).strip()
        except Exception:
            return ""

    def _embed_text(self, text: str) -> list[float]:
        query = (text or "").strip()
        if not query:
            return []

        service_url = self._resolve_embedding_service_url()
        if service_url:
            try:
                return [float(x) for x in EmbeddingServiceClient(service_url).embed_text(query)]
            except Exception:
                pass

        if self._embedder is None:
            self._embedder = create_embedder_from_yaml(self.embedding_config_path)
        return [float(x) for x in self._embedder.embed_text(query)]

    def _recommend(self, *, record: dict[str, Any], question: str, top_k: int) -> list[dict[str, Any]]:
        role = str(record.get("target_role") or "").strip()
        skills = [str(x).strip() for x in (record.get("skills") or []) if str(x).strip()]
        objective = str(record.get("career_objective") or "").strip()
        projects = [str(x).strip() for x in (record.get("projects") or []) if str(x).strip()][:4]

        query_parts = [question.strip(), role, ", ".join(skills[:12]), objective[:300]]
        query = ". ".join([x for x in query_parts if x])
        if not query:
            query = "goi y viec lam phu hop voi ky nang cv"

        cv_profile_text = "\n".join(
            [
                f"target_role: {role}",
                f"skills: {', '.join(skills[:20])}",
                f"career_objective: {objective[:500]}",
                f"projects: {'; '.join(projects)}",
            ]
        )

        try:
            q_vec = self._embed_text(query)
            cv_vec = self._embed_text(cv_profile_text)
            search_vec = _mean_vectors(q_vec, cv_vec)
            chunks = retrieve_chunks(
                question=query,
                postgres_config_path=self.postgres_config_path,
                retrieval_config_path=self.retrieval_config_path,
                embedding_config_path=self.embedding_config_path,
                query_embedding=search_vec,
                top_k_override=max(top_k * 2, top_k),
            )
        except Exception:
            chunks = self._retrieval.search(question=query, top_k=max(top_k * 2, top_k))

        dedup: dict[int, dict[str, Any]] = {}
        for c in chunks:
            doc_id = int(c.get("document_id", 0))
            if doc_id <= 0:
                continue
            old = dedup.get(doc_id)
            if old is None or float(c.get("distance", 1e9)) < float(old.get("distance", 1e9)):
                dedup[doc_id] = c

        sorted_rows = sorted(dedup.values(), key=lambda x: float(x.get("distance", 1e9)))[:top_k]
        out: list[dict[str, Any]] = []
        for i, row in enumerate(sorted_rows, start=1):
            content = str(row.get("content") or "").replace("\n", " ").strip()
            preview = content[:180] + ("..." if len(content) > 180 else "")
            out.append(
                {
                    "rank": i,
                    "job_title": str(row.get("title") or "Unknown"),
                    "distance": round(float(row.get("distance", 1.0)), 6),
                    "reason": preview,
                }
            )
        return out

    def _improve(self, *, record: dict[str, Any], role_result: dict[str, Any], limit: int = 8) -> list[dict[str, str]]:
        with open(self.role_profiles_path, "r", encoding="utf-8") as f:
            profiles = json.load(f) or {}

        selected_role = str(role_result.get("selected_role") or "")
        profile = None
        for _, p in profiles.items():
            role_name = str((p or {}).get("role_name") or "").strip()
            if role_name.lower() == selected_role.lower():
                profile = p
                break
        if profile is None and isinstance(profiles, dict):
            profile = profiles.get(selected_role)

        cv_skills = {str(x).strip().lower() for x in (record.get("skills") or []) if str(x).strip()}
        role_skills = [str(x).strip() for x in ((profile or {}).get("common_skills") or []) if str(x).strip()]

        missing = [s for s in role_skills if s.lower() not in cv_skills][:limit]
        out: list[dict[str, str]] = []
        for skill in missing:
            out.append(
                {
                    "skill": skill,
                    "why": f"Ky nang '{skill}' xuat hien pho bien trong role '{selected_role or 'target'}'.",
                }
            )
        return out

    def analyze(
        self,
        *,
        file_bytes: bytes,
        file_name: str,
        question: str,
        intent: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        record = self._extract(file_bytes=file_bytes, file_name=file_name)
        role_result = self._role.infer(cv_record=record, preferred_role=None, query=question, top_k=3)
        normalized_intent = _safe_intent(intent, question)

        score = self._score(record)
        recommendations: list[dict[str, Any]] = []
        improve_suggestions: list[dict[str, str]] = []

        if normalized_intent == "recommend":
            recommendations = self._recommend(record=record, question=question, top_k=max(1, int(top_k)))
        elif normalized_intent == "improve_cv":
            improve_suggestions = self._improve(record=record, role_result=role_result, limit=8)

        return {
            "intent": normalized_intent,
            "snapshot": {
                "target_role": str(record.get("target_role") or "Unknown"),
                "experience_years": str(record.get("experience_years") or "Unknown"),
                "skills": [str(x) for x in (record.get("skills") or [])][:20],
                "projects_count": len(record.get("projects") or []),
            },
            "score": score,
            "recommendations": recommendations,
            "improve_suggestions": improve_suggestions,
        }

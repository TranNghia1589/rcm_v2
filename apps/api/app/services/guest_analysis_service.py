from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import yaml

from apps.api.app.services.llm_service import get_ollama_service
from apps.api.app.services.role_inference_service import RoleInferenceService
from apps.api.app.services.rag.retrieval_service import RetrievalService
from src.models.cv.extract_cv_info import extract_cv_info
from src.models.rag.embed import create_embedder_from_yaml
from src.models.rag.embedding_service_client import EmbeddingServiceClient
from src.models.rag.retrieve import retrieve_chunks
from src.utils.infrastructure.db.postgres_client import PostgresClient, PostgresConfig


def _normalize_intent_text(text: str) -> str:
    raw = str(text or "").strip().lower()
    normalized = unicodedata.normalize("NFD", raw)
    no_diacritics = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", no_diacritics).strip()


def _safe_intent(intent: str, query: str, llm_intent: str | None = None) -> str:
    raw = (intent or "").strip().lower()
    if raw in {"score", "recommend", "improve_cv", "general"}:
        return raw

    if (llm_intent or "").strip().lower() in {"score", "recommend", "improve_cv", "general"}:
        return str(llm_intent).strip().lower()

    q_raw = (query or "").strip().lower()
    q_norm = _normalize_intent_text(query)
    q = f"{q_raw} {q_norm}".strip()

    recommend_keywords = [
        "goi y", "gợi ý", "recommend", "job", "viec lam", "việc làm", "cong viec", "công việc", "ung tuyen", "ứng tuyển",
    ]
    improve_keywords = [
        "cai thien", "cải thiện", "improve", "roadmap", "gap", "ky nang", "kỹ năng", "bo sung", "bổ sung",
    ]
    score_keywords = [
        "cham diem", "chấm điểm", "score", "danh gia", "đánh giá", "xep hang", "xếp hạng", "bao nhieu diem", "mấy điểm", "duoc cham",
    ]

    if any(k in q for k in improve_keywords):
        return "improve_cv"
    if any(k in q for k in recommend_keywords):
        return "recommend"
    if any(k in q for k in score_keywords):
        return "general"
    return "general"


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


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _extract_domain(text: str) -> str:
    """Extract domain/specialty from question context."""
    q = (str(text or "").strip().lower())
    domains = {
        "DATA_AI": ["data", "ai", "machine learning", "ml", "analytics", "python", "bigdata", "spark", "hadoop"],
        "WEB": ["web", "react", "vue", "angular", "nodejs", "frontend", "backend", "javascript", "typescript"],
        "MOBILE": ["mobile", "ios", "android", "flutter", "react native", "xamarin"],
        "DEVOPS": ["devops", "kubernetes", "docker", "cloud", "aws", "azure", "ci/cd", "jenkins"],
        "BACKEND": ["java", "spring", "enterprise", ".net", "c#"],
    }

    for domain, keywords in domains.items():
        if any(k in q for k in keywords):
            return domain
    return "GENERAL"


def _filter_skills_by_domain(skills: list[str], domain: str) -> list[str]:
    """Filter skills to prioritize those relevant to the domain."""
    if domain == "GENERAL":
        return skills

    domain_skills = {
        "DATA_AI": ["python", "sql", "machine learning", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "spark", "analytics", "statistics"],
        "WEB": ["react", "vue", "angular", "nodejs", "javascript", "typescript", "html", "css", "webpack", "api", "rest"],
        "MOBILE": ["ios", "android", "swift", "kotlin", "flutter", "react native", "mobile", "ui/ux"],
        "DEVOPS": ["docker", "kubernetes", "aws", "azure", "ci/cd", "jenkins", "terraform", "monitoring", "linux"],
        "BACKEND": ["java", "spring", "sql", "rest", "microservices", ".net", "enterprise", "architecture"],
    }

    relevant = domain_skills.get(domain, [])
    # Prioritize domain-relevant skills
    sorted_skills = sorted(
        skills,
        key=lambda s: any(r.lower() in s.lower() for r in relevant),
        reverse=True
    )
    return sorted_skills


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()



def _classify_intent_with_llm(question: str) -> str:
    try:
        llm_service = get_ollama_service()
        out = llm_service.classify_user_intent(question)
        return str(out or "").strip().lower()
    except Exception:
        return "general"
def _compatibility_percent(distance: float) -> float:
    score = (1.0 - float(distance)) * 100.0
    score = max(0.0, min(100.0, score))
    return round(score, 2)


def _extract_qty(requirements: str) -> str:
    txt = _norm_ws(requirements)
    m = re.search(r"s[oố]\s*l[uư][oợ]ng\s*tuy[eể]n\s*[:\-]?\s*([^.;,\n]+)", txt, flags=re.IGNORECASE)
    if m:
        return _norm_ws(m.group(1))
    return ""


def _build_job_description(requirements: str, exp_min: Any, exp_max: Any) -> str:
    req = _norm_ws(requirements)
    req_short = req[:220] + ("..." if len(req) > 220 else "")

    exp_txt = ""
    if exp_min is not None and exp_max is not None:
        exp_txt = f"Kinh nghiệm: {float(exp_min):g}-{float(exp_max):g} năm"
    elif exp_min is not None:
        exp_txt = f"Kinh nghiệm tối thiểu: {float(exp_min):g} năm"
    elif exp_max is not None:
        exp_txt = f"Kinh nghiệm tối đa: {float(exp_max):g} năm"

    qty_txt = _extract_qty(req)
    qty_line = f"Số lượng tuyển: {qty_txt}" if qty_txt else ""

    parts = [p for p in [exp_txt, qty_line, req_short] if p]
    return " | ".join(parts) if parts else "Chưa có mô tả chi tiết."


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

    def _fetch_job_details(self, job_urls: list[str]) -> dict[str, dict[str, Any]]:
        urls = [u.strip() for u in job_urls if str(u).strip()]
        if not urls:
            return {}

        placeholders = ",".join(["%s"] * len(urls))
        q = f"""
            SELECT job_url, requirements, experience_min_years, experience_max_years
            FROM jobs
            WHERE job_url IN ({placeholders})
        """

        result: dict[str, dict[str, Any]] = {}
        try:
            pg_cfg = PostgresConfig.from_yaml(self.postgres_config_path)
            with PostgresClient(pg_cfg) as client:
                rows = client.fetch_all(q, tuple(urls))
                for r in rows:
                    url = str(r[0] or "").strip()
                    if not url:
                        continue
                    result[url] = {
                        "requirements": str(r[1] or ""),
                        "experience_min_years": r[2],
                        "experience_max_years": r[3],
                    }
        except Exception:
            return {}
        return result

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

        job_urls = [str(x.get("source_id") or "").strip() for x in sorted_rows if str(x.get("source_id") or "").strip()]
        detail_map = self._fetch_job_details(job_urls)

        out: list[dict[str, Any]] = []
        for i, row in enumerate(sorted_rows, start=1):
            job_url = str(row.get("source_id") or "").strip()
            detail = detail_map.get(job_url, {})
            req_text = str(detail.get("requirements") or "")
            exp_min = detail.get("experience_min_years")
            exp_max = detail.get("experience_max_years")

            if not req_text:
                req_text = str(row.get("content") or "")

            out.append(
                {
                    "rank": i,
                    "job_title": str(row.get("title") or "Unknown"),
                    "compatibility_percent": _compatibility_percent(float(row.get("distance", 1.0))),
                    "job_description": _build_job_description(req_text, exp_min, exp_max),
                    "job_url": job_url,
                }
            )
        return out

    def _improve(self, *, record: dict[str, Any], role_result: dict[str, Any], limit: int = 8, question: str = "") -> list[dict[str, str]]:
        selected_role = str(role_result.get("selected_role") or "")
        domain = _extract_domain(question)
        
        # Prepare CV context for LLM
        cv_skills = [str(x).strip() for x in (record.get("skills") or []) if str(x).strip()]
        current_role = str(record.get("target_role") or "").strip()
        years_exp = str(record.get("experience_years") or "").strip()
        
        cv_context = f"""
Current Role: {current_role}
Experience: {years_exp}
Current Skills: {', '.join(cv_skills[:15])}
"""
        
        # Try to use LLM for intelligent suggestions
        try:
            llm_service = get_ollama_service()
            suggestions = llm_service.generate_improvement_suggestions(
                cv_context=cv_context,
                target_domain=domain,
                target_role=selected_role,
                question=question,
            )
            
            # If LLM returns suggestions, use them (prioritize LLM)
            if suggestions and len(suggestions) > 0:
                return suggestions[:limit]
        except Exception as e:
            print(f"[LLM fallback] {str(e)}, using keyword-based suggestions instead")
        
        # Fallback to keyword-based suggestions
        with open(self.role_profiles_path, "r", encoding="utf-8") as f:
            profiles = json.load(f) or {}

        profile = None
        for _, p in profiles.items():
            role_name = str((p or {}).get("role_name") or "").strip()
            if role_name.lower() == selected_role.lower():
                profile = p
                break
        if profile is None and isinstance(profiles, dict):
            profile = profiles.get(selected_role)

        cv_skills_set = {str(x).strip().lower() for x in (record.get("skills") or []) if str(x).strip()}
        role_skills = [str(x).strip() for x in ((profile or {}).get("common_skills") or []) if str(x).strip()]

        # Filter and prioritize skills based on domain
        filtered_role_skills = _filter_skills_by_domain(role_skills, domain)
        
        missing = [s for s in filtered_role_skills if s.lower() not in cv_skills_set][:limit]
        out: list[dict[str, str]] = []
        for skill in missing:
            why_msg = f"Kỹ năng '{skill}' quan trọng cho role '{selected_role or 'target'}'."
            if domain != "GENERAL":
                why_msg += f" [{domain}]"
            out.append({
                "skill": skill,
                "why": why_msg,
            })
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
        llm_intent = _classify_intent_with_llm(question) if (intent or "").strip().lower() == "auto" else None
        normalized_intent = _safe_intent(intent, question, llm_intent=llm_intent)

        score = self._score(record)
        recommendations: list[dict[str, Any]] = []
        improve_suggestions: list[dict[str, str]] = []

        if normalized_intent == "recommend":
            recommendations = self._recommend(record=record, question=question, top_k=max(1, int(top_k)))
        elif normalized_intent == "improve_cv":
            improve_suggestions = self._improve(record=record, role_result=role_result, limit=8, question=question)

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






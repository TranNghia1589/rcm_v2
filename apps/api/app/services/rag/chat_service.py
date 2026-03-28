from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
import re
from typing import Any
from uuid import UUID

from apps.api.app.services.rag.retrieval_service import RetrievalService
from src.cv.extract_cv_info import clean_project_entries
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
        self._retrieval = RetrievalService(
            postgres_config_path=self.postgres_config_path,
            retrieval_config_path=self.retrieval_config_path,
            embedding_config_path=self.embedding_config_path,
        )
        self._pg_cfg = PostgresConfig.from_yaml(self.postgres_config_path)
        self._prompt_cfg = PromptingConfig.from_yaml(self.prompting_config_path)
        self._generator = self._build_generator(self._prompt_cfg)

    def _build_generator(self, prompt_cfg: PromptingConfig) -> OllamaGenerator:
        return OllamaGenerator(
            OllamaConfig(
                url=prompt_cfg.ollama_url,
                model=prompt_cfg.ollama_model,
                timeout_sec=prompt_cfg.ollama_timeout_sec,
                temperature=prompt_cfg.ollama_temperature,
                retries=prompt_cfg.ollama_retries,
                retry_backoff_sec=prompt_cfg.ollama_retry_backoff_sec,
                keep_alive=prompt_cfg.ollama_keep_alive,
            )
        )

    def _refresh_runtime_config(self) -> None:
        latest_cfg = PromptingConfig.from_yaml(self.prompting_config_path)
        current = (
            self._prompt_cfg.ollama_url,
            self._prompt_cfg.ollama_model,
            self._prompt_cfg.ollama_timeout_sec,
            self._prompt_cfg.ollama_temperature,
            self._prompt_cfg.ollama_retries,
            self._prompt_cfg.ollama_retry_backoff_sec,
            self._prompt_cfg.ollama_keep_alive,
            self._prompt_cfg.max_context_chunks,
            self._prompt_cfg.max_history_messages,
            self._prompt_cfg.strict_no_fallback,
        )
        refreshed = (
            latest_cfg.ollama_url,
            latest_cfg.ollama_model,
            latest_cfg.ollama_timeout_sec,
            latest_cfg.ollama_temperature,
            latest_cfg.ollama_retries,
            latest_cfg.ollama_retry_backoff_sec,
            latest_cfg.ollama_keep_alive,
            latest_cfg.max_context_chunks,
            latest_cfg.max_history_messages,
            latest_cfg.strict_no_fallback,
        )
        if refreshed != current:
            self._prompt_cfg = latest_cfg
            self._generator = self._build_generator(latest_cfg)
        else:
            self._prompt_cfg = latest_cfg

    def _generate_with_retry(self, prompt: str, *, timeout_sec: int, temperature: float, retries: int) -> str:
        return self._generator.generate(
            prompt,
            timeout_sec=timeout_sec,
            temperature=temperature,
            retries=retries,
        )

    def _safe_json(self, value: Any, default: Any) -> Any:
        if value is None:
            return default
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return default
        return default

    def _normalize_uuid(self, session_id: str | None) -> str | None:
        if not session_id:
            return None
        try:
            return str(UUID(str(session_id)))
        except (ValueError, TypeError):
            return None

    def _resolve_existing_user_id(self, client: PostgresClient, user_id: int | None) -> int | None:
        if not user_id:
            return None
        row = client.fetch_one("SELECT user_id FROM users WHERE user_id = %s", (user_id,))
        if not row:
            return None
        return int(row[0])

    def _load_session(self, client: PostgresClient, session_id: str | None) -> dict[str, Any] | None:
        normalized = self._normalize_uuid(session_id)
        if not normalized:
            return None
        row = client.fetch_one(
            """
            SELECT session_id::text, user_id, cv_id, gap_report_id, title, model_name
            FROM chat_sessions
            WHERE session_id = %s::uuid
            """,
            (normalized,),
        )
        if not row:
            return None
        return {
            "session_id": str(row[0]),
            "user_id": int(row[1]) if row[1] is not None else None,
            "cv_id": int(row[2]) if row[2] is not None else None,
            "gap_report_id": int(row[3]) if row[3] is not None else None,
            "title": str(row[4]) if row[4] else None,
            "model_name": str(row[5]) if row[5] else "",
        }

    def _load_recent_history(self, client: PostgresClient, session_id: str | None) -> list[dict[str, str]]:
        normalized = self._normalize_uuid(session_id)
        if not normalized:
            return []
        rows = client.fetch_all(
            """
            SELECT role, content
            FROM chat_messages
            WHERE session_id = %s::uuid
            ORDER BY message_id DESC
            LIMIT %s
            """,
            (normalized, max(1, int(self._prompt_cfg.max_history_messages))),
        )
        history = [
            {"role": str(role), "content": str(content)}
            for role, content in reversed(rows)
            if content
        ]
        return history

    def _stringify_items(self, value: Any, limit: int = 8) -> str:
        if not value:
            return ""
        data = value
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return data[:300]
        if isinstance(data, list):
            rendered: list[str] = []
            for item in data[:limit]:
                if isinstance(item, dict):
                    label = (
                        item.get("role")
                        or item.get("skill")
                        or item.get("name")
                        or item.get("title")
                        or item.get("label")
                    )
                    score = item.get("score")
                    if label and score is not None:
                        rendered.append(f"{label} ({score})")
                    elif label:
                        rendered.append(str(label))
                    else:
                        rendered.append(json.dumps(item, ensure_ascii=False))
                else:
                    rendered.append(str(item))
            return ", ".join(x for x in rendered if x)
        if isinstance(data, dict):
            compact = []
            for key in [
                "role",
                "score",
                "matched_skills",
                "missing_skills",
                "recommended_next_skills",
                "common_skills",
            ]:
                if key in data and data[key]:
                    compact.append(f"{key}={data[key]}")
            if compact:
                return "; ".join(compact)
            return json.dumps(data, ensure_ascii=False)[:300]
        return str(data)

    def _append_context_part(self, parts: list[str], label: str, value: Any, *, limit: int = 8) -> None:
        rendered = self._stringify_items(value, limit=limit)
        if rendered:
            parts.append(f"{label}: {rendered}")

    def _load_cv_context(
        self,
        client: PostgresClient,
        *,
        cv_id: int | None,
        gap_report_id: int | None,
    ) -> dict[str, Any]:
        context: dict[str, Any] = {
            "cv_id": None,
            "gap_report_id": None,
            "user_id": None,
            "summary_text": "",
            "parsed_json": {},
            "target_role": None,
            "experience_years": None,
            "skills": [],
            "projects": [],
            "education_signals": [],
            "domain_fit": None,
            "best_fit_roles": [],
            "strengths": [],
            "missing_skills": [],
            "top_role_result": {},
            "role_ranking": [],
            "development_plan": [],
            "recommended_next_skills": [],
            "common_skills": [],
        }
        gap_row = None
        resolved_cv_id = cv_id
        resolved_gap_report_id = gap_report_id

        if gap_report_id:
            gap_row = client.fetch_one(
                """
                SELECT
                    gap_report_id,
                    cv_id,
                    domain_fit,
                    target_role_from_cv,
                    best_fit_roles,
                    strengths,
                    missing_skills,
                    top_role_result,
                    role_ranking,
                    market_gap_json
                FROM cv_gap_reports
                WHERE gap_report_id = %s
                """,
                (gap_report_id,),
            )
            if gap_row:
                resolved_gap_report_id = int(gap_row[0])
                resolved_cv_id = int(gap_row[1])

        cv_row = None
        if resolved_cv_id:
            cv_row = client.fetch_one(
                """
                SELECT cv_id, user_id, file_name, raw_text, parsed_json, target_role, experience_years
                FROM cv_profiles
                WHERE cv_id = %s
                """,
                (resolved_cv_id,),
            )

        if not cv_row:
            return context

        context["cv_id"] = int(cv_row[0])
        context["user_id"] = int(cv_row[1]) if cv_row[1] is not None else None
        context["target_role"] = str(cv_row[5]) if cv_row[5] is not None else None
        context["experience_years"] = float(cv_row[6]) if cv_row[6] is not None else None
        parsed_json = self._safe_json(cv_row[4], {})
        context["parsed_json"] = parsed_json
        skill_rows = client.fetch_all(
            """
            SELECT s.canonical_name
            FROM cv_skills cs
            JOIN skills s ON s.skill_id = cs.skill_id
            WHERE cs.cv_id = %s
            ORDER BY COALESCE(cs.confidence, 0) DESC, s.canonical_name
            LIMIT 15
            """,
            (context["cv_id"],),
        )
        skill_names = [str(row[0]) for row in skill_rows if row and row[0]]
        parsed_skills = parsed_json.get("skills", []) if isinstance(parsed_json, dict) else []
        merged_skills = skill_names or []
        for skill in parsed_skills:
            skill_text = str(skill).strip()
            if skill_text and skill_text not in merged_skills:
                merged_skills.append(skill_text)
        context["skills"] = merged_skills

        if not gap_row:
            gap_row = client.fetch_one(
                """
                SELECT
                    gap_report_id,
                    cv_id,
                    domain_fit,
                    target_role_from_cv,
                    best_fit_roles,
                    strengths,
                    missing_skills,
                    top_role_result,
                    role_ranking,
                    market_gap_json
                FROM cv_gap_reports
                WHERE cv_id = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (context["cv_id"],),
            )
            if gap_row:
                resolved_gap_report_id = int(gap_row[0])

        parts = [
            f"CV ID: {context['cv_id']}",
            f"File name: {cv_row[2] or 'Unknown'}",
            f"Target role trong CV: {cv_row[5] or 'Chưa rõ'}",
            f"Số năm kinh nghiệm: {cv_row[6] if cv_row[6] is not None else 'Chưa rõ'}",
        ]
        if isinstance(parsed_json, dict):
            if parsed_json.get("full_name"):
                parts.append(f"Họ tên ứng viên: {parsed_json.get('full_name')}")
            if parsed_json.get("email"):
                parts.append(f"Email ứng viên: {parsed_json.get('email')}")
            if parsed_json.get("phone"):
                parts.append(f"Số điện thoại: {parsed_json.get('phone')}")
        if merged_skills:
            parts.append(f"Kỹ năng trích xuất: {', '.join(merged_skills[:15])}")

        parsed_projects = parsed_json.get("projects", []) if isinstance(parsed_json, dict) else []
        parsed_education = parsed_json.get("education_signals", []) if isinstance(parsed_json, dict) else []
        raw_preview = parsed_json.get("raw_text_preview") if isinstance(parsed_json, dict) else None
        context["projects"] = clean_project_entries(parsed_projects if isinstance(parsed_projects, list) else [])
        context["education_signals"] = parsed_education if isinstance(parsed_education, list) else []

        parsed_summary = ""
        for key in ["summary", "profile_summary", "objective", "headline"]:
            value = parsed_json.get(key) if isinstance(parsed_json, dict) else None
            if value:
                parsed_summary = str(value)
                break
        if parsed_summary:
            parts.append(f"Tóm tắt hồ sơ: {parsed_summary[:500]}")
        elif raw_preview:
            parts.append(f"Tóm tắt CV đã xử lý: {str(raw_preview)[:500]}")
        elif cv_row[3]:
            parts.append(f"Trích đoạn CV: {str(cv_row[3]).replace(chr(10), ' ')[:500]}")
        self._append_context_part(parts, "Project hoặc kinh nghiệm đã trích xuất", parsed_projects, limit=5)
        self._append_context_part(parts, "Education signals", parsed_education, limit=5)

        if gap_row:
            context["gap_report_id"] = int(gap_row[0])
            best_fit_roles = self._safe_json(gap_row[4], [])
            strengths = self._safe_json(gap_row[5], [])
            missing_skills = self._safe_json(gap_row[6], [])
            top_role = self._safe_json(gap_row[7], {})
            role_ranking = self._safe_json(gap_row[8], [])
            market_gap = self._safe_json(gap_row[9], {})
            context["domain_fit"] = str(gap_row[2]) if gap_row[2] else None
            context["best_fit_roles"] = best_fit_roles if isinstance(best_fit_roles, list) else []
            context["strengths"] = strengths if isinstance(strengths, list) else []
            context["missing_skills"] = missing_skills if isinstance(missing_skills, list) else []
            context["top_role_result"] = top_role if isinstance(top_role, dict) else {}
            context["role_ranking"] = role_ranking if isinstance(role_ranking, list) else []
            if gap_row[2]:
                parts.append(f"Domain fit: {gap_row[2]}")
            if gap_row[3]:
                parts.append(f"Target role từ gap report: {gap_row[3]}")
            self._append_context_part(parts, "Best fit roles", best_fit_roles, limit=5)
            self._append_context_part(parts, "Điểm mạnh", strengths, limit=8)
            self._append_context_part(parts, "Kỹ năng hoặc tín hiệu còn thiếu", missing_skills, limit=8)
            self._append_context_part(parts, "Top role result", top_role, limit=6)
            self._append_context_part(parts, "Role ranking", role_ranking, limit=4)
            if isinstance(market_gap, dict):
                context["development_plan"] = market_gap.get("development_plan", []) or []
                context["recommended_next_skills"] = market_gap.get("recommended_next_skills", []) or []
                context["common_skills"] = market_gap.get("common_skills", []) or []
                self._append_context_part(
                    parts,
                    "Development plan",
                    market_gap.get("development_plan", []),
                    limit=6,
                )
                self._append_context_part(
                    parts,
                    "Recommended next skills",
                    market_gap.get("recommended_next_skills", []),
                    limit=8,
                )
                self._append_context_part(
                    parts,
                    "Common skills of target role",
                    market_gap.get("common_skills", []),
                    limit=8,
                )

        context["summary_text"] = "\n".join(parts)
        return context

    def _format_list(self, items: list[Any], limit: int = 5) -> str:
        values = [str(item).strip() for item in items if str(item).strip()]
        if not values:
            return ""
        return ", ".join(values[:limit])

    def _fit_label(self, domain_fit: str | None) -> str:
        mapping = {
            "high": "cao",
            "medium": "trung bình",
            "low": "thấp",
        }
        return mapping.get((domain_fit or "").lower(), "chưa rõ")

    def _build_contextual_advice(
        self,
        *,
        role: str,
        strengths: list[Any],
        missing_skills: list[Any],
        projects: list[Any],
        development_plan: list[Any],
    ) -> list[str]:
        advice: list[str] = []
        missing = [str(item).strip() for item in missing_skills if str(item).strip()]
        strong = [str(item).strip() for item in strengths if str(item).strip()]
        project_text = [str(item).strip() for item in projects if str(item).strip()]
        plan = [str(item).strip() for item in development_plan if str(item).strip()]

        if "Python" in missing:
            advice.append("Ưu tiên bù Python trước, vì đây là khoảng trống dễ bị hỏi ngay ở vòng screening cho role này.")
        if "Statistics" in missing:
            advice.append("Ôn lại Statistics theo hướng thực dụng: descriptive stats, hypothesis testing, cách đọc insight từ dữ liệu.")
        if "Power BI" in missing or "Dashboarding" in strong:
            advice.append("Nếu đã có nền tảng dashboard, hãy đóng gói lại thành 1 case study ngắn với bài toán, dữ liệu, KPI và tác động.")
        if "SQL" in strong:
            advice.append("Nên chuẩn bị 2-3 ví dụ truy vấn SQL bạn từng dùng để làm sạch dữ liệu, join bảng hoặc tạo báo cáo.")
        if not project_text:
            advice.append("Bổ sung ít nhất 1 project gần role mục tiêu để CV bớt thiên về mô tả kỹ năng chung.")
        else:
            advice.append("Hãy viết lại project theo form bối cảnh, việc bạn làm, công cụ dùng và kết quả đo được để CV thuyết phục hơn.")
        if plan:
            advice.append("Thứ tự nên đi là: " + " -> ".join(plan[:3]) + ".")
        if not advice:
            advice.append(f"Nên tiếp tục đào sâu các tín hiệu đã có để CV cho role {role} trở nên cụ thể và đáng tin hơn.")
        return advice[:4]

    def _build_context_locked_answer(
        self,
        *,
        question: str,
        intent: str,
        cv_context: dict[str, Any],
        history: list[dict[str, str]],
    ) -> str:
        if not cv_context.get("cv_id"):
            return ""

        role = (
            cv_context.get("top_role_result", {}).get("role")
            or (cv_context.get("best_fit_roles") or [None])[0]
            or cv_context.get("target_role")
            or "role mục tiêu"
        )
        domain_fit = str(cv_context.get("domain_fit") or "").lower() or None
        fit_label = self._fit_label(domain_fit)
        experience = cv_context.get("experience_years")
        skills = list(cv_context.get("skills") or [])
        strengths = list(cv_context.get("strengths") or [])
        missing_skills = list(cv_context.get("missing_skills") or [])
        projects = list(cv_context.get("projects") or [])
        education = list(cv_context.get("education_signals") or [])
        development_plan = list(cv_context.get("development_plan") or [])
        recommended_next_skills = list(cv_context.get("recommended_next_skills") or [])
        common_skills = list(cv_context.get("common_skills") or [])
        top_role = cv_context.get("top_role_result") or {}
        score = top_role.get("score")

        if intent == "cv_improvement":
            lines = [
                f"CV này hiện phù hợp ở mức {fit_label} với vị trí {role}."
            ]
            evidence: list[str] = []
            if experience is not None:
                evidence.append(f"{experience:.0f} năm kinh nghiệm")
            if strengths:
                evidence.append("điểm mạnh nổi bật là " + self._format_list(strengths, 5))
            elif skills:
                evidence.append("đã có các kỹ năng " + self._format_list(skills, 6))
            if projects:
                evidence.append("có project liên quan như " + self._format_list(projects, 2))
            if score is not None:
                evidence.append(f"score fit hiện tại khoảng {score}")
            if evidence:
                lines.append("Căn cứ chính: " + "; ".join(evidence) + ".")

            if missing_skills:
                lines.append("Thiếu hụt ưu tiên cần bù: " + self._format_list(missing_skills, 6) + ".")
            elif common_skills:
                lines.append("Kỹ năng nên rà soát thêm so với role mục tiêu: " + self._format_list(common_skills, 6) + ".")

            if projects:
                lines.append(
                    "Từ project đã xử lý trong CV, HR hoặc hiring manager sẽ kỳ vọng bạn giải thích rõ cách bạn dùng "
                    + self._format_list(projects, 2)
                    + "."
                )

            plan = development_plan or recommended_next_skills or missing_skills
            if plan:
                lines.append("Ưu tiên cải thiện tiếp theo: " + self._format_list(plan, 5) + ".")
            if education:
                lines.append("Tín hiệu học vấn/chứng chỉ hiện có: " + self._format_list(education, 4) + ".")
            advice = self._build_contextual_advice(
                role=role,
                strengths=strengths,
                missing_skills=missing_skills,
                projects=projects,
                development_plan=development_plan or recommended_next_skills,
            )
            if advice:
                lines.append("Tư vấn sát CV hiện tại:")
                lines.extend(f"- {item}" for item in advice)
            return "\n".join(lines)

        if intent == "hr_recruiter":
            lines = [
                f"Mức fit hiện tại của ứng viên cho role {role}: {fit_label}."
            ]
            positive_signals: list[str] = []
            if experience is not None:
                positive_signals.append(f"{experience:.0f} năm kinh nghiệm")
            if strengths:
                positive_signals.append("strengths: " + self._format_list(strengths, 5))
            if projects:
                positive_signals.append("project gần role: " + self._format_list(projects, 2))
            if positive_signals:
                lines.append("Tín hiệu tốt: " + "; ".join(positive_signals) + ".")

            if missing_skills:
                lines.append("Điểm cần xác minh thêm: " + self._format_list(missing_skills, 5) + ".")

            screening_questions: list[str] = []
            if projects:
                screening_questions.append(
                    "Hãy mô tả chi tiết project "
                    + self._format_list(projects, 1)
                    + " và vai trò thực tế của bạn trong project đó."
                )
            for skill in missing_skills[:3]:
                screening_questions.append(f"Bạn đã dùng {skill} ở mức nào trong công việc hoặc project thật?")
            if not screening_questions and strengths:
                for skill in strengths[:2]:
                    screening_questions.append(f"Bạn có thể đưa ví dụ cụ thể khi dùng {skill} để tạo ra kết quả gì?")
            if screening_questions:
                lines.append("Câu hỏi screening nên hỏi:")
                lines.extend(f"- {item}" for item in screening_questions[:4])
            decision = "nên mời interview vòng đầu" if domain_fit == "high" else "nên screening kỹ trước khi mời interview"
            lines.append(f"Khuyến nghị tuyển dụng: {decision}.")
            return "\n".join(lines)

        if intent == "career_transition":
            lines = [
                f"Nếu đang đi theo hướng {role}, mức fit hiện tại là {fit_label}."
            ]
            if strengths:
                lines.append("Nền tảng đã có thể tận dụng: " + self._format_list(strengths, 5) + ".")
            if projects:
                lines.append("Project hiện có nên khai thác lại: " + self._format_list(projects, 2) + ".")
            if missing_skills:
                lines.append("Khoảng trống cần bù trước: " + self._format_list(missing_skills, 5) + ".")
            plan = development_plan or recommended_next_skills or missing_skills
            if plan:
                lines.append("Lộ trình ưu tiên sát CV hiện tại: " + self._format_list(plan, 5) + ".")
            advice = self._build_contextual_advice(
                role=role,
                strengths=strengths,
                missing_skills=missing_skills,
                projects=projects,
                development_plan=development_plan or recommended_next_skills,
            )
            if advice:
                lines.append("Tư vấn hành động:")
                lines.extend(f"- {item}" for item in advice)
            return "\n".join(lines)

        if intent == "job_recommendation":
            lines = [
                f"Role gần nhất với CV hiện tại là {role}, mức fit {fit_label}."
            ]
            if strengths:
                lines.append("Lý do phù hợp: " + self._format_list(strengths, 5) + ".")
            if missing_skills:
                lines.append("Để tăng khả năng trúng tuyển, nên bù: " + self._format_list(missing_skills, 5) + ".")
            if projects:
                lines.append("Nên nhấn mạnh project: " + self._format_list(projects, 2) + ".")
            advice = self._build_contextual_advice(
                role=role,
                strengths=strengths,
                missing_skills=missing_skills,
                projects=projects,
                development_plan=development_plan or recommended_next_skills,
            )
            if advice:
                lines.append("Tư vấn apply:")
                lines.extend(f"- {item}" for item in advice[:3])
            return "\n".join(lines)

        return ""

    def _looks_follow_up(self, question: str) -> bool:
        q = (question or "").strip().lower()
        if not q:
            return False
        starters = [
            "vậy",
            "thế",
            "còn",
            "rồi",
            "nếu vậy",
            "tiếp theo",
            "với cv này",
            "cv này",
            "hr",
            "recruiter",
        ]
        if any(q.startswith(token) for token in starters):
            return True
        if len(q.split()) <= 8 and any(token in q for token in ["thiếu gì", "ổn không", "hợp không", "được không"]):
            return True
        return False

    def _build_retrieval_query(
        self,
        *,
        question: str,
        intent: str,
        history: list[dict[str, str]],
        cv_context_text: str,
    ) -> str:
        parts = [question.strip()]
        if self._looks_follow_up(question) and history:
            recent_user_turns = [x["content"] for x in history if x["role"] == "user"][-2:]
            if recent_user_turns:
                parts.append("Ngữ cảnh câu trước: " + " | ".join(recent_user_turns))
        if cv_context_text and intent in {
            "cv_improvement",
            "job_recommendation",
            "career_transition",
            "hr_recruiter",
        }:
            parts.append("Ngữ cảnh CV: " + cv_context_text[:500])
        if intent == "career_transition":
            parts.append("Tap trung vao roadmap ky nang, project, role chuyen doi.")
        elif intent == "job_recommendation":
            parts.append("Tap trung vao role, yeu cau tuyen dung, skill match.")
        elif intent == "hr_recruiter":
            parts.append("Tap trung vao screening, fit, risk, interview signals.")
        return ". ".join(p for p in parts if p).strip()

    def _sanitize_answer(self, answer: str, chunks: list[dict[str, Any]]) -> str:
        if not answer:
            return answer
        valid_chunk_ids = {str(int(c["chunk_id"])) for c in chunks if c.get("chunk_id") is not None}

        def replace_invalid(match: re.Match[str]) -> str:
            value = match.group(1)
            if not valid_chunk_ids:
                return ""
            return match.group(0) if value in valid_chunk_ids else ""

        sanitized = re.sub(r"\s*\[chunk_id=(\d+)\]", replace_invalid, answer)
        sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
        return sanitized.strip()

    def _rerank_chunks(self, question: str, chunks: list[dict[str, Any]], intent: str) -> list[dict[str, Any]]:
        q_tokens = set(re.findall(r"[a-zA-ZÀ-ỹ0-9]+", (question or "").lower()))
        intent_boost = {
            "job_recommendation": {"job", "việc", "vị", "tuyển", "position", "role"},
            "cv_improvement": {"cv", "kỹ", "năng", "skills", "improve"},
            "career_transition": {"chuyển", "trái", "ngành", "lộ", "trình", "roadmap"},
            "hr_recruiter": {"hr", "recruiter", "screen", "interview", "ứng", "viên"},
            "industry_insight": {"ngành", "market", "xu", "hướng", "role", "career"},
            "general": set(),
        }
        boosts = intent_boost.get(intent, set())

        def score(c: dict[str, Any]) -> float:
            text = f"{c.get('title', '')} {c.get('content', '')}".lower()
            t_tokens = set(re.findall(r"[a-zA-ZÀ-ỹ0-9]+", text))
            overlap = len(q_tokens & t_tokens)
            intent_overlap = len(boosts & t_tokens)
            dist = float(c.get("distance", 1.0))
            return overlap * 1.5 + intent_overlap * 2.0 - dist * 2.0

        return sorted(chunks, key=score, reverse=True)

    def _ensure_session(
        self,
        client: PostgresClient,
        *,
        session_id: str | None,
        user_id: int | None,
        cv_id: int | None,
        gap_report_id: int | None,
        title: str | None,
    ) -> str:
        normalized = self._normalize_uuid(session_id)
        model_name = self._prompt_cfg.ollama_model
        if normalized:
            existing = self._load_session(client, normalized)
            if existing:
                client.execute(
                    """
                    UPDATE chat_sessions
                    SET user_id = COALESCE(%s, user_id),
                        cv_id = COALESCE(%s, cv_id),
                        gap_report_id = COALESCE(%s, gap_report_id),
                        title = COALESCE(%s, title),
                        model_name = %s
                    WHERE session_id = %s::uuid
                    """,
                    (user_id, cv_id, gap_report_id, title, model_name, normalized),
                )
                return normalized

        row = client.fetch_one(
            """
            INSERT INTO chat_sessions (user_id, cv_id, gap_report_id, title, model_name)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING session_id::text
            """,
            (user_id, cv_id, gap_report_id, title, model_name),
        )
        return str(row[0])

    def _save_message(
        self,
        client: PostgresClient,
        *,
        session_id: str,
        role: str,
        content: str,
        retrieved_chunk_ids: list[int] | None = None,
        grounded: bool | None = None,
        latency_ms: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        client.execute(
            """
            INSERT INTO chat_messages (
                session_id,
                role,
                content,
                retrieved_chunk_ids,
                grounded,
                latency_ms,
                metadata
            )
            VALUES (%s::uuid, %s, %s, %s, %s, %s, %s::jsonb)
            """,
            (
                session_id,
                role,
                content,
                retrieved_chunk_ids or None,
                grounded,
                latency_ms,
                json.dumps(metadata or {}, ensure_ascii=False),
            ),
        )

    def ask(
        self,
        *,
        question: str,
        top_k: int = 5,
        session_id: str | None = None,
        user_id: int | None = None,
        cv_id: int | None = None,
        gap_report_id: int | None = None,
        title: str | None = None,
    ) -> dict[str, Any]:
        self._refresh_runtime_config()
        t0 = perf_counter()
        intent = classify_intent(question)
        requested_session_id = self._normalize_uuid(session_id)

        session: dict[str, Any] | None = None
        history: list[dict[str, str]] = []
        cv_context: dict[str, Any] = {"cv_id": None, "gap_report_id": None, "user_id": None, "summary_text": ""}
        resolved_user_id = user_id

        with PostgresClient(self._pg_cfg) as client:
            session = self._load_session(client, requested_session_id)
            if session:
                resolved_user_id = resolved_user_id or session["user_id"]
                cv_id = cv_id or session["cv_id"]
                gap_report_id = gap_report_id or session["gap_report_id"]
                title = title or session["title"]
            resolved_user_id = self._resolve_existing_user_id(client, resolved_user_id)
            history = self._load_recent_history(client, session["session_id"] if session else requested_session_id)
            cv_context = self._load_cv_context(client, cv_id=cv_id, gap_report_id=gap_report_id)
            if cv_context["user_id"] is not None and resolved_user_id is None:
                resolved_user_id = int(cv_context["user_id"])

        retrieval_query = self._build_retrieval_query(
            question=question,
            intent=intent,
            history=history,
            cv_context_text=str(cv_context.get("summary_text", "")),
        )

        chunks = self._retrieval.search(question=retrieval_query, top_k=max(6, top_k + 2))
        chunks = self._rerank_chunks(question, chunks, intent)[:top_k]

        prompt = build_grounded_prompt(
            question=question,
            retrieved_chunks=chunks,
            config=self._prompt_cfg,
            intent=intent,
            conversation_history=history,
            internal_cv_context=str(cv_context.get("summary_text", "")),
        )

        used_fallback = False
        fallback_reason = ""
        fallback_stage = ""
        answer = ""
        answer_mode = "llm"
        locked_answer = ""
        if cv_context.get("cv_id") and intent in {
            "cv_improvement",
            "job_recommendation",
            "career_transition",
            "hr_recruiter",
        }:
            locked_answer = self._build_context_locked_answer(
                question=question,
                intent=intent,
                cv_context=cv_context,
                history=history,
            )
        try:
            if locked_answer and not chunks:
                answer = locked_answer
                answer_mode = "context_locked"
            else:
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
                        pass
        except Exception as exc:
            fallback_reason = f"{exc.__class__.__name__}: {str(exc)}"[:500]
            fallback_stage = "generate"
            if self._prompt_cfg.strict_no_fallback:
                raise RuntimeError(f"LLM generation failed at {fallback_stage}: {fallback_reason}") from exc
            used_fallback = True
            answer = locked_answer or build_fallback_answer(question, chunks)
            answer_mode = "context_locked_fallback" if locked_answer else "fallback"

        answer = self._sanitize_answer(answer, chunks)
        latency_ms = int((perf_counter() - t0) * 1000.0)
        sources = [
            {
                "chunk_id": int(c["chunk_id"]),
                "document_id": int(c["document_id"]),
                "title": str(c.get("title", "")),
                "distance": float(c["distance"]),
            }
            for c in chunks
        ]

        saved_to_history = False
        history_error = ""
        active_session_id = requested_session_id
        try:
            with PostgresClient(self._pg_cfg) as client:
                active_session_id = self._ensure_session(
                    client,
                    session_id=requested_session_id,
                    user_id=resolved_user_id,
                    cv_id=cv_context["cv_id"],
                    gap_report_id=cv_context["gap_report_id"],
                    title=title or question[:80],
                )
                self._save_message(
                    client,
                    session_id=active_session_id,
                    role="user",
                    content=question,
                    metadata={
                        "intent": intent,
                        "top_k": top_k,
                        "retrieval_query": retrieval_query,
                        "resolved_cv_id": cv_context["cv_id"],
                        "resolved_gap_report_id": cv_context["gap_report_id"],
                    },
                )
                self._save_message(
                    client,
                    session_id=active_session_id,
                    role="assistant",
                    content=answer,
                    retrieved_chunk_ids=[int(c["chunk_id"]) for c in chunks],
                    grounded=bool(chunks) and not used_fallback,
                    latency_ms=latency_ms,
                    metadata={
                        "intent": intent,
                        "used_fallback": used_fallback,
                        "fallback_reason": fallback_reason,
                        "fallback_stage": fallback_stage,
                        "retrieval_count": len(chunks),
                        "history_turns_used": len(history),
                        "source_document_ids": [int(c["document_id"]) for c in chunks],
                        "answer_mode": answer_mode,
                    },
                )
                saved_to_history = True
        except Exception as exc:
            history_error = f"{exc.__class__.__name__}: {str(exc)}"[:500]

        return {
            "answer": answer,
            "sources": sources,
            "retrieval_count": len(chunks),
            "used_fallback": used_fallback,
            "fallback_reason": fallback_reason,
            "fallback_stage": fallback_stage,
            "session_id": active_session_id,
            "resolved_cv_id": cv_context["cv_id"],
            "resolved_gap_report_id": cv_context["gap_report_id"],
            "history_turns_used": len(history),
            "saved_to_history": saved_to_history,
            "history_error": history_error,
            "debug": {
                "intent": intent,
                "retrieval_query": retrieval_query,
                "history_messages_loaded": len(history),
                "has_cv_context": bool(cv_context["summary_text"]),
                "answer_mode": answer_mode,
            },
        }

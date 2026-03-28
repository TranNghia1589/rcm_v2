from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from uuid import uuid4

from src.cv.extract_cv_info import clean_project_entries, extract_cv_info, load_cv_text
from src.infrastructure.db.postgres_client import PostgresClient, PostgresConfig
from src.matching.gap_analysis import analyze_cv_against_roles, build_development_plan, load_json


@dataclass
class CVService:
    postgres_config_path: str | Path
    role_profiles_path: str | Path

    def __post_init__(self) -> None:
        self._pg_cfg = PostgresConfig.from_yaml(self.postgres_config_path)
        self._role_profiles = load_json(Path(self.role_profiles_path))

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

    def _normalize_experience_years(self, value: Any) -> float | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() == "unknown":
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _ensure_user(
        self,
        client: PostgresClient,
        *,
        user_id: int | None,
        email: str | None,
        full_name: str | None,
        phone: str | None,
    ) -> int:
        if user_id is not None:
            row = client.fetch_one("SELECT user_id FROM users WHERE user_id = %s", (int(user_id),))
            if row:
                client.execute(
                    """
                    UPDATE users
                    SET email = COALESCE(%s, email),
                        full_name = COALESCE(%s, full_name),
                        phone = COALESCE(%s, phone)
                    WHERE user_id = %s
                    """,
                    (
                        (email or "").strip() or None,
                        (full_name or "").strip() or None,
                        (phone or "").strip() or None,
                        int(user_id),
                    ),
                )
                return int(user_id)

        safe_email = (email or "").strip()
        safe_name = (full_name or "").strip() or "Guest User"
        safe_phone = (phone or "").strip() or None
        if not safe_email:
            safe_email = f"guest_cv_{uuid4().hex[:12]}@local"

        row = client.fetch_one(
            """
            INSERT INTO users (email, full_name, phone)
            VALUES (%s, %s, %s)
            ON CONFLICT (email) DO UPDATE
            SET full_name = COALESCE(EXCLUDED.full_name, users.full_name),
                phone = COALESCE(EXCLUDED.phone, users.phone)
            RETURNING user_id
            """,
            (safe_email, safe_name, safe_phone),
        )
        return int(row[0])

    def _ensure_skill_id(self, client: PostgresClient, skill_name: str) -> int:
        row = client.fetch_one(
            """
            INSERT INTO skills (canonical_name)
            VALUES (%s)
            ON CONFLICT (canonical_name) DO UPDATE
            SET canonical_name = EXCLUDED.canonical_name
            RETURNING skill_id
            """,
            (skill_name.strip().lower(),),
        )
        return int(row[0])

    def _insert_cv_skills(self, client: PostgresClient, *, cv_id: int, skills: list[str]) -> None:
        for skill in skills:
            skill_name = str(skill).strip()
            if not skill_name:
                continue
            skill_id = self._ensure_skill_id(client, skill_name)
            client.execute(
                """
                INSERT INTO cv_skills (cv_id, skill_id, source, confidence)
                VALUES (%s, %s, 'extractor', %s)
                ON CONFLICT (cv_id, skill_id, source) DO NOTHING
                """,
                (cv_id, skill_id, 0.95),
            )

    def _build_gap_result(self, extracted: dict[str, Any]) -> dict[str, Any]:
        gap_result = analyze_cv_against_roles(extracted, self._role_profiles)
        top_role_result = gap_result.get("top_role_result", {})
        development_plan = build_development_plan(top_role_result, extracted)
        gap_result["development_plan"] = development_plan
        gap_result["market_gap_json"] = {
            "recommended_next_skills": top_role_result.get("recommended_next_skills", []),
            "common_skills": top_role_result.get("common_skills", []),
            "development_plan": development_plan,
            "projects": extracted.get("projects", []),
            "education_signals": extracted.get("education_signals", []),
            "cv_skills": extracted.get("skills", []),
        }
        return gap_result

    def upload_cv(
        self,
        *,
        file_name: str,
        content: bytes,
        user_id: int | None = None,
        email: str | None = None,
        full_name: str | None = None,
        phone: str | None = None,
        source_type: str = "upload",
    ) -> dict[str, Any]:
        suffix = Path(file_name or "cv.txt").suffix.lower()
        if suffix not in {".pdf", ".docx", ".txt"}:
            raise ValueError("Unsupported CV format. Only .pdf, .docx, .txt are allowed.")
        if not content:
            raise ValueError("Uploaded CV is empty.")

        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                temp_path = tmp.name

            raw_text = load_cv_text(temp_path)
            extracted = extract_cv_info(temp_path)
            extracted["file_name"] = file_name
            extracted["projects"] = clean_project_entries(list(extracted.get("projects", [])))
            if email and not extracted.get("email"):
                extracted["email"] = email
            if phone and not extracted.get("phone"):
                extracted["phone"] = phone
            if full_name and not extracted.get("full_name"):
                extracted["full_name"] = full_name
            gap_result = self._build_gap_result(extracted)

            with PostgresClient(self._pg_cfg) as client:
                resolved_user_id = self._ensure_user(
                    client,
                    user_id=user_id,
                    email=extracted.get("email") or email,
                    full_name=extracted.get("full_name") or full_name,
                    phone=extracted.get("phone") or phone,
                )
                cv_row = client.fetch_one(
                    """
                    INSERT INTO cv_profiles (
                        user_id,
                        file_name,
                        source_type,
                        raw_text,
                        parsed_json,
                        target_role,
                        experience_years,
                        education_signals
                    )
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s::jsonb)
                    RETURNING cv_id, created_at, updated_at
                    """,
                    (
                        resolved_user_id,
                        file_name,
                        source_type,
                        raw_text,
                        json.dumps(extracted, ensure_ascii=False),
                        extracted.get("target_role"),
                        self._normalize_experience_years(extracted.get("experience_years")),
                        json.dumps(extracted.get("education_signals", []), ensure_ascii=False),
                    ),
                )
                cv_id = int(cv_row[0])
                self._insert_cv_skills(client, cv_id=cv_id, skills=list(extracted.get("skills", [])))
                gap_row = client.fetch_one(
                    """
                    INSERT INTO cv_gap_reports (
                        cv_id,
                        domain_fit,
                        target_role_from_cv,
                        best_fit_roles,
                        strengths,
                        missing_skills,
                        top_role_result,
                        role_ranking,
                        market_gap_json,
                        model_name
                    )
                    VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s)
                    RETURNING gap_report_id
                    """,
                    (
                        cv_id,
                        gap_result.get("domain_fit"),
                        gap_result.get("target_role_from_cv"),
                        json.dumps(gap_result.get("best_fit_roles", []), ensure_ascii=False),
                        json.dumps(gap_result.get("strengths", []), ensure_ascii=False),
                        json.dumps(gap_result.get("missing_skills", []), ensure_ascii=False),
                        json.dumps(gap_result.get("top_role_result", {}), ensure_ascii=False),
                        json.dumps(gap_result.get("role_ranking", []), ensure_ascii=False),
                        json.dumps(gap_result.get("market_gap_json", {}), ensure_ascii=False),
                        "gap_analysis_v1",
                    ),
                )
                gap_report_id = int(gap_row[0])

            return {
                "user_id": resolved_user_id,
                "cv_id": cv_id,
                "gap_report_id": gap_report_id,
                "file_name": file_name,
                "target_role": extracted.get("target_role"),
                "experience_years": self._normalize_experience_years(extracted.get("experience_years")),
                "skills": list(extracted.get("skills", [])),
                "extracted": extracted,
                "gap_report": gap_result,
            }
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def get_cv_detail(self, cv_id: int) -> dict[str, Any] | None:
        with PostgresClient(self._pg_cfg) as client:
            row = client.fetch_one(
                """
                SELECT
                    cv_id,
                    user_id,
                    file_name,
                    source_type,
                    target_role,
                    experience_years,
                    raw_text,
                    parsed_json,
                    education_signals,
                    created_at,
                    updated_at
                FROM cv_profiles
                WHERE cv_id = %s
                """,
                (int(cv_id),),
            )
            if not row:
                return None
            skill_rows = client.fetch_all(
                """
                SELECT s.canonical_name
                FROM cv_skills cs
                JOIN skills s ON s.skill_id = cs.skill_id
                WHERE cs.cv_id = %s
                ORDER BY s.canonical_name
                """,
                (int(cv_id),),
            )
            gap_row = client.fetch_one(
                """
                SELECT
                    gap_report_id,
                    domain_fit,
                    target_role_from_cv,
                    best_fit_roles,
                    strengths,
                    missing_skills,
                    top_role_result,
                    role_ranking,
                    market_gap_json,
                    model_name,
                    created_at
                FROM cv_gap_reports
                WHERE cv_id = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (int(cv_id),),
            )

        latest_gap_report: dict[str, Any] = {}
        if gap_row:
            latest_gap_report = {
                "gap_report_id": int(gap_row[0]),
                "domain_fit": gap_row[1],
                "target_role_from_cv": gap_row[2],
                "best_fit_roles": self._safe_json(gap_row[3], []),
                "strengths": self._safe_json(gap_row[4], []),
                "missing_skills": self._safe_json(gap_row[5], []),
                "top_role_result": self._safe_json(gap_row[6], {}),
                "role_ranking": self._safe_json(gap_row[7], []),
                "market_gap_json": self._safe_json(gap_row[8], {}),
                "model_name": gap_row[9],
                "created_at": str(gap_row[10]),
            }

        parsed_json = self._safe_json(row[7], {})
        if isinstance(parsed_json, dict):
            parsed_json["projects"] = clean_project_entries(list(parsed_json.get("projects", [])))

        return {
            "cv_id": int(row[0]),
            "user_id": int(row[1]),
            "file_name": str(row[2]),
            "source_type": str(row[3]),
            "target_role": str(row[4]) if row[4] is not None else None,
            "experience_years": float(row[5]) if row[5] is not None else None,
            "raw_text": str(row[6] or ""),
            "parsed_json": parsed_json,
            "education_signals": self._safe_json(row[8], []),
            "skills": [str(skill_row[0]) for skill_row in skill_rows if skill_row and skill_row[0]],
            "latest_gap_report": latest_gap_report,
            "created_at": str(row[9]),
            "updated_at": str(row[10]),
        }

    def list_user_cvs(self, user_id: int) -> dict[str, Any]:
        with PostgresClient(self._pg_cfg) as client:
            rows = client.fetch_all(
                """
                SELECT cv_id, user_id, file_name, target_role, experience_years, created_at, updated_at
                FROM cv_profiles
                WHERE user_id = %s
                ORDER BY created_at DESC
                """,
                (int(user_id),),
            )
        items = [
            {
                "cv_id": int(row[0]),
                "user_id": int(row[1]),
                "file_name": str(row[2]),
                "target_role": str(row[3]) if row[3] is not None else None,
                "experience_years": float(row[4]) if row[4] is not None else None,
                "created_at": str(row[5]),
                "updated_at": str(row[6]),
            }
            for row in rows
        ]
        return {"user_id": int(user_id), "items": items}

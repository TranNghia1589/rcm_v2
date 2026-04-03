from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _norm_skill_set(values: list[str]) -> set[str]:
    out: set[str] = set()
    for item in values:
        txt = _norm_text(str(item))
        if txt:
            out.add(txt)
    return out


@dataclass
class RoleInferenceService:
    role_profiles_path: str | Path

    def _load_profiles(self) -> dict[str, dict[str, Any]]:
        with open(self.role_profiles_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else {}

    def infer(
        self,
        *,
        cv_record: dict[str, Any],
        preferred_role: str | None = None,
        query: str = "",
        top_k: int = 3,
    ) -> dict[str, Any]:
        profiles = self._load_profiles()
        cv_skills = _norm_skill_set([str(x) for x in (cv_record.get("skills") or [])])
        extracted_role = str(cv_record.get("target_role") or "").strip()
        signal_text = " ".join(
            [
                str(cv_record.get("raw_text_preview") or ""),
                str(cv_record.get("career_objective") or ""),
                str(query or ""),
            ]
        ).lower()

        rows: list[dict[str, Any]] = []
        for role_key, profile in profiles.items():
            role_name = str(profile.get("role_name") or role_key).strip() or role_key
            common_skills = _norm_skill_set([str(x) for x in (profile.get("common_skills") or [])])
            common_keywords = _norm_skill_set([str(x) for x in (profile.get("common_keywords") or [])])

            skill_ratio = 0.0
            if common_skills:
                skill_ratio = len(cv_skills & common_skills) / max(len(common_skills), 1)

            kw_hit = 0
            kw_total = 0
            for kw in common_keywords:
                if len(kw) < 3:
                    continue
                kw_total += 1
                if kw in signal_text:
                    kw_hit += 1
            keyword_ratio = kw_hit / max(kw_total, 1) if kw_total else 0.0

            title_bonus = 1.0 if _norm_text(extracted_role) == _norm_text(role_name) else 0.0
            confidence = 0.65 * skill_ratio + 0.2 * keyword_ratio + 0.15 * title_bonus

            reasons: list[str] = []
            if skill_ratio > 0:
                reasons.append(f"matched_skills_ratio={skill_ratio:.2f}")
            if keyword_ratio > 0:
                reasons.append(f"keyword_overlap={keyword_ratio:.2f}")
            if title_bonus > 0:
                reasons.append("matched_extracted_target_role")

            rows.append(
                {
                    "role": role_name,
                    "confidence": round(max(0.0, min(confidence, 1.0)), 4),
                    "reasons": reasons or ["low_signal_match"],
                }
            )

        rows.sort(key=lambda x: float(x["confidence"]), reverse=True)
        top = rows[: max(1, int(top_k))]
        selected = top[0] if top else {"role": "Unknown", "confidence": 0.0, "reasons": ["no_profiles"]}

        preferred = _norm_text(preferred_role or "")
        if preferred:
            for item in rows:
                if _norm_text(item["role"]) == preferred:
                    selected = item
                    break

        selected_conf = float(selected.get("confidence", 0.0))
        return {
            "selected_role": str(selected.get("role", "Unknown")),
            "confidence": round(selected_conf, 4),
            "requires_confirmation": selected_conf < 0.45 and not preferred,
            "candidates": top,
        }

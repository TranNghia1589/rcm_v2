from __future__ import annotations

import re
from typing import Any

from src.data_contracts.schemas import CVExtractedRecord, SCHEMA_VERSION


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"none", "null", "nan", "n/a", "na", "[none]", "[]"}:
        return ""
    return text


def _clean_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, tuple):
        raw_items = list(value)
    else:
        text = _clean_text(value)
        if not text:
            return []
        raw_items = [x.strip() for x in re.split(r"[,\n;|]", text)]

    out: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        v = _clean_text(item)
        if not v:
            continue
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def normalize_cv_record(payload: dict[str, Any]) -> dict[str, Any]:
    record = CVExtractedRecord()

    for key in record.to_dict().keys():
        if key not in payload:
            continue
        value = payload.get(key)
        if isinstance(getattr(record, key), list):
            setattr(record, key, _clean_list(value))
        elif isinstance(getattr(record, key), str):
            setattr(record, key, _clean_text(value))
        else:
            setattr(record, key, value)

    if not record.schema_version:
        record.schema_version = SCHEMA_VERSION

    if not record.cv_id and record.file_name:
        record.cv_id = record.file_name

    if not record.target_role:
        record.target_role = "Unknown"
    if not record.experience_years:
        record.experience_years = "Unknown"

    return record.to_dict()


def validate_cv_record(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = ["schema_version", "file_name", "skills", "target_role", "experience_years"]
    for key in required:
        if key not in payload:
            errors.append(f"Missing required field: {key}")

    if str(payload.get("schema_version", "")).strip() != SCHEMA_VERSION:
        errors.append(f"Unsupported schema_version: {payload.get('schema_version')}")

    if not str(payload.get("file_name", "")).strip():
        errors.append("file_name must not be empty")

    skills = payload.get("skills", [])
    if not isinstance(skills, list):
        errors.append("skills must be a list[str]")

    score = payload.get("matched_score")
    if score is not None:
        try:
            _ = float(score)
        except Exception:
            errors.append("matched_score must be float or null")

    return errors

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[3]

SKILL_TEXT_COLUMNS = [
    "title",
    "detail_title",
    "tags",
    "desc_mota",
    "desc_yeucau",
    "desc_quyenloi",
]

ROLE_COLUMN_CANDIDATES = [
    "source_field_name",
    "role",
    "category",
    "field",
]

EXP_COLUMN_CANDIDATES = [
    "detail_experience",
    "exp_list",
    "experience",
]

WORD_RE = re.compile(r"[a-z0-9\+\#\.\-]{2,}")
SPACE_RE = re.compile(r"\s+")

STOPWORDS = {
    "va",
    "voi",
    "cho",
    "cac",
    "nhu",
    "co",
    "khong",
    "trong",
    "tai",
    "mot",
    "nam",
    "kinh",
    "nghiem",
    "ung",
    "vien",
    "data",
    "ai",
    "ml",
    "job",
    "the",
    "to",
    "for",
    "with",
    "and",
    "or",
    "of",
    "in",
    "on",
    "a",
    "an",
    "is",
    "as",
    "be",
    "that",
    "this",
}


def _normalize_space(text: str) -> str:
    return SPACE_RE.sub(" ", text.strip())


def normalize_role_name(raw: str) -> str:
    txt = _normalize_space(str(raw)).replace("_", " ").replace("-", " ")
    if not txt:
        return "Unknown"
    titled = " ".join(part.capitalize() for part in txt.split())
    return titled


def normalize_alias(alias: str) -> str:
    txt = _normalize_space(str(alias).lower())
    txt = txt.replace("–", "-")
    return txt


def latest_job_raw_path(root: Path | None = None) -> Path:
    base = (root or BASE_DIR) / "data" / "raw" / "jobs"
    candidates = sorted(
        [p for p in base.glob("*") if p.suffix.lower() in {".xlsx", ".csv", ".parquet", ".jsonl"}],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No job raw file found under: {base}")
    return candidates[0]


def _safe_cell_text(value: Any) -> str:
    if value is None:
        return ""
    txt = str(value).strip()
    if txt.lower() in {"nan", "none"}:
        return ""
    return txt


def _read_xlsx_rows(path: Path) -> list[dict[str, Any]]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    def col_to_index(ref: str) -> int:
        letters = "".join(ch for ch in ref if ch.isalpha())
        idx = 0
        for ch in letters:
            idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
        return max(0, idx - 1)

    with ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", ns):
                parts = [node.text or "" for node in si.findall(".//a:t", ns)]
                shared_strings.append("".join(parts))

        sheet_root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows = sheet_root.findall(".//a:sheetData/a:row", ns)
        if not rows:
            return []

        def parse_cell(cell: ET.Element) -> str:
            cell_type = cell.attrib.get("t")
            value_node = cell.find("a:v", ns)
            if value_node is not None and value_node.text is not None:
                raw = value_node.text
                if cell_type == "s":
                    try:
                        return shared_strings[int(raw)]
                    except Exception:
                        return raw
                return raw
            inline_node = cell.find("a:is", ns)
            if inline_node is not None:
                return "".join(part.text or "" for part in inline_node.findall(".//a:t", ns))
            return ""

        parsed_rows: list[list[str]] = []
        for row in rows:
            cells = row.findall("a:c", ns)
            if not cells:
                continue
            max_index = max(col_to_index(c.attrib.get("r", "")) for c in cells)
            values = [""] * (max_index + 1)
            for cell in cells:
                idx = col_to_index(cell.attrib.get("r", ""))
                values[idx] = _safe_cell_text(parse_cell(cell))
            parsed_rows.append(values)

    if not parsed_rows:
        return []
    headers = parsed_rows[0]
    records: list[dict[str, Any]] = []
    for row in parsed_rows[1:]:
        record = {}
        for i, key in enumerate(headers):
            if not key:
                continue
            value = row[i] if i < len(row) else ""
            record[str(key)] = _safe_cell_text(value)
        records.append(record)
    return records


def load_records(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input path not found: {p}")
    suffix = p.suffix.lower()

    if suffix == ".csv":
        with open(p, "r", encoding="utf-8-sig", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                txt = line.strip()
                if txt:
                    rows.append(json.loads(txt))
        return rows
    if suffix == ".parquet":
        df = pd.read_parquet(p)
        return df.fillna("").to_dict(orient="records")
    if suffix == ".xlsx":
        return _read_xlsx_rows(p)

    raise ValueError(f"Unsupported input format: {p}")


def pick_first_key(record: dict[str, Any], candidates: Iterable[str]) -> str:
    for key in candidates:
        if key in record and str(record.get(key, "")).strip():
            return str(record.get(key, "")).strip()
    return ""


def build_skill_text(record: dict[str, Any]) -> str:
    chunks = []
    for key in SKILL_TEXT_COLUMNS:
        value = _safe_cell_text(record.get(key, ""))
        if value:
            chunks.append(value)
    return "\n".join(chunks)


def extract_keywords(texts: list[str], top_k: int = 10, min_len: int = 3) -> list[str]:
    freq: dict[str, int] = {}
    for text in texts:
        lowered = str(text).lower()
        for token in WORD_RE.findall(lowered):
            if len(token) < min_len:
                continue
            if token.isdigit():
                continue
            if token in STOPWORDS:
                continue
            freq[token] = freq.get(token, 0) + 1
    sorted_tokens = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [token for token, _ in sorted_tokens[:top_k]]


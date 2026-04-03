from __future__ import annotations

import sys

def display(obj):
    """
    Console-safe display for script mode.
    Avoid IPython display() here because Windows cp1252 console can raise UnicodeEncodeError.
    """
    text = str(obj)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(safe_text)

import re
import os
import json
import math
import html
import time
import unicodedata
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

try:
    from underthesea import word_tokenize
    HAS_UNDERTHESEA = True
except Exception:
    HAS_UNDERTHESEA = False

pd.set_option("display.max_columns", 200)
pd.set_option("display.max_colwidth", 200)
pd.set_option("display.width", 200)

NOTEBOOK_VERSION = "preprocessing_final_phobert"

BASE_DIR = Path(__file__).resolve().parents[3]
RAW_JOBS_DIR = BASE_DIR / "data" / "raw" / "jobs"


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

def find_latest_raw_file(data_dir: Path) -> Path:
    candidates = list(data_dir.glob("topcv_all_fields_merged_*.csv")) + list(data_dir.glob("topcv_all_fields_merged_*.xlsx"))
    if not candidates:
        raise FileNotFoundError(f"Khong tim thay raw jobs file trong: {data_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

RAW_INPUT_PATH = find_latest_raw_file(RAW_JOBS_DIR)
ARTIFACT_DIR = BASE_DIR / "experiments" / "artifacts" / "matching"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TAG = os.getenv("PREPROCESS_OUTPUT_TAG", "").strip()


def artifact_name(base: str) -> str:
    return f"{base}_{OUTPUT_TAG}" if OUTPUT_TAG else base

RUN_EMBEDDING = env_bool("PREPROCESS_RUN_EMBEDDING", True)
RUN_SECTION_EMBEDDING = env_bool("PREPROCESS_RUN_SECTION_EMBEDDING", True)
SAVE_INTERMEDIATE = True

PHOBERT_MODEL_NAME = "vinai/phobert-base"
PHOBERT_MAX_LENGTH_MATCH = 256
PHOBERT_MAX_LENGTH_CHATBOT = 320
PHOBERT_MAX_LENGTH_CHUNK = 256
PHOBERT_BATCH_SIZE = 16
NORMALIZE_EMBEDDINGS = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("NOTEBOOK_VERSION:", NOTEBOOK_VERSION)
print("RAW_INPUT_PATH:", RAW_INPUT_PATH)
print("ARTIFACT_DIR:", ARTIFACT_DIR.resolve())
print("RUN_EMBEDDING:", RUN_EMBEDDING)
print("RUN_SECTION_EMBEDDING:", RUN_SECTION_EMBEDDING)
print("PHOBERT_MODEL_NAME:", PHOBERT_MODEL_NAME)
print("DEVICE:", DEVICE)
print("HAS_UNDERTHESEA:", HAS_UNDERTHESEA)

def normalize_empty_value(val):
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None

    s = str(val).strip()
    if not s:
        return None

    lowered = s.lower().strip()
    empty_tokens = {
        "nan", "none", "null", "n/a", "na", "-", "--", "---",
        "không", "không rõ", "chưa rõ", "chưa cập nhật", "đang cập nhật",
        "not specified", "unknown"
    }
    return None if lowered in empty_tokens else s

def safe_str(x):
    x = normalize_empty_value(x)
    return "" if x is None else str(x)

def get_series(df, col, default=None):
    if col in df.columns:
        return df[col]
    if default is None:
        default = [None] * len(df)
    return pd.Series(default, index=df.index)

def first_non_empty(*values):
    for v in values:
        v = normalize_empty_value(v)
        if v is not None:
            return v
    return None

def normalize_unicode(text):
    text = safe_str(text)
    return unicodedata.normalize("NFC", text)

def remove_html(text):
    text = safe_str(text)
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return text

def normalize_dash(text):
    text = safe_str(text)
    dash_map = {
        "–": "-",
        "—": "-",
        "−": "-",
        "•": "-",
        "●": "-",
        "▪": "-",
        "►": "-",
        "✅": "-",
        "✔": "-",
    }
    for k, v in dash_map.items():
        text = text.replace(k, v)
    return text

def deduplicate_list(values):
    out = []
    seen = set()
    for v in values:
        key = safe_str(v).strip()
        if not key:
            continue
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out

def deduplicate_text_lines(text, min_key_len=12):
    text = safe_str(text)
    if not text:
        return ""
    out_lines = []
    seen = set()
    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        key = re.sub(r"\s+", " ", raw.lower())
        if len(key) < min_key_len:
            out_lines.append(raw)
            continue
        if key not in seen:
            seen.add(key)
            out_lines.append(raw)
    return "\n".join(out_lines).strip()

def clean_text_light(text):
    text = normalize_empty_value(text)
    if text is None:
        return ""

    text = normalize_unicode(text)
    text = remove_html(text)
    text = normalize_dash(text)
    text = text.replace("\\n", "\n")
    text = re.sub(r"[\u200b-\u200f\uFEFF]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = deduplicate_text_lines(text)
    return text.strip()

def clean_text_preserve_structure(text):
    text = clean_text_light(text)
    if not text:
        return ""
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n\s*-\s*", "\n- ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def clean_text_strict(text):
    text = clean_text_light(text)
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ/+\-#\.]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_text_for_phobert(text):
    text = normalize_empty_value(text)
    if text is None:
        return ""

    text = normalize_unicode(text)
    text = remove_html(text)
    text = normalize_dash(text)
    text = text.replace("\\n", "\n")

    text = re.sub(r"[\u200b-\u200f\uFEFF]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    text = re.sub(r"[^\w\sÀ-ỹ\.,;:/\-\+\#\(\)%\n]", " ", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)

    return text.strip()

def truncate_by_words(text, max_words=220):
    text = safe_str(text)
    if not text:
        return ""
    words = text.split()
    return " ".join(words[:max_words]).strip()

def save_table(df, base_path: Path):
    base_path = Path(base_path)
    try:
        out_path = str(base_path) + ".parquet"
        df.to_parquet(out_path, index=False)
        return out_path
    except Exception:
        out_path = str(base_path) + ".csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        return out_path

def load_raw_data(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(input_path)
    elif suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    print(f"[INFO] Loaded raw data: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


df_raw = load_raw_data(RAW_INPUT_PATH)
display(df_raw.head(3))
print(df_raw.columns.tolist())

def merge_semantic_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    out["job_url"] = get_series(df, "job_url")
    out["job_id"] = get_series(df, "job_id")

    out["job_title_raw"] = [
        first_non_empty(a, b)
        for a, b in zip(
            get_series(df, "detail_title"),
            get_series(df, "title")
        )
    ]
    out["job_slug_raw"] = get_series(df, "source_field_name")

    out["salary_raw"] = [
        first_non_empty(a, b)
        for a, b in zip(
            get_series(df, "detail_salary"),
            get_series(df, "salary_list")
        )
    ]

    out["location_raw"] = [
        first_non_empty(a, b)
        for a, b in zip(
            get_series(df, "address_list"),
            get_series(df, "detail_location")
        )
    ]

    out["working_addresses_raw"] = get_series(df, "working_addresses")

    out["working_times_raw"] = get_series(df, "working_times")

    out["experience_raw"] = [
        first_non_empty(a, b)
        for a, b in zip(
            get_series(df, "exp_list"),
            get_series(df, "detail_experience")
        )
    ]
    out["education_level_raw"] = get_series(df, "education_level")
    out["employment_type_raw"] = get_series(df, "employment_type")
    out["job_level_raw"] = get_series(df, "job_level")
    out["job_quantity_raw"] = get_series(df, "job_quantity")

    out["description_raw"] = get_series(df, "desc_mota")
    out["requirements_raw"] = get_series(df, "desc_yeucau")
    out["benefits_raw"] = get_series(df, "desc_quyenloi")

    out["tags_raw"] = get_series(df, "tags")

    out["deadline_raw"] = get_series(df, "deadline")

    out["company_name_raw"] = [
        first_non_empty(a, b)
        for a, b in zip(
            get_series(df, "company_name_full"),
            get_series(df, "company_name")
        )
    ]
    out["company_website_raw"] = get_series(df, "company_website")
    out["company_field_raw"] = get_series(df, "company_field_from_job")

    out["company_scale_raw"] = [
        first_non_empty(a, b)
        for a, b in zip(
            get_series(df, "company_scale_from_job"),
            get_series(df, "company_scale")
        )
    ]
    
    out["company_address_raw"] = [
        first_non_empty(a, b)
        for a, b in zip(
            get_series(df, "company_address_from_job"),
            get_series(df, "company_address")
        )
    ]
    out["company_description_raw"] = get_series(df, "company_description")

    return out


df = merge_semantic_columns(df_raw)
print("[INFO] After merging:", df.shape)
display(df.head(3))
df.info()

def detect_language_type(text: str) -> str:
    text = safe_str(text)
    if not text:
        return "empty"

    has_vi = bool(re.search(r"[ăâêôơưđàáạảãèéẹẻẽìíịỉĩòóọỏõùúụủũỳýỵỷỹ]", text.lower()))
    has_en = bool(re.search(r"[a-z]", text.lower()))

    if has_vi and has_en:
        return "mixed"
    if has_vi:
        return "vi"
    if has_en:
        return "en"
    return "other"


audit_rows = []
combined_text = (
    get_series(df, "job_title_raw", "") .fillna("") + " " +
    get_series(df, "description_raw", "") .fillna("") + " " +
    get_series(df, "requirements_raw", "") .fillna("")
)

lang_type = combined_text.apply(detect_language_type)

audit_rows.append({
    "n_rows": len(df),
    "dup_by_url_ratio": df["job_url"].duplicated().mean() if "job_url" in df.columns else np.nan,
    "has_minimum_content_ratio": (
        (df["job_title_raw"].fillna("").str.len() > 0) &
        (
            (df["description_raw"].fillna("").str.len() > 0) |
            (df["requirements_raw"].fillna("").str.len() > 0)
        )
    ).mean(),
    "mixed_ratio": (lang_type == "mixed").mean(),
    "en_ratio": (lang_type == "en").mean(),
    "vi_ratio": (lang_type == "vi").mean(),
})

audit_df = pd.DataFrame(audit_rows)
missing_df = pd.DataFrame({
    "column": df.columns,
    "missing_ratio": [df[c].isna().mean() for c in df.columns]
}).sort_values("missing_ratio", ascending=False)

display(audit_df)
display(missing_df)

df_clean = df.copy()

for col in df_clean.columns:
    df_clean[col] = df_clean[col].apply(normalize_empty_value)

print("df_raw shape :", df.shape)
print("df_clean shape:", df_clean.shape)
display(df_clean.head(3))

df_clean.info()

base_text_cols = [
    "job_title_raw",
    "job_slug_raw",
    "salary_raw",
    "location_raw",
    "working_addresses_raw",
    "working_times_raw",
    "experience_raw",
    "education_level_raw",
    "employment_type_raw",
    "job_level_raw",
    "job_quantity_raw",
    "description_raw",
    "requirements_raw",
    "benefits_raw",
    "tags_raw",
    "deadline_raw",
    "company_name_raw",
    "company_address_raw",
    "company_field_raw",
    "company_scale_raw",
    "company_description_raw",
]

for col in base_text_cols:
    if col in df_clean.columns:
        prefix = col.replace("_raw", "")
        df_clean[f"{prefix}_clean_light"] = df_clean[col].apply(clean_text_light)
        df_clean[f"{prefix}_clean_struct"] = df_clean[col].apply(clean_text_preserve_structure)
        df_clean[f"{prefix}_clean_strict"] = df_clean[col].apply(clean_text_strict)
        df_clean[f"{prefix}_clean_phobert"] = df_clean[col].apply(clean_text_for_phobert)

print("[INFO] Đã tạo xong các cột clean_*")
clean_cols = [c for c in df_clean.columns if "_clean_" in c]
print("Số cột clean:", len(clean_cols))
display(df_clean[[c for c in clean_cols[:12]]].head(2))

preview_cols = [
    "working_addresses_clean_light", "working_addresses_clean_struct", "working_addresses_clean_strict", "working_addresses_clean_phobert",
    "company_address_clean_light", "company_address_clean_struct", "company_address_clean_strict", "company_address_clean_phobert"
    
]
preview_cols = [c for c in preview_cols if c in df_clean.columns]

display(df_clean[preview_cols].head(3))

empty_ratio_df = pd.DataFrame({
    "column": preview_cols,
    "empty_ratio": [(df_clean[c].fillna("").str.strip() == "").mean() for c in preview_cols]
})
display(empty_ratio_df)


TITLE_TRAILING_NOISE_PATTERNS = [
    r"\|\s*.*$",
    r"[_-]+\s*.*$",

    # Salary / compensation tails
    r"\-\s*luong.*$",
    r"\-\s*thu nhap.*$",
    r"\-\s*salary.*$",
    r"\-\s*upto.*$",
    r"\-\s*up to.*$",
    r"\-\s*benefit.*$",
    r"\-\s*phuc loi.*$",
    r"\(\s*up to\s*\d+.*\)",

    # Location tails frequently seen in raw titles
    r"\-\s*ha noi.*$",
    r"\-\s*hn.*$",
    r"\-\s*ho chi minh.*$",
    r"\-\s*tp\.?\s*hcm.*$",
    r"\-\s*hcm.*$",
    r"\-\s*da nang.*$",
    r"\-\s*hai phong.*$",
    r"\-\s*tai ha noi.*$",
    r"\-\s*tai ho chi minh.*$",
    r"\-\s*lam viec tai.*$",

    # Work mode / urgency tails
    r"\-\s*di lam ngay.*$",
    r"\-\s*ob som.*$",
    r"\-\s*remote.*$",
    r"\-\s*hybrid.*$",
    r"\-\s*onsite.*$",
    r"\(\s*remote\s*\)",
    r"\(\s*hybrid\s*\)",
    r"\(\s*onsite\s*\)",
    r"\(\s*urgent\s*\)",
    r"\(\s*hot\s*\)",

    # Internal code tails from raw dataset
    r"\-\s*ta\d+.*$",
    r"\-\s*ho\d{2}\.\d+.*$",
    r"\-\s*holt\.\d+.*$",

    # Experience tails
    r"\-\s*tu\s*\d+\s*nam.*$",
    r"\-\s*\d+\s*nam kinh nghiem.*$",
    r"\-\s*khong yeu cau kinh nghiem.*$",

    # Noisy bracket tails observed in title/detail_title
    r"\(\s*junior\s*\)",
    r"\(\s*middle\s*\)",
    r"\(\s*senior\s*\)",
    r"\(\s*leader\s*\)",
    r"\(\s*fresher\s*\)",
    r"\(\s*fulltime\s*\)",
    r"\(\s*m/f/d\s*\)",
]

BRACKET_NOISE_KEYWORDS = {
    # Major cities / common location aliases
    "ha noi", "hn",
    "ho chi minh", "hcm", "tp hcm", "tphcm", "sai gon",
    "da nang", "hai phong", "can tho",

    # Work mode / urgency
    "remote", "hybrid", "onsite", "urgent", "hot", "di lam ngay", "ob som",

    # Internal posting codes
    "ta105", "ta150", "ta171", "ta172", "ta174", "ta188",
    "ho26.74", "ho26.76", "ho26.77", "ho26.78", "ho26.79", "ho26.84", "ho26.85",
    "holt.07", "holt.08", "holt.13", "holt.14", "holt.15", "holt.16", "holt.17",

    # Non-role campaign/domain noise often found in brackets
    "banking", "fintech", "game industry", "collection analytics", "domain erp", "domain edtech",
    "junior", "middle", "senior", "leader", "fresher", "fulltime", "m/f/d",
}

TITLE_SYNONYM_MAP = {
    "chuyên viên phân tích dữ liệu": "data analyst",
    "nhân viên phân tích dữ liệu": "data analyst",
    "cvcc phân tích dữ liệu": "data analyst",
    "cvcc khoa học dữ liệu": "data scientist",
    "chuyên viên dữ liệu": "data specialist",
    "chuyên gia dữ liệu": "data specialist",
    "chuyên viên kết nối phân tích dữ liệu": "data analyst",
    "chuyên viên xử lý dữ liệu": "data processing specialist",
    "nhân viên xử lý dữ liệu": "data processing specialist",
    "data analysis executive": "data analyst",
    "business data analyst": "business intelligence analyst",
    "business data analyst lead": "business intelligence analyst lead",
    "assistant manager data analyst": "data analyst manager",
    "retail data analyst supervisor": "data analyst supervisor",
    "data analyst workforce management": "workforce data analyst",
    "data analyst consultant": "data analyst",
    "data analyst teamleader": "data analyst lead",
    "trưởng nhóm data analyst": "data analyst lead",
    "trưởng phòng đào tạo data analyst": "data analyst manager",
    "senior data science & analysis": "senior data scientist",
    "hr data analysis": "hr data analyst",
    "ecommerce business data analyst": "business intelligence analyst",
    "phân tích dữ liệu kinh doanh": "business data analyst",

    "fp&a analyst": "fp&a analyst",
    "finance planning & analysis associate": "fp&a analyst",
    "junior fp&a analyst": "fp&a analyst",
    "credit risk analytics and modelling expert": "risk analytics expert",
    "expert, fraud risk data analytics and portfolio management": "fraud risk analytics expert",
    "chuyên viên dữ liệu tài chính": "financial data analyst",
    "chuyên viên phân tích và quản lý dữ liệu tài chính": "financial data analyst",

    "bi analyst": "business intelligence analyst",
    "business analyst": "business analyst",
    "chuyên viên business analyst": "business analyst",
    "chuyên viên phân tích nghiệp vụ": "business analyst",
    "chuyên viên phân tích hệ thống & nghiệp vụ": "business analyst",
    "chuyên viên phân tích nghiệp vụ mảng data": "data business analyst",
    "nhân viên phân tích dữ liệu ( business analyst )": "business analyst",
    "business analyst data": "data business analyst",
    "project manager dự án ai hub": "ai project manager",
    "quản lý dự án erp": "erp project manager",
    "chuyên viên quản trị dự án data warehouse": "data warehouse project manager",
    "project assistant": "project coordinator",
    "project assistant intern": "project coordinator intern",
    "project assistant - non-tech": "project coordinator",
    "trưởng nhóm phân tích kinh doanh": "business analysis lead",

    "data engineer": "data engineer",
    "chuyên viên dữ liệu ( data engineer )": "data engineer",
    "nhân viên data engineer": "data engineer",
    "kỹ sư dữ liệu": "data engineer",
    "kỹ sư dữ liệu lớn": "big data engineer",
    "big data admin": "big data engineer",
    "aws data engineer": "data engineer",
    "data engineer aws": "data engineer",
    "junior data integration engineer": "data integration engineer",
    "data platform engineer": "data platform engineer",
    "data platform operation": "data platform engineer",
    "database engineer": "database engineer",
    "database engineer ( dba )": "database administrator",
    "quản trị cơ sở dữ liệu": "database administrator",
    "database developer": "database developer",
    "reporting engineer power bi microsoft fabric": "bi engineer",
    "fresher data warehouse": "data warehouse engineer",
    "data management": "data management specialist",

    "data scientist": "data scientist",
    "nhà khoa học dữ liệu": "data scientist",
    "chuyên viên phát triển khoa học dữ liệu": "data scientist",
    "chuyên viên mô hình và phân tích nâng cao": "data scientist",
    "chuyên viên cao cấp mô hình hóa và phân tích nâng cao": "senior data scientist",
    "chuyên viên, chuyên viên cao cấp khoa học dữ liệu": "data scientist",
    "ml engineer": "machine learning engineer",
    "machine learning engineer": "machine learning engineer",
    "ai engineer": "ai engineer",
    "kỹ sư ai": "ai engineer",
    "kỹ sư trí tuệ nhân tạo": "ai engineer",
    "chuyên viên trí tuệ nhân tạo": "ai engineer",
    "chuyên viên trí tuệ nhân tạo (ai)": "ai engineer",
    "chuyên viên ai": "ai engineer",
    "lập trình viên ai": "ai engineer",
    "nhân viên ai system engineer": "ai system engineer",
    "ai developer": "ai engineer",
    "ai generative engineer": "generative ai engineer",
    "ai engineering": "ai engineer",
    "ai - machine learning engineer": "machine learning engineer",
    "ai platform architect": "ai architect",
    "ai platform engineer": "ai platform engineer",
    "ai system engineer": "ai system engineer",
    "ai automation manager": "ai automation manager",
    "ai and automation intern": "ai automation intern",
    "product & ai automation intern": "ai automation intern",
    "intern ai solution engineer": "ai solutions engineer intern",
    "senior ai expert": "senior ai expert",
    "ai expert": "ai expert",
    "senior agentic ai expert": "senior ai expert",
    "nlp research engineer": "nlp engineer",
    "senior nlp engineer": "senior nlp engineer",
    "senior ai engineer / nlp engineer": "senior nlp engineer",
    "machine vision engineer": "computer vision engineer",
    "fresher ai computer vision engineer": "computer vision engineer",
    "senior ai research engineer": "ai research engineer",
    "ai research intern": "ai research intern",
    "ai quantitative researcher intern": "ai research intern",
    "software engineer (prompt engineering)": "prompt engineer",
    "ai artist": "ai creative specialist",
    "trưởng nhóm ai engineer": "ai engineer lead",
    "trưởng nhóm ai": "ai lead",
    "giám đốc trí tuệ nhân tạo (ai)": "ai director",
    "trưởng phòng công nghệ thị giác máy tính": "head of computer vision",
    "fresher ai (nlp)": "nlp engineer",
    "fresher ai": "ai engineer",

    "data governance specialist": "data governance specialist",
    "data quality analyst": "data quality analyst",
    "data labeling specialist": "data labeling specialist",
    "thực tập sinh xử lý dữ liệu": "data processing intern",
    "điều phối dự án label data tiếng anh": "data labeling coordinator",
    "nhân viên gán nhãn dữ liệu tiếng nhật": "data labeling specialist",
    "nhân viên dán nhãn - tiếng hàn": "data labeling specialist",
    "nhân viên nhập liệu - xử lý dữ liệu": "data entry specialist",
    "nhân viên nhập và xử lý dữ liệu tiếng nhật": "data processing specialist",
    "nhân viên ngôn ngữ dữ liệu tiếng anh": "language data specialist",
    "nhân viên ngôn ngữ dữ liệu tiếng trung": "language data specialist",
    "nhân viên ngôn ngữ dữ liệu - tiếng pháp": "language data specialist",
    "nhân viên ngôn ngữ dữ liệu - tiếng tây ban nha /bồ đào nha/thái": "language data specialist",

    "on job training ai": "ai trainee",
    "mb trainee - ai engineer": "ai trainee",
    "thực tập sinh ai": "ai intern",
    "ai engineer intern": "ai intern",
    "thực tập sinh backend ai": "ai backend intern",
    "thực tập sinh computer vision ai": "computer vision intern",
    "thực tập sinh nghiên cứu & ứng dụng ai": "ai research intern",
    "thực tập sinh nội dung ai": "ai content intern",
    "data analyst intern": "data analyst intern",
    "intern data analyst": "data analyst intern",
    "thực tập sinh data analyst": "data analyst intern",
    "data science intern": "data science intern",
    "data engineer intern": "data engineer intern",
    "thực tập sinh data engineer": "data engineer intern",
    "fresher data engineer": "data engineer",
    "fresher data analytics engineer": "analytics engineer",
    "fresher data warehouse engineer": "data warehouse engineer",

    "aiops specialist": "aiops specialist",
    "presale data and analytics": "data analytics presales",
    "power bi leader": "bi lead",
    "giảng viên power bi": "power bi trainer",
    "chuyên viên power bi,tableau": "bi analyst",
    "chuyên viên cao cấp kiến trúc giải pháp dữ liệu": "data architect",
    "data architect": "data architect",
    "gsd engineer - garment standard data engineer - kỹ sư hệ thống dữ liệu chuẩn": "data engineer",
    "tự động hoá dữ liệu": "data automation specialist",
    
    # Software / backend / frontend / fullstack
    "software engineer": "software engineer",
    "backend developer": "backend developer",
    "frontend developer": "frontend developer",
    "fullstack developer": "fullstack developer",
    "java developer": "backend developer",
    "java engineer": "backend developer",
    "python developer": "backend developer",
    ".net developer": "backend developer",
    ". net developer": "backend developer",
    "lap trinh vien .net": "backend developer",
    "react native developer": "frontend developer",
    "angular developer": "frontend developer",

    # Cloud / DevOps
    "devops engineer": "devops engineer",
    "senior devops engineer": "devops engineer",
    "devsecops engineer": "devops engineer",
    "cloud engineer": "cloud engineer",
    "cloud engineer | devops engineer": "cloud devops engineer",

    # QA / Testing
    "automation tester": "automation tester",
    "performance tester": "automation tester",
    "tester": "automation tester",
    "nhan vien tester": "automation tester",

    # DBA
    "database administrator": "database administrator",
    "database administrator (dba)": "database administrator",
    "dba": "database administrator",

    # IoT / Embedded
    "ky su iot (iot engineer)": "iot engineer",
    "iot engineer": "iot engineer",
    "embedded engineer/lap trinh nhung": "embedded engineer",
    "embedded engineer": "embedded engineer",

    # Product / BA
    "product owner": "product owner",
    "product manager": "product manager",
    "product owner/product manager": "product manager",
    "product analyst/research": "product analyst",
    "it business analyst": "business analyst",
    "business analyst leader": "business analysis lead",
    "senior business analyst": "business analyst",
}


JOB_FAMILY_RULES = {
    "data_analytics": [
        "data analyst", "business intelligence analyst", "analytics engineer", "product analyst",
        "fp&a analyst", "business data analyst", "workforce data analyst", "financial data analyst",
        "risk analytics", "fraud risk", "hr data analyst", "bi analyst", "bi engineer", "reporting analyst",
        "power bi", "tableau",
    ],
    "data_engineering": [
        "data engineer", "etl developer", "big data engineer", "data integration engineer",
        "data warehouse engineer", "data platform engineer", "database engineer", "database administrator",
        "database developer", "data architect", "kafka", "spark", "hadoop", "fabric",
    ],
    "data_science_ml": [
        "data scientist", "machine learning engineer", "ml engineer", "ai engineer", "nlp engineer",
        "computer vision engineer", "ai research engineer", "generative ai engineer", "prompt engineer",
        "ai architect", "ai platform engineer", "ai system engineer", "aiops specialist", "ai expert",
    ],
    "software_engineering": [
        # Explicit roles from scope
        "software engineer", "backend developer", "frontend developer", "fullstack developer",
        # Common variants
        "backend engineer", "frontend engineer", "fullstack engineer", "java developer",
        "python developer", ".net developer", "nodejs", "react", "angular", "react native",
    ],
    "cloud_devops_sre": [
        # Explicit roles from scope
        "cloud engineer", "devops engineer",
        # Common variants
        "devsecops", "site reliability", "sre",
        "platform engineer", "kubernetes", "terraform", "ci/cd",
    ],
    "qa_testing": [
        # Explicit roles from scope
        "automation tester",
        # Common variants
        "performance tester", "qa engineer", "test engineer", "tester",
        "selenium", "cypress", "playwright",
    ],
    "iot_embedded": [
        # Explicit roles from scope
        "iot engineer", "ky su iot", "ky su iot (iot engineer)",
        "embedded engineer", "embedded engineer/lap trinh nhung", "lap trinh nhung",
        # Common variants
        "firmware engineer", "embedded",
        "electronics", "microcontroller",
    ],
    "product_project_ba": [
        # Explicit roles from scope
        "product analyst", "product analyst/research",
        "business analyst", "business analyst (phan tich nghiep vu)",
        "product owner", "product manager", "product owner/product manager",
        # Common variants
        "data business analyst", "project manager", "project coordinator",
        "scrum master", "business analysis lead",
    ],
    "database_platform": [
        # Explicit role from scope
        "database administrator", "database administrator (dba)", "dba",
        # Adjacent variants
        "database engineer", "database developer", "sql server dba",
    ],
    "data_governance_quality": [
        "data governance", "data quality", "data steward", "data management specialist",
        "data labeling", "data processing", "data entry", "language data specialist",
    ],
    "operations_support": [
        "operation", "coordinator", "assistant", "support", "back office",
    ],
}

JOB_FAMILY_DESCRIPTION_HINTS = {
    "data_analytics": [
        "dashboard", "report", "business intelligence", "power bi", "tableau", "sql", "kpi", "insight",
        "phan tich du lieu",
    ],
    "data_engineering": [
        "etl", "pipeline", "data pipeline", "data warehouse", "dwh", "airflow", "spark",
        "kafka", "hadoop", "lakehouse", "fabric", "batch", "streaming",
    ],
    "data_science_ml": [
        "machine learning", "deep learning", "nlp", "llm", "rag", "computer vision", "model training",
        "tensorflow", "pytorch", "generative ai", "genai", "feature engineering",
    ],
    "software_engineering": [
        # Explicit role hints
        "software engineer", "backend developer", "frontend developer", "fullstack developer",
        # Stack/activity hints
        "backend", "frontend", "fullstack", "api", "microservice", "rest", "graphql",
        "react", "nodejs", "java", ".net", "web development",
    ],
    "cloud_devops_sre": [
        # Explicit role hints
        "cloud engineer", "devops engineer",
        # Stack/activity hints
        "cloud", "aws", "azure", "gcp", "kubernetes", "docker", "terraform", "ci/cd",
        "monitoring", "observability", "infrastructure as code",
    ],
    "qa_testing": [
        # Explicit role hints
        "automation tester",
        # Stack/activity hints
        "automation test", "test automation", "qa", "quality assurance", "selenium", "cypress",
        "playwright", "regression test", "performance test",
    ],
    "iot_embedded": [
        # Explicit role hints
        "iot engineer", "embedded engineer", "lap trinh nhung",
        # Stack/activity hints
        "iot", "embedded", "firmware", "microcontroller", "sensor", "mqtt", "rtos", "hardware",
    ],
    "product_project_ba": [
        # Explicit role hints
        "product analyst", "product analyst/research", "business analyst", "phan tich nghiep vu",
        "product owner", "product manager", "product owner/product manager",
        # Process/activity hints
        "business analyst", "phan tich nghiep vu", "user story", "brd", "frd", "stakeholder",
        "scrum", "agile", "project management", "product roadmap", "backlog", "prioritization",
    ],
    "database_platform": [
        # Explicit role hints
        "database administrator", "dba",
        # Stack/activity hints
        "database", "sql server", "postgresql", "mysql", "query tuning", "index", "backup", "replication",
    ],
    "data_governance_quality": [
        "data governance", "data quality", "master data", "data steward", "label", "labeling",
        "gan nhan", "data validation", "metadata", "lineage",
    ],
    "operations_support": [
        "van hanh", "operation", "dieu phoi", "coordinator", "assistant", "ho tro", "nhap lieu",
        "back office",
    ],
}

def strip_bracket_noise(text):
    text = safe_str(text)
    if not text:
        return ""

    matches_round = re.findall(r"\((.*?)\)", text)
    for m in matches_round:
        normalized = clean_text_strict(m)
        if normalized in BRACKET_NOISE_KEYWORDS:
            text = text.replace(f"({m})", " ")

    matches_square = re.findall(r"\[(.*?)\]", text)
    for m in matches_square:
        normalized = clean_text_strict(m)
        if normalized in BRACKET_NOISE_KEYWORDS:
            text = text.replace(f"[{m}]", " ")

    return re.sub(r"\s+", " ", text).strip()

def normalize_job_title(text):
    text = clean_text_light(text)
    if not text:
        return ""

    text = strip_bracket_noise(text)

    text = re.sub(r"\b(?:TA\d+|HO\d{2}\.\d+|HOLT\.\d+)\b", " ", text, flags=re.I)

    for pat in TITLE_TRAILING_NOISE_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.I)

    text = re.sub(r"\s+", " ", text).strip(" -|_")
    lowered = text.lower().strip()

    if lowered in TITLE_SYNONYM_MAP:
        return TITLE_SYNONYM_MAP[lowered]

    return lowered

def infer_job_family_from_title(job_title_canonical):
    t = safe_str(job_title_canonical).lower()
    for family, keywords in JOB_FAMILY_RULES.items():
        if any(k in t for k in keywords):
            return family
    return "unknown"

def infer_job_family_from_description(text):
    t = clean_text_strict(text)
    if not t:
        return "unknown"

    scores = {}
    for family, keywords in JOB_FAMILY_DESCRIPTION_HINTS.items():
        score = 0
        for kw in keywords:
            if kw in t:
                score += 1
        scores[family] = score

    best_family = max(scores, key=scores.get)
    best_score = scores[best_family]

    return best_family if best_score > 0 else "unknown"

def resolve_job_family(title_family, desc_family):
    title_family = safe_str(title_family) or "unknown"
    desc_family = safe_str(desc_family) or "unknown"

    if title_family != "unknown" and desc_family == "unknown":
        return title_family, "title"

    if title_family == "unknown" and desc_family != "unknown":
        return desc_family, "description"

    if title_family != "unknown" and desc_family != "unknown":
        if title_family == desc_family:
            return title_family, "title+description"
        return title_family, "title_priority"

    return "unknown", "unknown"

df_clean["job_title_display"] = df_clean["job_title_clean_light"].fillna("")
df_clean["job_title_canonical"] = df_clean["job_title_raw"].apply(normalize_job_title)

df_clean["job_family_from_title"] = df_clean["job_title_canonical"].apply(infer_job_family_from_title)
df_clean["job_family_from_description"] = df_clean["description_clean_strict"].apply(infer_job_family_from_description)

family_resolved = [
    resolve_job_family(a, b)
    for a, b in zip(df_clean["job_family_from_title"], df_clean["job_family_from_description"])
]
df_clean["job_family"] = [x[0] for x in family_resolved]
df_clean["job_family_source"] = [x[1] for x in family_resolved]

df_clean["job_title_for_phobert"] = df_clean["job_title_display"].where(
    df_clean["job_title_display"].fillna("").str.strip() != "",
    df_clean["job_title_canonical"]
).apply(clean_text_for_phobert)

display(
    df_clean[
        [
            "job_title_raw",
            "job_title_display",
            "job_title_canonical",
            "job_family_from_title",
            "job_family_from_description",
            "job_family",
            "job_family_source",
            "job_title_for_phobert",
        ]
    ].head(10)
)

VIETNAM_CITY_ALIASES = {
    "cao bằng": "Cao Bằng",
    "cao bang": "Cao Bằng",
    "sơn la": "Sơn La",
    "son la": "Sơn La",
    "lai châu": "Lai Châu",
    "lai chau": "Lai Châu",
    "lạng sơn": "Lạng Sơn",
    "lang son": "Lạng Sơn",
    "tuyên quang": "Tuyên Quang",
    "tuyen quang": "Tuyên Quang",
    "lào cai": "Lào Cai",
    "lao cai": "Lào Cai",
    "thái nguyên": "Thái Nguyên",
    "thai nguyen": "Thái Nguyên",
    "điện biên": "Điện Biên",
    "dien bien": "Điện Biên",
    "phú thọ": "Phú Thọ",
    "phu tho": "Phú Thọ",
    "bắc ninh": "Bắc Ninh",
    "bac ninh": "Bắc Ninh",
    "hà nội": "TP. Hà Nội",
    "ha noi": "TP. Hà Nội",
    "quảng ninh": "Quảng Ninh",
    "quang ninh": "Quảng Ninh",
    "hải phòng": "TP. Hải Phòng",
    "hai phong": "TP. Hải Phòng",
    "hưng yên": "Hưng Yên",
    "hung yen": "Hưng Yên",
    "ninh bình": "Ninh Bình",
    "ninh binh": "Ninh Bình",
    "thanh hóa": "Thanh Hóa",
    "thanh hoa": "Thanh Hóa",
    "nghệ an": "Nghệ An",
    "nghe an": "Nghệ An",
    "hà tĩnh": "Hà Tĩnh",
    "ha tinh": "Hà Tĩnh",
    "quảng trị": "Quảng Trị",
    "quang tri": "Quảng Trị",
    "huế": "TP. Huế",
    "hue": "TP. Huế",
    "đà nẵng": "TP. Đà Nẵng",
    "da nang": "TP. Đà Nẵng",
    "quảng ngãi": "Quảng Ngãi",
    "quang ngai": "Quảng Ngãi",
    "gia lai": "Gia Lai",
    "đắk lắk": "Đắk Lắk",
    "dak lak": "Đắk Lắk",
    "khánh hòa": "Khánh Hòa",
    "khanh hoa": "Khánh Hòa",
    "lâm đồng": "Lâm Đồng",
    "lam dong": "Lâm Đồng",
    "đồng nai": "Đồng Nai",
    "dong nai": "Đồng Nai",
    "hồ chí minh": "TP. Hồ Chí Minh",
    "ho chi minh": "TP. Hồ Chí Minh",
    "tây ninh": "Tây Ninh",
    "tay ninh": "Tây Ninh",
    "đồng tháp": "Đồng Tháp",
    "dong thap": "Đồng Tháp",
    "vĩnh long": "Vĩnh Long",
    "vinh long": "Vĩnh Long",
    "cần thơ": "TP. Cần Thơ",
    "can tho": "TP. Cần Thơ",
    "an giang": "An Giang",
    "cà mau": "Cà Mau",
    "ca mau": "Cà Mau",
}

WORK_MODE_RULES = {
    "remote": [r"\bremote\b", r"làm việc từ xa", r"work from home", r"\bwfh\b"],
    "hybrid": [r"\bhybrid\b", r"linh hoạt", r"kết hợp onsite và remote"],
    "onsite": [r"\bonsite\b", r"tại văn phòng", r"làm việc tại công ty"],
}


def detect_city_from_text(text):
    t = clean_text_strict(text)
    for alias, canonical in VIETNAM_CITY_ALIASES.items():
        if alias in t:
            return canonical
    return None


def infer_work_mode(*texts):
    merged = " ".join([clean_text_strict(t) for t in texts if safe_str(t)])
    if not merged:
        return "unknown"
    for mode, patterns in WORK_MODE_RULES.items():
        for p in patterns:
            if re.search(p, merged, flags=re.I):
                return mode
    return "unknown"


def has_multi_location(text):
    t = clean_text_strict(text)
    hits = set()
    for alias, canonical in VIETNAM_CITY_ALIASES.items():
        if alias in t:
            hits.add(canonical)
    return len(hits) >= 2


def parse_working_address(raw_text):
    text = clean_text_light(raw_text)
    city = detect_city_from_text(text)
    district = None
    m = re.search(r"quận\s+(\d+|[a-zà-ỹ]+)", text, flags=re.I)
    if m:
        district = m.group(0).strip()

    return {
        "working_address_clean": text,
        "location_city": city,
        "location_district": district,
        "is_multi_location": has_multi_location(text)
    }


def normalize_location(location_raw, working_addresses_raw):
    city_1 = detect_city_from_text(location_raw)
    city_2 = detect_city_from_text(working_addresses_raw)
    return first_non_empty(city_1, city_2, clean_text_light(location_raw), clean_text_light(working_addresses_raw))


def parse_deadline(raw, reference_date=None):
    reference_date = reference_date or datetime.today().date()
    text = clean_text_light(raw)

    if not text:
        return {
            "deadline_clean": "",
            "deadline_date": None,
            "days_to_deadline": None,
            "deadline_type": "unknown",
            "is_expired": None,
        }

    m = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", text)
    if m:
        day, month, year = map(int, m.groups())
        if year < 100:
            year += 2000
        try:
            dt = datetime(year, month, day).date()
            return {
                "deadline_clean": text,
                "deadline_date": str(dt),
                "days_to_deadline": (dt - reference_date).days,
                "deadline_type": "absolute_date",
                "is_expired": dt < reference_date,
            }
        except Exception:
            pass

    m = re.search(r"(\d+)\s*ngày", clean_text_strict(text))
    if m:
        days = int(m.group(1))
        dt = reference_date + timedelta(days=days)
        return {
            "deadline_clean": text,
            "deadline_date": str(dt),
            "days_to_deadline": days,
            "deadline_type": "relative_days",
            "is_expired": False,
        }

    return {
        "deadline_clean": text,
        "deadline_date": None,
        "days_to_deadline": None,
        "deadline_type": "unknown",
        "is_expired": None,
    }

address_parsed = df_clean["working_addresses_raw"].apply(parse_working_address)
address_df = pd.DataFrame(address_parsed.tolist(), index=df_clean.index)

for c in address_df.columns:
    df_clean[c] = address_df[c]

df_clean["location_norm"] = [
    normalize_location(a, b)
    for a, b in zip(df_clean["location_raw"], df_clean["working_addresses_raw"])
]

df_clean["work_mode"] = [
    infer_work_mode(a, b, c, d)
    for a, b, c, d in zip(
        df_clean["job_title_raw"],
        df_clean["location_raw"],
        df_clean["working_addresses_raw"],
        df_clean["description_raw"]
    )
]

deadline_parsed = df_clean["deadline_raw"].apply(parse_deadline)
deadline_df = pd.DataFrame(deadline_parsed.tolist(), index=df_clean.index)
for c in deadline_df.columns:
    df_clean[c] = deadline_df[c]

display(df_clean[
    [
        "location_raw", "working_addresses_raw", "location_city",
        "location_district", "location_norm", "work_mode",
        "deadline_raw", "deadline_date", "days_to_deadline", "deadline_type", "is_expired"
    ]
].head(10))

def clean_salary(raw):
    text = clean_text_light(raw).lower()
    text = text.replace("thoả", "thỏa")
    return text


def parse_numeric_token(num_text):
    s = safe_str(num_text).strip().replace(" ", "")
    if not s:
        return None

    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif s.count(".") > 1:
        s = s.replace(".", "")
    elif s.count(",") > 1:
        s = s.replace(",", "")
    else:
        s = s.replace(",", ".")

    try:
        return float(s)
    except Exception:
        return None


def detect_salary_multiplier(text, currency):
    t = text.lower()
    if currency == "usd":
        return 1.0
    if "triệu" in t or "trieu" in t:
        return 1_000_000
    if "nghìn" in t or "ngan" in t or "k" in t:
        return 1_000
    return 1.0


def parse_salary_range(raw):
    text = clean_salary(raw)
    if not text:
        return {
            "salary_clean": "",
            "salary_min_value": None,
            "salary_max_value": None,
            "salary_currency": "unknown",
            "salary_period": "month",
            "salary_type": "unknown",
            "salary_is_negotiable": None,
            "salary_min_vnd_month": None,
            "salary_max_vnd_month": None,
        }

    currency = "usd" if "usd" in text or "$" in text else "vnd"
    negotiable = ("thỏa thuận" in text) or ("negotiable" in text)

    nums = re.findall(r"\d+(?:[.,]\d+)?", text)
    nums = [parse_numeric_token(x) for x in nums]
    nums = [x for x in nums if x is not None]

    multiplier = detect_salary_multiplier(text, currency)

    salary_min = None
    salary_max = None
    salary_type = "unknown"

    if negotiable and not nums:
        salary_type = "negotiable"
    elif "up to" in text or "tối đa" in text:
        if nums:
            salary_max = nums[0] * multiplier
            salary_type = "upper_bound"
    elif len(nums) >= 2:
        salary_min = nums[0] * multiplier
        salary_max = nums[1] * multiplier
        salary_type = "range"
    elif len(nums) == 1:
        salary_min = nums[0] * multiplier
        salary_max = nums[0] * multiplier
        salary_type = "fixed"

    if currency == "usd":
        fx = 25000
        if salary_min is not None:
            salary_min = salary_min * fx
        if salary_max is not None:
            salary_max = salary_max * fx

    return {
        "salary_clean": text,
        "salary_min_value": salary_min,
        "salary_max_value": salary_max,
        "salary_currency": currency,
        "salary_period": "month",
        "salary_type": salary_type,
        "salary_is_negotiable": negotiable,
        "salary_min_vnd_month": salary_min,
        "salary_max_vnd_month": salary_max,
    }


salary_parsed = df_clean["salary_raw"].apply(parse_salary_range)
salary_df = pd.DataFrame(salary_parsed.tolist(), index=df_clean.index)
for c in salary_df.columns:
    df_clean[c] = salary_df[c]

display(df_clean[
    [
        "salary_raw", "salary_clean", "salary_min_value", "salary_max_value",
        "salary_currency", "salary_type", "salary_is_negotiable",
        "salary_min_vnd_month", "salary_max_vnd_month"
    ]
].head(10))

def clean_experience(text):
    return clean_text_strict(text)


def parse_experience_range(raw):
    text = clean_experience(raw)
    if not text:
        return {
            "experience_clean": "",
            "experience_min_years": None,
            "experience_max_years": None,
            "experience_type": "unknown",
        }

    if "không yêu cầu" in text or "no experience" in text:
        return {
            "experience_clean": text,
            "experience_min_years": 0.0,
            "experience_max_years": 0.0,
            "experience_type": "no_experience",
        }

    m_month = re.search(r"(\d+(?:\.\d+)?)\s*tháng", text)
    if m_month:
        months = float(m_month.group(1))
        years = months / 12.0
        return {
            "experience_clean": text,
            "experience_min_years": years,
            "experience_max_years": years,
            "experience_type": "fixed",
        }

    nums = re.findall(r"\d+(?:\.\d+)?", text)
    nums = [float(x) for x in nums] if nums else []

    if len(nums) >= 2:
        return {
            "experience_clean": text,
            "experience_min_years": nums[0],
            "experience_max_years": nums[1],
            "experience_type": "range",
        }

    if len(nums) == 1:
        val = nums[0]
        if "từ" in text or "+" in text or "trên" in text or "ít nhất" in text:
            return {
                "experience_clean": text,
                "experience_min_years": val,
                "experience_max_years": None,
                "experience_type": "minimum",
            }
        if "dưới" in text:
            return {
                "experience_clean": text,
                "experience_min_years": 0.0,
                "experience_max_years": val,
                "experience_type": "maximum",
            }
        return {
            "experience_clean": text,
            "experience_min_years": val,
            "experience_max_years": val,
            "experience_type": "fixed",
        }

    return {
        "experience_clean": text,
        "experience_min_years": None,
        "experience_max_years": None,
        "experience_type": "unknown",
    }


EDUCATION_MAP = {
    "phd": ["tiến sĩ", "phd", "doctor"],
    "master": ["thạc sĩ", "master"],
    "bachelor": ["đại học", "cử nhân", "bachelor"],
    "college": ["cao đẳng", "college"],
    "high_school": ["trung học", "high school"],
}

EMPLOYMENT_TYPE_MAP = {
    "full_time": ["toàn thời gian", "full time", "full-time"],
    "part_time": ["bán thời gian", "part time", "part-time"],
    "internship": ["thực tập", "internship", "intern"],
    "contract": ["hợp đồng", "contract"],
    "freelance": ["freelance", "cộng tác viên"],
    "temporary": ["temporary", "thời vụ"],
}


def normalize_education_level(text):
    t = clean_text_strict(text)
    for level, kws in EDUCATION_MAP.items():
        if any(k in t for k in kws):
            return level
    return "unknown"


def normalize_employment_type(text):
    t = clean_text_strict(text)
    for level, kws in EMPLOYMENT_TYPE_MAP.items():
        if any(k in t for k in kws):
            return level
    return "unknown"


JOB_LEVEL_RULES = {
    "director": [r"\bdirector\b", r"\bhead\b", r"giám đốc", r"trưởng phòng"],
    "manager": [r"\bmanager\b", r"quản lý", r"trưởng bộ phận"],
    "lead": [r"\blead\b", r"team lead", r"leader", r"trưởng nhóm"],
    "senior": [r"\bsenior\b", r"\bsr\b", r"cao cấp"],
    "middle": [r"\bmiddle\b", r"\bmid\b"],
    "junior": [r"\bjunior\b", r"\bjr\b"],
    "fresher": [r"\bfresher\b", r"trainee", r"tập sự"],
    "intern": [r"\bintern\b", r"thực tập"],
}


def normalize_job_level(text):
    t = clean_text_strict(text)
    if not t:
        return "unknown"

    for lvl, patterns in JOB_LEVEL_RULES.items():
        for p in patterns:
            if re.search(p, t, flags=re.I):
                return lvl
    return "unknown"


exp_parsed = df_clean["experience_raw"].apply(parse_experience_range)
exp_df = pd.DataFrame(exp_parsed.tolist(), index=df_clean.index)
for c in exp_df.columns:
    df_clean[c] = exp_df[c]

df_clean["education_level_norm"] = df_clean["education_level_raw"].apply(normalize_education_level)
df_clean["employment_type_norm"] = df_clean["employment_type_raw"].apply(normalize_employment_type)
df_clean["job_level_norm"] = df_clean["job_level_raw"].apply(normalize_job_level)

df_clean["seniority_final"] = df_clean["job_level_norm"].fillna("unknown")
df_clean["seniority_source"] = np.where(
    df_clean["seniority_final"].fillna("unknown") != "unknown",
    "job_level_raw",
    "unknown"
)

display(df_clean[
    [
        "experience_raw", "experience_min_years", "experience_max_years", "experience_type",
        "education_level_raw", "education_level_norm",
        "employment_type_raw", "employment_type_norm",
        "job_level_raw", "job_level_norm",
        "seniority_final", "seniority_source"
    ]
].head(10))

quality_summary = pd.DataFrame({
    "metric": [
        "job_family_unknown_rate",
        "job_family_from_title_unknown_rate",
        "job_family_from_description_unknown_rate",
        "seniority_final_unknown_rate",
    ],
    "value": [
        (df_clean["job_family"] == "unknown").mean(),
        (df_clean["job_family_from_title"] == "unknown").mean(),
        (df_clean["job_family_from_description"] == "unknown").mean(),
        (df_clean["seniority_final"] == "unknown").mean(),
    ]
})

display(quality_summary)

display(
    df_clean[
        [
            "job_title_raw",
            "job_title_canonical",
            "description_raw",
            "job_family_from_title",
            "job_family_from_description",
            "job_family",
            "job_family_source",
            "job_level_raw",
            "job_level_norm",
            "seniority_final",
        ]
    ].sample(min(10, len(df_clean)), random_state=42)
)

def normalize_tags(text):
    text = clean_text_light(text)
    if not text:
        return []
    parts = re.split(r"[,\|;/\n]+", text)
    parts = [clean_text_strict(p) for p in parts]
    parts = [p for p in parts if p]
    return deduplicate_list(parts)


df_clean["tags_list"] = df_clean["tags_raw"].apply(normalize_tags)
df_clean["tags_text_phobert"] = df_clean["tags_list"].apply(
    lambda xs: ", ".join(xs) if isinstance(xs, list) else ""
).apply(clean_text_for_phobert)

display(df_clean[["tags_raw", "tags_list", "tags_text_phobert"]].head(10))

SKILL_TAXONOMY = {
    "python": {"aliases": ["python"], "group": "programming"},
    "sql": {"aliases": ["sql", "postgresql", "mysql", "sql server"], "group": "database"},
    "excel": {"aliases": ["excel", "microsoft excel"], "group": "analytics_tools"},
    "power bi": {"aliases": ["power bi", "powerbi"], "group": "bi_tools"},
    "tableau": {"aliases": ["tableau"], "group": "bi_tools"},
    "pandas": {"aliases": ["pandas"], "group": "python_libs"},
    "numpy": {"aliases": ["numpy"], "group": "python_libs"},
    "scikit-learn": {"aliases": ["scikit-learn", "sklearn"], "group": "ml_libs"},
    "tensorflow": {"aliases": ["tensorflow"], "group": "ml_libs"},
    "pytorch": {"aliases": ["pytorch", "torch"], "group": "ml_libs"},
    "machine learning": {"aliases": ["machine learning", "ml"], "group": "ml_concepts"},
    "deep learning": {"aliases": ["deep learning", "dl"], "group": "ml_concepts"},
    "statistics": {"aliases": ["statistics", "thống kê"], "group": "analytics_concepts"},
    "etl": {"aliases": ["etl"], "group": "data_engineering"},
    "airflow": {"aliases": ["airflow", "apache airflow"], "group": "data_engineering"},
    "spark": {"aliases": ["spark", "apache spark", "pyspark"], "group": "data_engineering"},
    "hadoop": {"aliases": ["hadoop"], "group": "data_engineering"},
    "kafka": {"aliases": ["kafka", "apache kafka"], "group": "data_engineering"},
    "aws": {"aliases": ["aws", "amazon web services"], "group": "cloud"},
    "gcp": {"aliases": ["gcp", "google cloud platform"], "group": "cloud"},
    "azure": {"aliases": ["azure"], "group": "cloud"},
    "docker": {"aliases": ["docker"], "group": "devops"},
    "kubernetes": {"aliases": ["kubernetes", "k8s"], "group": "devops"},
    "git": {"aliases": ["git", "github", "gitlab"], "group": "dev_tools"},
    "api": {"aliases": ["api", "rest api", "restful api"], "group": "backend"},
    "fastapi": {"aliases": ["fastapi"], "group": "backend"},
    "flask": {"aliases": ["flask"], "group": "backend"},
    "llm": {"aliases": ["llm", "large language model", "large language models"], "group": "ai"},
    "nlp": {"aliases": ["nlp", "natural language processing"], "group": "ai"},
    "computer vision": {"aliases": ["computer vision", "cv"], "group": "ai"},
    "data modeling": {"aliases": ["data modeling", "data model"], "group": "data_modeling"},
    "data warehouse": {"aliases": ["data warehouse", "dwh"], "group": "data_modeling"},
    "bigquery": {"aliases": ["bigquery"], "group": "database"},
    "snowflake": {"aliases": ["snowflake"], "group": "database"},
    "oracle": {"aliases": ["oracle"], "group": "database"},
    "sas": {"aliases": ["sas"], "group": "analytics_tools"},
    "r": {"aliases": [" r ", "(r)", "ngôn ngữ r"], "group": "programming"},
    "communication": {"aliases": ["communication", "giao tiếp"], "group": "soft_skills"},
    "problem solving": {"aliases": ["problem solving", "giải quyết vấn đề"], "group": "soft_skills"},
}

REQUIRED_HINTS = [
    "bắt buộc", "required", "must have", "cần có", "thành thạo", "proficient", "kinh nghiệm với"
]
PREFERRED_HINTS = [
    "ưu tiên", "preferred", "nice to have", "là lợi thế", "plus point"
]


def alias_to_regex(alias):
    alias = safe_str(alias)
    if not alias:
        return None

    alias_strict = alias.strip()
    if alias_strict.lower() == "r":
        return r"(?<!\w)r(?!\w)"
    if alias_strict.lower() == "ml":
        return r"(?<!\w)ml(?!\w)"
    if alias_strict.lower() == "cv":
        return r"(?<!\w)cv(?!\w)"
    return r"(?<!\w)" + re.escape(alias_strict) + r"(?!\w)"


SKILL_PATTERNS = {}
for skill, meta in SKILL_TAXONOMY.items():
    pats = []
    for alias in meta["aliases"]:
        pat = alias_to_regex(alias)
        if pat:
            pats.append(re.compile(pat, flags=re.I))
    SKILL_PATTERNS[skill] = pats


def infer_skill_importance(segment, source_field):
    s = clean_text_strict(segment)
    if any(h in s for h in PREFERRED_HINTS):
        return "preferred"
    if source_field == "requirements":
        return "required"
    if any(h in s for h in REQUIRED_HINTS):
        return "required"
    return "mentioned"


def extract_skill_records_from_text(text, source_field="unknown"):
    text = clean_text_preserve_structure(text)
    if not text:
        return []

    segments = [seg.strip() for seg in re.split(r"[\n•\-;]+", text) if seg.strip()]
    records = []

    for seg in segments:
        for skill, patterns in SKILL_PATTERNS.items():
            matched = False
            for p in patterns:
                if p.search(seg):
                    matched = True
                    break
            if matched:
                records.append({
                    "skill": skill,
                    "skill_group": SKILL_TAXONOMY[skill]["group"],
                    "source_field": source_field,
                    "importance": infer_skill_importance(seg, source_field),
                    "excerpt": seg[:300]
                })

    seen = set()
    out = []
    for r in records:
        key = (r["skill"], r["source_field"], r["importance"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def merge_skill_records(*record_lists):
    merged = []
    seen = set()
    for records in record_lists:
        for r in records:
            key = (r["skill"], r["source_field"], r["importance"])
            if key not in seen:
                seen.add(key)
                merged.append(r)
    return merged


def list_from_records(records, key, importance_filter=None):
    vals = []
    for r in records:
        if importance_filter and r.get("importance") != importance_filter:
            continue
        if r.get(key):
            vals.append(r[key])
    return deduplicate_list(vals)

title_skill_records = df_clean["job_title_clean_phobert"].apply(lambda x: extract_skill_records_from_text(x, "title"))
tag_skill_records = df_clean["tags_text_phobert"].apply(lambda x: extract_skill_records_from_text(x, "tags"))
req_skill_records = df_clean["requirements_clean_phobert"].apply(lambda x: extract_skill_records_from_text(x, "requirements"))
desc_skill_records = df_clean["description_clean_phobert"].apply(lambda x: extract_skill_records_from_text(x, "description"))

df_clean["skill_records"] = [
    merge_skill_records(a, b, c, d)
    for a, b, c, d in zip(title_skill_records, tag_skill_records, req_skill_records, desc_skill_records)
]

df_clean["skills_extracted"] = df_clean["skill_records"].apply(lambda rs: list_from_records(rs, "skill"))
df_clean["skills_required"] = df_clean["skill_records"].apply(lambda rs: list_from_records(rs, "skill", "required"))
df_clean["skills_preferred"] = df_clean["skill_records"].apply(lambda rs: list_from_records(rs, "skill", "preferred"))
df_clean["skill_groups"] = df_clean["skill_records"].apply(lambda rs: list_from_records(rs, "skill_group"))

df_clean["skills_text_phobert"] = df_clean["skills_extracted"].apply(
    lambda xs: ", ".join(xs) if isinstance(xs, list) else ""
)
df_clean["skills_required_text_phobert"] = df_clean["skills_required"].apply(
    lambda xs: ", ".join(xs) if isinstance(xs, list) else ""
)
df_clean["skills_preferred_text_phobert"] = df_clean["skills_preferred"].apply(
    lambda xs: ", ".join(xs) if isinstance(xs, list) else ""
)

display(df_clean[
    [
        "job_title_raw", "skills_extracted", "skills_required",
        "skills_preferred", "skill_groups",
        "skills_text_phobert", "skills_required_text_phobert"
    ]
].head(10))

job_skill_rows = []
for _, row in df_clean.iterrows():
    for r in row["skill_records"]:
        job_skill_rows.append({
            "job_url": row.get("job_url"),
            "job_title_display": row.get("job_title_display"),
            "job_title_canonical": row.get("job_title_canonical"),
            "job_family": row.get("job_family"),
            "location_norm": row.get("location_norm"),
            "skill": r["skill"],
            "skill_group": r["skill_group"],
            "source_field": r["source_field"],
            "importance": r["importance"],
            "excerpt": r["excerpt"],
        })

job_skill_map_df = pd.DataFrame(job_skill_rows)
display(job_skill_map_df.head(10))

if len(job_skill_map_df) > 0:
    role_skill_stats_df = (
        job_skill_map_df.groupby(["job_family", "skill", "importance"])
        .size()
        .reset_index(name="job_count")
        .sort_values(["job_family", "job_count"], ascending=[True, False])
    )
else:
    role_skill_stats_df = pd.DataFrame(columns=["job_family", "skill", "importance", "job_count"])

display(role_skill_stats_df.head(20))
print("Tỷ lệ job không extract được skill:", (df_clean["skills_extracted"].apply(len) == 0).mean())

def format_salary_brief(row):
    mn = row.get("salary_min_vnd_month")
    mx = row.get("salary_max_vnd_month")
    if pd.notna(mn) and pd.notna(mx):
        return f"{int(mn):,}-{int(mx):,} VND/tháng"
    if pd.notna(mn):
        return f"từ {int(mn):,} VND/tháng"
    if row.get("salary_is_negotiable") is True:
        return "thỏa thuận"
    return ""


def build_job_text_sparse(row):
    parts = []

    for field in [
        row.get("job_title_canonical"),
        row.get("job_family"),
        row.get("seniority_final"),
        row.get("tags_text_phobert"),
        row.get("skills_text_phobert"),
        row.get("skills_required_text_phobert"),
        row.get("requirements_clean_strict"),
        row.get("description_clean_strict"),
    ]:
        s = safe_str(field)
        if s and s != "unknown":
            parts.append(s)

    return "\n".join(parts).strip()


def build_job_text_phobert_match(row):
    parts = []

    title = row.get("job_title_for_phobert")
    family = row.get("job_family")
    seniority = row.get("seniority_final")
    required_skills = row.get("skills_required_text_phobert")
    preferred_skills = row.get("skills_preferred_text_phobert")
    exp = row.get("experience_min_years")
    location = row.get("location_norm")
    work_mode = row.get("work_mode")
    req = truncate_by_words(row.get("requirements_clean_phobert"), 160)

    if title:
        parts.append(f"Vị trí: {title}")
    if family and family != "unknown":
        parts.append(f"Nhóm nghề: {family}")
    if seniority and seniority != "unknown":
        parts.append(f"Cấp bậc: {seniority}")
    if required_skills:
        parts.append(f"Kỹ năng bắt buộc: {required_skills}")
    if preferred_skills:
        parts.append(f"Kỹ năng ưu tiên: {preferred_skills}")
    if exp is not None and not pd.isna(exp):
        parts.append(f"Kinh nghiệm tối thiểu: {exp} năm")
    if location:
        parts.append(f"Địa điểm: {location}")
    if work_mode and work_mode != "unknown":
        parts.append(f"Hình thức làm việc: {work_mode}")
    if req:
        parts.append(f"Yêu cầu chính: {req}")

    return "\n".join(parts).strip()


def build_job_text_phobert_chatbot(row):
    parts = []

    if row.get("job_title_for_phobert"):
        parts.append(f"Vị trí tuyển dụng: {row['job_title_for_phobert']}")
    if row.get("job_family") and row["job_family"] != "unknown":
        parts.append(f"Nhóm công việc: {row['job_family']}")
    if row.get("seniority_final") and row["seniority_final"] != "unknown":
        parts.append(f"Cấp bậc: {row['seniority_final']}")
    if row.get("location_norm"):
        parts.append(f"Địa điểm: {row['location_norm']}")
    if row.get("work_mode") and row["work_mode"] != "unknown":
        parts.append(f"Hình thức làm việc: {row['work_mode']}")
    salary_brief = format_salary_brief(row)
    if salary_brief:
        parts.append(f"Mức lương: {salary_brief}")
    if row.get("skills_required_text_phobert"):
        parts.append(f"Kỹ năng bắt buộc: {row['skills_required_text_phobert']}")
    if row.get("skills_preferred_text_phobert"):
        parts.append(f"Kỹ năng ưu tiên: {row['skills_preferred_text_phobert']}")
    if row.get("requirements_clean_phobert"):
        parts.append(f"Yêu cầu:\n{truncate_by_words(row['requirements_clean_phobert'], 220)}")
    if row.get("description_clean_phobert"):
        parts.append(f"Mô tả công việc:\n{truncate_by_words(row['description_clean_phobert'], 220)}")
    if row.get("benefits_clean_phobert"):
        parts.append(f"Quyền lợi:\n{truncate_by_words(row['benefits_clean_phobert'], 120)}")

    return "\n\n".join(parts).strip()

df_clean["job_text_sparse"] = df_clean.apply(build_job_text_sparse, axis=1)
df_clean["job_text_phobert_match"] = df_clean.apply(build_job_text_phobert_match, axis=1)
df_clean["job_text_phobert_chatbot"] = df_clean.apply(build_job_text_phobert_chatbot, axis=1)

df_clean["dense_encoder_route"] = "phobert"
df_clean["dense_similarity_metric"] = "cosine"

display(df_clean[
    [
        "job_title_raw",
        "job_family",
        "seniority_final",
        "job_text_sparse",
        "job_text_phobert_match",
        "job_text_phobert_chatbot"
    ]
].head(3))

def split_long_text(text, max_chars=700, overlap=80):
    text = clean_text_preserve_structure(text)
    if not text:
        return []

    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for para in paras:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            if len(para) <= max_chars:
                current = para
            else:
                start = 0
                while start < len(para):
                    end = min(start + max_chars, len(para))
                    chunk = para[start:end]
                    chunks.append(chunk.strip())
                    start = max(end - overlap, end)
                current = ""

    if current:
        chunks.append(current)

    return [c.strip() for c in chunks if c.strip()]


SECTION_PRIORITY = {
    "title": 5.0,
    "requirements": 4.5,
    "description": 4.0,
    "benefits": 3.0,
    "company": 2.5,
}


def build_chunk_text_phobert(row, section_type: str, chunk_text: str):
    title = safe_str(row.get("job_title_for_phobert"))
    family = safe_str(row.get("job_family"))
    seniority = safe_str(row.get("seniority_final"))

    section_map = {
        "title": "Tiêu đề",
        "requirements": "Yêu cầu",
        "description": "Mô tả công việc",
        "benefits": "Quyền lợi",
        "company": "Giới thiệu công ty",
    }
    section_label = section_map.get(section_type, section_type)

    parts = []
    if title:
        parts.append(f"Vị trí: {title}")
    if family and family != "unknown":
        parts.append(f"Nhóm nghề: {family}")
    if seniority and seniority != "unknown":
        parts.append(f"Cấp bậc: {seniority}")
    parts.append(f"Phần: {section_label}")
    parts.append(truncate_by_words(chunk_text, 180))
    return "\n".join([p for p in parts if p]).strip()


def build_job_section_records(row):
    rows = []

    section_map = {
        "title": safe_str(row.get("job_title_for_phobert")),
        "requirements": safe_str(row.get("requirements_clean_phobert")),
        "description": safe_str(row.get("description_clean_phobert")),
        "benefits": safe_str(row.get("benefits_clean_phobert")),
        "company": safe_str(row.get("company_description_clean_phobert")),
    }

    for section_type, text in section_map.items():
        if not text:
            continue

        chunks = [text] if section_type == "title" else split_long_text(text, max_chars=700, overlap=80)
        for chunk_order, chunk_text in enumerate(chunks):
            rows.append({
                "job_url": row.get("job_url"),
                "job_title_display": row.get("job_title_display"),
                "job_title_canonical": row.get("job_title_canonical"),
                "job_family": row.get("job_family"),
                "job_family_source": row.get("job_family_source"),
                "seniority_final": row.get("seniority_final"),
                "location_norm": row.get("location_norm"),
                "work_mode": row.get("work_mode"),
                "section_type": section_type,
                "chunk_order": chunk_order,
                "section_priority": SECTION_PRIORITY.get(section_type, 1.0),
                "chunk_text_raw": chunk_text,
                "chunk_text_phobert": build_chunk_text_phobert(row, section_type, chunk_text),
            })

    return rows


section_rows = []
for _, row in df_clean.iterrows():
    section_rows.extend(build_job_section_records(row))

job_sections_df = pd.DataFrame(section_rows)
display(job_sections_df.head(10))
print("job_sections_df shape:", job_sections_df.shape)

def maybe_segment_vi_text(text):
    text = safe_str(text)
    if not text:
        return ""
    if HAS_UNDERTHESEA:
        try:
            return word_tokenize(text, format="text")
        except Exception:
            return text
    return text


def prepare_phobert_text(text: str) -> str:
    text = clean_text_for_phobert(text)
    text = maybe_segment_vi_text(text)
    return text if text else "[EMPTY]"


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def cosine_similarity_matrix(query_emb, doc_embs):
    return np.dot(query_emb, doc_embs.T)

import os
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

HF_CACHE_DIR = Path("./hf_cache")
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or None

# Override via env when chạy pipeline script:
# PREPROCESS_USE_LOCAL_ONLY=1 / 0
# PREPROCESS_HF_OFFLINE=1 / 0
USE_LOCAL_ONLY = env_bool("PREPROCESS_USE_LOCAL_ONLY", True)
HF_OFFLINE = env_bool("PREPROCESS_HF_OFFLINE", True)
os.environ["HF_HUB_ETAG_TIMEOUT"] = "10"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

if HF_OFFLINE:
    os.environ["HF_HUB_OFFLINE"] = "1"


def load_phobert_model(model_name, cache_dir, local_only=False):
    """
    Load tokenizer + model theo thứ tự ưu tiên:
    1) local-only nếu được yêu cầu
    2) online với cache_dir
    3) fallback local-only nếu online fail
    """
    print(f"[INFO] Loading tokenizer/model from: {model_name}")
    print(f"[INFO] Cache dir: {cache_dir.resolve()}")
    print(f"[INFO] local_files_only={local_only}")

    if local_only:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            local_files_only=True,
            token=HF_TOKEN
        )
        # Try safetensors first, then fallback to pytorch_model.bin in local cache.
        last_exc = None
        for use_safetensors in (True, False):
            try:
                model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=str(cache_dir),
                    local_files_only=True,
                    use_safetensors=use_safetensors,
                    token=HF_TOKEN
                )
                return tokenizer, model
            except Exception as exc:
                last_exc = exc
        raise last_exc  # type: ignore[misc]

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            token=HF_TOKEN
        )
        try:
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                use_safetensors=True,
                token=HF_TOKEN
            )
        except Exception:
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                use_safetensors=False,
                token=HF_TOKEN
            )
        return tokenizer, model

    except Exception as e:
        print("[WARN] Online load failed. Trying local cache only...")
        print("[WARN] Reason:", repr(e))

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            local_files_only=True,
            token=HF_TOKEN
        )
        try:
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                local_files_only=True,
                use_safetensors=True,
                token=HF_TOKEN
            )
        except Exception:
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                local_files_only=True,
                use_safetensors=False,
                token=HF_TOKEN
            )
        return tokenizer, model


if RUN_EMBEDDING:
    try:
        tokenizer, model = load_phobert_model(
            model_name=PHOBERT_MODEL_NAME,
            cache_dir=HF_CACHE_DIR,
            local_only=USE_LOCAL_ONLY
        )

        model.to(DEVICE)
        model.eval()

        print(f"[INFO] Device: {DEVICE}")
        print("[INFO] PhoBERT loaded successfully.")

        def encode_phobert_texts(
            texts,
            batch_size=PHOBERT_BATCH_SIZE,
            max_length=PHOBERT_MAX_LENGTH_MATCH
        ):
            if texts is None or len(texts) == 0:
                hidden_size = getattr(model.config, "hidden_size", 768)
                return np.empty((0, hidden_size), dtype=np.float32)

            embeddings = []
            prepared = [prepare_phobert_text(t) for t in texts]

            for i in tqdm(range(0, len(prepared), batch_size), desc="Encoding"):
                batch = prepared[i:i + batch_size]

                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )

                input_ids = encoded["input_ids"].to(DEVICE)
                attention_mask = encoded["attention_mask"].to(DEVICE)

                with torch.no_grad():
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )

                batch_emb = mean_pooling(output, attention_mask)

                if NORMALIZE_EMBEDDINGS:
                    batch_emb = F.normalize(batch_emb, p=2, dim=1)

                embeddings.append(batch_emb.cpu().numpy().astype(np.float32))

            return np.vstack(embeddings)

        _test_emb = encode_phobert_texts(["data analyst"], batch_size=1, max_length=16)
        print(f"[INFO] Test embedding shape: {_test_emb.shape}")

    except Exception as e:
        print("[ERROR] Failed to load or initialize PhoBERT.")
        print("Reason:", repr(e))
        raise

else:
    print("[INFO] RUN_EMBEDDING=False -> skip model loading.")

job_dense_embeddings = None

df_clean["embedding_row_id"] = np.arange(len(df_clean))

if RUN_EMBEDDING:
    job_dense_embeddings = encode_phobert_texts(
        df_clean["job_text_phobert_match"].fillna("").tolist(),
        batch_size=PHOBERT_BATCH_SIZE,
        max_length=PHOBERT_MAX_LENGTH_MATCH,
    )
    df_clean["has_dense_embedding"] = 1
    print("job_dense_embeddings shape:", job_dense_embeddings.shape)
else:
    df_clean["has_dense_embedding"] = 0
    print("[INFO] Skip job-level embedding.")

section_dense_embeddings = None

if RUN_EMBEDDING and RUN_SECTION_EMBEDDING and len(job_sections_df) > 0:
    job_sections_df["section_embedding_row_id"] = np.arange(len(job_sections_df))
    section_dense_embeddings = encode_phobert_texts(
        job_sections_df["chunk_text_phobert"].fillna("").tolist(),
        batch_size=PHOBERT_BATCH_SIZE,
        max_length=PHOBERT_MAX_LENGTH_CHUNK,
    )
    job_sections_df["has_dense_embedding"] = 1
    print("section_dense_embeddings shape:", section_dense_embeddings.shape)
else:
    job_sections_df["has_dense_embedding"] = 0
    print("[INFO] Skip section-level embedding.")

def encode_query_for_matching(query: str, max_length=PHOBERT_MAX_LENGTH_MATCH):
    q = encode_phobert_texts([query], batch_size=1, max_length=max_length)
    return q[0]


def retrieve_top_jobs(query: str, top_k: int = 10):
    if job_dense_embeddings is None:
        raise RuntimeError("job_dense_embeddings is None. Hãy bật RUN_EMBEDDING=True")

    q_emb = encode_query_for_matching(query, max_length=PHOBERT_MAX_LENGTH_MATCH)
    scores = job_dense_embeddings @ q_emb
    top_idx = np.argsort(scores)[::-1][:top_k]

    cols = [
        "job_url", "job_title_display", "job_family", "location_norm",
        "work_mode", "skills_required", "skills_preferred", "job_text_phobert_match"
    ]
    out = df_clean.iloc[top_idx][cols].copy()
    out["cosine_score"] = scores[top_idx]
    return out.reset_index(drop=True)


def retrieve_top_sections(query: str, top_k: int = 10):
    if section_dense_embeddings is None:
        raise RuntimeError("section_dense_embeddings is None. Hãy bật RUN_SECTION_EMBEDDING=True")

    q_emb = encode_query_for_matching(query, max_length=PHOBERT_MAX_LENGTH_CHUNK)
    scores = section_dense_embeddings @ q_emb
    top_idx = np.argsort(scores)[::-1][:top_k]

    cols = [
        "job_url", "job_title_display", "job_family", "section_type",
        "chunk_order", "section_priority", "chunk_text_raw"
    ]
    out = job_sections_df.iloc[top_idx][cols].copy()
    out["cosine_score"] = scores[top_idx]
    return out.reset_index(drop=True)


demo_query = "Data Analyst cần SQL Power BI và kinh nghiệm phân tích dữ liệu"
try:
    display(retrieve_top_jobs(demo_query, top_k=5))
    display(retrieve_top_sections(demo_query, top_k=5))
except Exception as e:
    print("[INFO] Demo retrieval skipped:", e)

DOWNSTREAM_FIELD_GUIDE = {
    "tfidf_input": "job_text_sparse",
    "phobert_matching_input": "job_text_phobert_match",
    "phobert_chatbot_input": "job_text_phobert_chatbot",
    "chatbot_chunk_table": "job_sections_df",
    "chatbot_chunk_text_field": "chunk_text_phobert",
    "skill_table": "job_skill_map_df",
    "role_skill_stats": "role_skill_stats_df",
    "job_family_field": "job_family (final: title + description)",
    "seniority_field": "seniority_final (from job_level_raw)",
    "retrieval_metric": "cosine_similarity_on_l2_normalized_embeddings",
}

pd.Series(DOWNSTREAM_FIELD_GUIDE)

matching_cols = [
    "job_url", "job_id",
    "job_title_display", "job_title_canonical", "job_family",
    "location_norm", "location_city", "location_district", "work_mode",
    "salary_min_vnd_month", "salary_max_vnd_month", "salary_is_negotiable",
    "experience_min_years", "experience_max_years", "experience_type",
    "education_level_norm", "employment_type_norm", "job_level_norm", "seniority_from_title",
    "skills_extracted", "skills_required", "skills_preferred", "skill_groups",
    "job_text_sparse", "job_text_phobert_match", "job_text_phobert_chatbot",
    "embedding_row_id", "has_dense_embedding", "dense_encoder_route", "dense_similarity_metric",
]

matching_cols = [c for c in matching_cols if c in df_clean.columns]
df_matching_ready = df_clean[matching_cols].copy()
df_matching_ready["dense_model_name"] = PHOBERT_MODEL_NAME
df_matching_ready["dense_similarity_metric"] = "cosine"

display(df_matching_ready.head(3))
print(df_matching_ready.shape)

chatbot_cols = [
    "job_url", "job_id",
    "job_title_display", "job_title_canonical",
    "job_family", "job_family_source",
    "seniority_final",
    "location_norm", "work_mode",
    "salary_min_vnd_month", "salary_max_vnd_month",
    "experience_min_years", "experience_max_years",
    "skills_extracted", "skills_required", "skills_preferred",
    "requirements_clean_phobert", "description_clean_phobert", "benefits_clean_phobert",
    "job_text_phobert_chatbot",
    "embedding_row_id", "has_dense_embedding"
]

chatbot_cols = [c for c in chatbot_cols if c in df_clean.columns]
df_chatbot_ready = df_clean[chatbot_cols].copy()

display(df_chatbot_ready.head(3))
print(df_chatbot_ready.shape)

section_cols = [
    "section_embedding_row_id",
    "job_url", "job_title_display", "job_title_canonical",
    "job_family", "location_norm", "work_mode",
    "section_type", "chunk_order", "section_priority",
    "chunk_text_raw", "chunk_text_phobert",
    "has_dense_embedding"
]
section_cols = [c for c in section_cols if c in job_sections_df.columns]

job_sections_ready = job_sections_df[section_cols].copy()
display(job_sections_ready.head(3))
print(job_sections_ready.shape)

artifact_paths = {}

artifact_paths["jobs_matching_ready"] = save_table(
    df_matching_ready, ARTIFACT_DIR / artifact_name("jobs_matching_ready")
)

artifact_paths["jobs_chatbot_ready"] = save_table(
    df_chatbot_ready, ARTIFACT_DIR / artifact_name("jobs_chatbot_ready")
)

artifact_paths["jobs_chatbot_sections"] = save_table(
    job_sections_ready, ARTIFACT_DIR / artifact_name("jobs_chatbot_sections")
)

artifact_paths["job_skill_map"] = save_table(
    job_skill_map_df, ARTIFACT_DIR / artifact_name("job_skill_map")
)

artifact_paths["role_skill_stats"] = save_table(
    role_skill_stats_df, ARTIFACT_DIR / artifact_name("role_skill_stats")
)

artifact_paths["job_embedding_index"] = save_table(
    df_clean[["embedding_row_id", "job_url", "job_title_display", "job_text_phobert_match"]],
    ARTIFACT_DIR / artifact_name("job_embedding_index")
)

if len(job_sections_df) > 0:
    section_index_cols = [
        "section_embedding_row_id",
        "job_url",
        "job_title_display",
        "section_type",
        "chunk_order",
        "chunk_text_raw",
    ]
    section_index_cols = [c for c in section_index_cols if c in job_sections_df.columns]
    artifact_paths["job_section_embedding_index"] = save_table(
        job_sections_df[section_index_cols],
        ARTIFACT_DIR / artifact_name("job_section_embedding_index")
    )

if RUN_EMBEDDING and job_dense_embeddings is not None:
    job_emb_path = ARTIFACT_DIR / f"{artifact_name('job_dense_embeddings')}.npy"
    np.save(job_emb_path, job_dense_embeddings)
    artifact_paths["job_dense_embeddings"] = str(job_emb_path)

if RUN_EMBEDDING and RUN_SECTION_EMBEDDING and section_dense_embeddings is not None:
    section_emb_path = ARTIFACT_DIR / f"{artifact_name('section_dense_embeddings')}.npy"
    np.save(section_emb_path, section_dense_embeddings)
    artifact_paths["section_dense_embeddings"] = str(section_emb_path)

print("[INFO] Saved main artifacts.")
pd.Series(artifact_paths)

manifest = {
    "notebook_version": NOTEBOOK_VERSION,
    "run_timestamp": datetime.now().isoformat(),
    "raw_input_path": str(RAW_INPUT_PATH),
    "artifact_dir": str(ARTIFACT_DIR.resolve()),
    "n_jobs": int(len(df_clean)),
    "n_sections": int(len(job_sections_df)),
    "embedding_config": {
        "model_name": PHOBERT_MODEL_NAME,
        "metric": "cosine",
        "normalize_embeddings": NORMALIZE_EMBEDDINGS,
        "job_text_field": "job_text_phobert_match",
        "chatbot_text_field": "job_text_phobert_chatbot",
        "section_text_field": "chunk_text_phobert",
        "max_length_match": PHOBERT_MAX_LENGTH_MATCH,
        "max_length_chatbot": PHOBERT_MAX_LENGTH_CHATBOT,
        "max_length_chunk": PHOBERT_MAX_LENGTH_CHUNK,
        "batch_size": PHOBERT_BATCH_SIZE,
        "segmentation": "underthesea_if_available",
    },
    "downstream_field_guide": DOWNSTREAM_FIELD_GUIDE,

    "derived_fields": {
        "job_title_canonical": "normalized from job_title_raw after removing trailing noise, bracket noise, and synonym mapping",
        "job_family_from_title": "inferred from job_title_canonical using JOB_FAMILY_RULES",
        "job_family_from_description": "inferred from description_clean_strict using JOB_FAMILY_DESCRIPTION_HINTS",
        "job_family": "final resolved job family using title priority over description",
        "job_family_source": "source of final job_family: title / description / title+description / title_priority / unknown",
        "job_level_norm": "normalized from job_level_raw using JOB_LEVEL_RULES",
        "seniority_final": "final seniority derived from job_level_raw only",
        "seniority_source": "source of final seniority, expected to be job_level_raw or unknown",
    },

    "output_schema_notes": {
        "matching_table": "df_matching_ready uses job_family and seniority_final as final structured fields",
        "chatbot_table": "df_chatbot_ready uses job_family and seniority_final for job summary display",
        "section_table": "job_sections_ready uses job_family and seniority_final for chunk-level retrieval context",
    },

    "artifacts": artifact_paths,
}

manifest_path = ARTIFACT_DIR / f"{artifact_name('manifest_phobert')}.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print("[INFO] Manifest saved:", manifest_path)
display(pd.Series(manifest))

length_stats = pd.DataFrame({
    "job_text_sparse_len": df_clean["job_text_sparse"].fillna("").str.len(),
    "job_text_phobert_match_len": df_clean["job_text_phobert_match"].fillna("").str.len(),
    "job_text_phobert_chatbot_len": df_clean["job_text_phobert_chatbot"].fillna("").str.len(),
})

display(length_stats.describe())

if len(job_sections_df) > 0:
    section_stats = (
        job_sections_df.assign(chunk_len=job_sections_df["chunk_text_phobert"].fillna("").str.len())
        .groupby("section_type")["chunk_len"]
        .describe()
        .reset_index()
    )
    display(section_stats)

print("Avg skills per job:", df_clean["skills_extracted"].apply(len).mean())
print("Jobs without extracted skills:", (df_clean["skills_extracted"].apply(len) == 0).sum())







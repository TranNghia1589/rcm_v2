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

NOTEBOOK_VERSION = "preprocessing_v3_phobert"

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
SAVE_INTERMEDIATE = env_bool("PREPROCESS_SAVE_INTERMEDIATE", True)

PHOBERT_MODEL_NAME = "vinai/phobert-base"
PHOBERT_MAX_LENGTH_MATCH = 256
PHOBERT_MAX_LENGTH_CHATBOT = 320
PHOBERT_MAX_LENGTH_CHUNK = 256
PHOBERT_BATCH_SIZE = 16
NORMALIZE_EMBEDDINGS = True
LOCAL_MODEL_DIR_RAW = os.getenv("PREPROCESS_LOCAL_MODEL_DIR", "").strip()
HF_CACHE_DIR = Path(os.getenv("PREPROCESS_HF_CACHE_DIR", str(BASE_DIR / "hf_cache"))).resolve()
HF_HUB_CACHE_DIR = Path(os.getenv("PREPROCESS_HF_HUB_CACHE_DIR", str(HF_CACHE_DIR))).resolve()
USE_LOCAL_ONLY = env_bool("PREPROCESS_USE_LOCAL_ONLY", True)
HF_OFFLINE = env_bool("PREPROCESS_HF_OFFLINE", USE_LOCAL_ONLY)
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
HF_HUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
# Default to the merged local PhoBERT folder already prepared in this project cache.
DEFAULT_LOCAL_MODEL_DIR = HF_CACHE_DIR / "models--vinai--phobert-base" / "merged_local"
LOCAL_MODEL_DIR = Path(LOCAL_MODEL_DIR_RAW).resolve() if LOCAL_MODEL_DIR_RAW else DEFAULT_LOCAL_MODEL_DIR.resolve()
# Keep HF cache paths explicit so preprocess always resolves model files from expected local cache.
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HUB_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE_DIR))
if HF_OFFLINE:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("NOTEBOOK_VERSION:", NOTEBOOK_VERSION)
print("RAW_INPUT_PATH:", RAW_INPUT_PATH)
print("ARTIFACT_DIR:", ARTIFACT_DIR.resolve())
print("RUN_EMBEDDING:", RUN_EMBEDDING)
print("RUN_SECTION_EMBEDDING:", RUN_SECTION_EMBEDDING)
print("SAVE_INTERMEDIATE:", SAVE_INTERMEDIATE)
print("PHOBERT_MODEL_NAME:", PHOBERT_MODEL_NAME)
print("LOCAL_MODEL_DIR:", LOCAL_MODEL_DIR)
print("HF_CACHE_DIR:", HF_CACHE_DIR)
print("HF_HUB_CACHE_DIR:", HF_HUB_CACHE_DIR)
print("PREPROCESS_USE_LOCAL_ONLY:", USE_LOCAL_ONLY)
print("PREPROCESS_HF_OFFLINE:", HF_OFFLINE)
print("DEVICE:", DEVICE)
print("HAS_UNDERTHESEA:", HAS_UNDERTHESEA)


def log_info(msg):
    print(f"[INFO] {msg}")


def log_error(step, err):
    print(f"[ERROR] Step '{step}' failed: {err}")


def run_step(step_name, fn):
    log_info(f"Starting: {step_name}")
    try:
        fn()
        log_info(f"Completed: {step_name}")
    except Exception as e:
        log_error(step_name, e)
        raise


# Chuẩn hóa giá trị rỗng
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

# Chuẩn hóa thành chuỗi
def safe_str(x):
    x = normalize_empty_value(x)
    return "" if x is None else str(x)

# Lấy Series từ DataFrame, nếu không có thì trả về Series mặc định
def get_series(df, col, default=None):
    if col in df.columns:
        return df[col]
    if default is None:
        default = [None] * len(df)
    return pd.Series(default, index=df.index)

# Lấy giá trị đầu tiên không rỗng từ danh sách các giá trị
def first_non_empty(*values):
    for v in values:
        v = normalize_empty_value(v)
        if v is not None:
            return v
    return None

# Chuẩn hóa unicode
def normalize_unicode(text):
    text = safe_str(text)
    return unicodedata.normalize("NFC", text)

# Xóa thẻ html
def remove_html(text):
    text = safe_str(text)
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return text

# Chuẩn hóa các kiểu gạch đầu dòng khác nhau về "-"
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

# Loại bỏ phần tử trùng lặp trong list, giữ nguyên thứ tự
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
# Chuẩn hóa văn bản để phục vụ cho việc matching, loại bỏ dấu câu không cần thiết, chuyển về chữ thường, chuẩn hóa unicode, loại bỏ dấu tiếng Việt để tăng khả năng matching
def normalize_for_match(text):
    text = clean_text_light(text)
    if not text:
        return ""

    text = text.lower()
    text = text.replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\s/\+\-#\.]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Loại bỏ dòng trùng lặp trong văn bản, giữ nguyên thứ tự và cấu trúc cơ bản
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

# Các hàm làm sạch văn bản ở nhiều mức độ khác nhau
# Làm sạch nhẹ, giữ nguyên cấu trúc cơ bản (giữ format gốc, chỉ loại bỏ những thứ thực sự không cần thiết)
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

# Làm sạch nghiêm ngặt, loại bỏ hầu hết dấu câu, chỉ giữ lại chữ và số (giữ nguyên cấu trúc cơ bản như xuống dòng và gạch đầu dòng để preserve meaning)
def clean_text_preserve_structure(text):
    text = clean_text_light(text)
    if not text:
        return ""
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n\s*-\s*", "\n- ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# Làm sạch nghiêm ngặt hơn, chỉ giữ lại chữ, số và một số dấu câu cơ bản (để vector hóa và matching)
def clean_text_strict(text):
    text = clean_text_light(text)
    if not text:
        return ""
    text = text.lower() # chuyển về chữ thường để chuẩn hóa
    text = re.sub(r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ/+\-#\.]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Làm sạch cho PhoBERT, giữ lại dấu câu cần thiết để preserve meaning 
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

    # giữ dấu câu đủ để preserve meaning cho PhoBERT
    text = re.sub(r"[^\w\sÀ-ỹ\.,;:/\-\+\#\(\)%\n]", " ", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)

    return text.strip()

# Rút gọn văn bản theo số lượng từ, giữ nguyên cấu trúc cơ bản
def truncate_by_words(text, max_words=220):
    text = safe_str(text)
    if not text:
        return ""
    words = text.split()
    return " ".join(words[:max_words]).strip()

# Lưu DataFrame thành Parquet nếu có thể, nếu không thì lưu CSV
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

# Import dữ liệu vào schema chung
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
            get_series(df, "detail_location"),
            get_series(df, "address_list")
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

# Kiểm tra tỷ lệ trùng lặp, tỷ lệ missing, và phân loại ngôn ngữ
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

# =========================
# TITLE PROCESSING - REVISED
# =========================

TITLE_SEGMENT_NOISE_PATTERNS = [
    # generic separators / garbage
    r"^\|.*$",
    r"^_+$",

    # salary / compensation / benefit
    r"^luong.*$",
    r"^thu nhap.*$",
    r"^salary.*$",
    r"^upto.*$",
    r"^up to.*$",
    r"^offer.*$",
    r"^phuc loi.*$",
    r"^benefit.*$",
    r"^gross.*$",
    r"^net.*$",
    r"^\d+\s*tr.*$",
    r"^\d+\s*trieu.*$",

    # location / workplace
    r"^ha noi.*$",
    r"^hn.*$",
    r"^ho chi minh.*$",
    r"^tp\.?\s*hcm.*$",
    r"^tphcm.*$",
    r"^hcm.*$",
    r"^sai gon.*$",
    r"^da nang.*$",
    r"^hai phong.*$",
    r"^can tho.*$",
    r"^bac ninh.*$",
    r"^thu duc.*$",
    r"^cau giay.*$",
    r"^dong da.*$",
    r"^quan \d+.*$",
    r"^tai ha noi.*$",
    r"^tai ho chi minh.*$",
    r"^tai tp\.?\s*hcm.*$",
    r"^lam viec tai.*$",

    # work mode / urgency / shift
    r"^di lam ngay.*$",
    r"^ob som.*$",
    r"^remote.*$",
    r"^hybrid.*$",
    r"^onsite.*$",
    r"^urgent.*$",
    r"^hot.*$",
    r"^\d{1,2}h\s*-\s*\d{1,2}h.*$",

    # org / department noise
    r"^khoi .*",
    r"^phong .*",
    r"^ban .*",
    r"^team .*",
    r"^du an .*",
    r"^mang .*",
    r"^domain .*",

    # internal code
    r"^ta\d+.*$",
    r"^ho\d{2}\.\d+.*$",
    r"^holt\.\d+.*$",
    r"^it\d+.*$",
    r"^\d+\..*$",

    # experience / requirement
    r"^tu\s*\d+\s*nam.*$",
    r"^\d+\+?\s*nam.*$",
    r"^\d+\s*nam kinh nghiem.*$",
    r"^kinh nghiem.*$",
    r"^yeu cau.*$",
    r"^khong yeu cau kinh nghiem.*$",
    r"^at least.*$",
    r"^english required.*$",
    r"^good at english.*$",
    r"^tieng anh.*$",
    r"^jlpt.*$",
    r"^n\d\+?.*$",

    # company / branding / contextual tails
    r"^funtap.*$",
    r"^coolmate.*$",
    r"^misa.*$",
    r"^banking project.*$",
    r"^healthcare.*$",
    r"^bank.*$",
    r"^fintech.*$",
    r"^edtech.*$",
    r"^ecommerce.*$",
    r"^mobile app.*$",
    r"^web app.*$",

    # standalone seniority / working form in brackets or split tails
    r"^intern$",
    r"^junior$",
    r"^middle$",
    r"^senior$",
    r"^lead$",
    r"^leader$",
    r"^team lead$",
    r"^fresher$",
    r"^fulltime$",
    r"^full time$",
    r"^m/f/d$",
    r"^all levels$",
]

BRACKET_NOISE_KEYWORDS = {
    # location
    "ha noi", "hn", "ho chi minh", "hcm", "tp hcm", "tphcm", "sai gon",
    "da nang", "hai phong", "can tho", "bac ninh", "thu duc", "cau giay", "dong da",

    # work mode / urgency
    "remote", "hybrid", "onsite", "urgent", "hot", "di lam ngay", "ob som",

    # internal codes
    "ta105", "ta150", "ta171", "ta172", "ta174", "ta188",
    "ho26.37", "ho26.69", "ho26.74", "ho26.76", "ho26.77", "ho26.78", "ho26.79", "ho26.84", "ho26.85", "ho26.95", "ho26.96",
    "holt.03", "holt.04", "holt.05", "holt.07", "holt.08", "holt.09", "holt.11", "holt.12", "holt.13", "holt.14", "holt.15", "holt.16", "holt.17", "holt.18", "holt.19",

    # standalone seniority / working form
    "intern", "junior", "middle", "senior", "lead", "leader", "team lead", "fresher", "fulltime", "full time", "m/f/d", "all levels",

    # generic contextual noise in brackets
    "banking", "fintech", "game industry", "domain erp", "domain edtech",
    "salary upto 25m", "salary upto 35m", "salary upto 50m",
}

# IMPORTANT:
# Keys in TITLE_SYNONYM_MAP are stored in normalize_for_match() form.
# Canonical values are intentionally compact/stable for grouping and downstream analysis.
TITLE_SYNONYM_MAP = {
    # =====================
    # DATA ANALYST
    # =====================
    "data analyst": "data analyst",
    "senior data analyst": "data analyst",
    "middle data analyst": "data analyst",
    "junior data analyst": "data analyst",
    "chuyen vien phan tich du lieu": "data analyst",
    "nhan vien phan tich du lieu": "data analyst",
    "phan tich du lieu": "data analyst",
    "phan tich du lieu kinh doanh": "data analyst",
    "chuyen vien phan tich va thiet ke mo hinh du lieu": "data analyst",
    "chuyen vien phan tich du lieu nguoi dung": "data analyst",
    "business data analyst": "data analyst",
    "business customer data analyst": "data analyst",
    "customer data analyst": "data analyst",
    "data analysis executive": "data analyst",
    "assistant manager data analyst": "data analyst",
    "strategic data lead": "data analyst",
    "data analyst teamleader": "data analyst",
    "customer data analyst team lead": "data analyst",
    "hr data analysis": "data analyst",
    "ecommerce business data analyst": "data analyst",
    "cvcc phan tich du lieu": "data analyst",
    "chuyen vien du lieu tai chinh": "data analyst",
    "chuyen vien phan tich va quan ly du lieu tai chinh": "data analyst",
    "credit risk analytics and modelling expert": "data analyst",
    "expert fraud risk data analytics and portfolio management": "data analyst",
    "fp&a analyst": "data analyst",
    "finance planning & analysis associate": "data analyst",
    "junior fp&a analyst": "data analyst",
    "bi analyst": "data analyst",
    "power bi leader": "data analyst",
    "presale data and analytics": "data analyst",

    # =====================
    # DATA ENGINEER
    # =====================
    "data engineer": "data engineer",
    "fresher data engineer": "data engineer",
    "thuc tap sinh data engineer": "data engineer",
    "data engineer intern": "data engineer",
    "nhan vien data engineer": "data engineer",
    "nhan vien data engineering": "data engineer",
    "chuyen vien du lieu data engineer": "data engineer",
    "ky su du lieu": "data engineer",
    "ky su du lieu lon": "data engineer",
    "big data engineer": "data engineer",
    "big data admin": "data engineer",
    "data integration engineer": "data engineer",
    "chuyen vien data integration engineer": "data engineer",
    "junior data integration engineer": "data engineer",
    "data warehouse engineer": "data engineer",
    "fresher data warehouse": "data engineer",
    "fresher r&d engineer data warehouse": "data engineer",
    "data platform engineer": "data engineer",
    "data platform operation": "data engineer",
    "data center storage engineer": "data engineer",
    "data engineer aws": "data engineer",
    "aws data engineer": "data engineer",
    "data engineer java python scala hadoop kafka": "data engineer",
    "data management": "data engineer",
    "chuyen vien du lieu": "data engineer",
    "chuyen gia du lieu": "data engineer",
    "gsd engineer garment standard data engineer ky su he thong du lieu chuan": "data engineer",

    # =====================
    # DATA SCIENTIST
    # =====================
    "data scientist": "data scientist",
    "senior data scientist": "data scientist",
    "data scientist expert": "data scientist",
    "nha khoa hoc du lieu": "data scientist",
    "cvcc khoa hoc du lieu": "data scientist",
    "chuyen vien phat trien khoa hoc du lieu": "data scientist",
    "chuyen vien mo hinh va phan tich nang cao": "data scientist",
    "chuyen vien cao cap mo hinh hoa va phan tich nang cao": "data scientist",
    "chuyen vien chuyen vien cao cap khoa hoc du lieu": "data scientist",
    "chuyen vien phat trien khoa hoc du lieu data scientist": "data scientist",
    "data science intern": "data scientist",
    "data scientist python llm ai machine learning": "data scientist",

    # =====================
    # AI ENGINEER
    # =====================
    "ai engineer": "ai engineer",
    "ai engineer intern": "ai engineer",
    "ctv ai engineer": "ai engineer",
    "ai developer": "ai engineer",
    "ai software engineer": "ai engineer",
    "ai software engineer ky su phan mem ai": "ai engineer",
    "ai engineering": "ai engineer",
    "ai generative engineer": "ai engineer",
    "ai machine learning engineer": "ai engineer",
    "machine learning engineer": "ai engineer",
    "ml engineer": "ai engineer",
    "agentic engineer": "ai engineer",
    "ai agent engineer": "ai engineer",
    "edge ai engineer": "ai engineer",
    "edge ai engineer leader": "ai engineer",
    "ai platform engineer": "ai engineer",
    "ai platform architect": "ai engineer",
    "ai system engineer": "ai engineer",
    "nhan vien ai system engineer": "ai engineer",
    "ky su ai": "ai engineer",
    "ky su tri tue nhan tao": "ai engineer",
    "ky su tri tue nhan tao ai": "ai engineer",
    "ky su ai ai engineer": "ai engineer",
    "ky su phat trien ai ai engineer": "ai engineer",
    "chuyen vien tri tue nhan tao": "ai engineer",
    "chuyen vien tri tue nhan tao ai": "ai engineer",
    "chuyen vien ai": "ai engineer",
    "chuyen vien ai ai engineer": "ai engineer",
    "lap trinh vien ai": "ai engineer",
    "fresher ai": "ai engineer",
    "fresher ai nlp": "ai engineer",
    "fresher ai computer vision engineer": "ai engineer",
    "aiops specialist": "ai engineer",
    "prompt engineer": "ai engineer",
    "software engineer prompt engineering": "ai engineer",
    "ai engineer computer vision npl llm": "ai engineer",
    "ai engineer nlp": "ai engineer",
    "ai engineer llm": "ai engineer",
    "ai engineer generative ai llm": "ai engineer",
    "ai engineer rag langchain python n8n": "ai engineer",
    "ai engineer python microservices docker": "ai engineer",
    "ai engineer healthcare ai product": "ai engineer",
    "ai engineer ky su ai": "ai engineer",
    "ai native software engineering lead": "ai engineer",
    "mid level ai powered cross platform engineer": "ai engineer",
    "kỹ sư cloud ai": "ai engineer",
    "chuyen vien lap trinh iot ai": "ai engineer",
    "chuyen vien phat trien ai app tu dong hoa": "ai engineer",
    "chuyen vien phat trien du an ai": "ai engineer",
    "nhan vien phat trien giai phap ai marketing": "ai engineer",
    "product & ai automation intern": "ai engineer",
    "ai and automation intern": "ai engineer",
    "mb trainee ai engineer": "ai engineer",
    "on job training ai": "ai engineer",
    "intern ai solution engineer": "ai engineer",

    # =====================
    # AI RESEARCHER
    # =====================
    "ai researcher": "ai researcher",
    "ai research engineer": "ai researcher",
    "senior ai research engineer": "ai researcher",
    "nlp research engineer": "ai researcher",
    "ai research intern": "ai researcher",
    "ai quantitative researcher intern": "ai researcher",
    "nhan vien nghien cuu al ai researcher": "ai researcher",
    "thuc tap sinh nghien cuu ung dung ai": "ai researcher",

    # =====================
    # DATA LABELING
    # =====================
    "data labeling specialist": "data labeling specialist",
    "data labeling coordinator": "data labeling specialist",
    "dieu phoi du an label data tieng anh": "data labeling specialist",
    "thuc tap sinh xu ly du lieu": "data labeling specialist",
    "data processing analyst": "data labeling specialist",
    "data processing specialist": "data labeling specialist",
    "chuyen vien xu ly du lieu": "data labeling specialist",
    "nhan vien xu ly du lieu": "data labeling specialist",
    "nhan vien xu ly du lieu team scan kols shopee": "data labeling specialist",
    "data entry specialist": "data labeling specialist",
    "nhan vien nhap lieu xu ly du lieu": "data labeling specialist",
    "nhan vien nhap va xu ly du lieu tieng nhat": "data labeling specialist",
    "nhan vien gan nhan du lieu tieng nhat": "data labeling specialist",
    "nhan vien ngon ngu du lieu tieng anh": "data labeling specialist",
    "nhan vien ngon ngu du lieu tieng trung": "data labeling specialist",
    "nhan vien ngon ngu du lieu tieng phap": "data labeling specialist",
    "nhan vien ngon ngu du lieu tieng tay ban nha bo dao nha thai": "data labeling specialist",
    "language data specialist": "data labeling specialist",
    "cong tac vien xu ly du lieu hinh anh lam fulltime offline": "data labeling specialist",

    # =====================
    # SOFTWARE ENGINEER
    # =====================
    "software engineer": "software engineer",
    "java software engineer": "software engineer",
    "bridge software engineer": "software engineer",
    "lead software engineer desktop": "software engineer",
    "middle software engineer java net frontend": "software engineer",
    "lap trinh vien phat trien phan mem software engineer": "software engineer",
    "laup trinh vien software engineer it": "software engineer",
    "ky su giai phap phan mem software developer": "software engineer",
    "ky su phat trien phan mem": "software engineer",
    "chuyen vien phat trien phan mem": "software engineer",
    "chuyen vien phat trien ung dung": "software engineer",
    "nhan vien software developer backend dev": "software engineer",
    "nhan vien software developer froned dev": "software engineer",
    "nhan vien trien khai phan mem mes software engineer": "software engineer",

    # =====================
    # BACKEND DEVELOPER
    # =====================
    "backend developer": "backend developer",
    "backend engineer": "backend developer",
    "backend lead engineer": "backend developer",
    "backend technical lead": "backend developer",
    "backend middle developer": "backend developer",
    "middle backend developer": "backend developer",
    "middle back end engineer": "backend developer",
    "middle senior backend engineer": "backend developer",
    "nhan vien backend": "backend developer",
    "nhan vien backend engineer developer": "backend developer",
    "nhan vien lap trinh backend": "backend developer",
    "lap trinh vien backend": "backend developer",
    "lap trinh vien backend developer java": "backend developer",
    "lap trinh vien backend python": "backend developer",
    "lap trinh vien backend net java nodejs php": "backend developer",
    "lap trinh vien back end developer": "backend developer",
    "java developer": "backend developer",
    "java engineer": "backend developer",
    "java backend": "backend developer",
    "java backend developer": "backend developer",
    "java back end developer": "backend developer",
    "java backend senior tech lead": "backend developer",
    "middle java backend": "backend developer",
    "middle java backend developer": "backend developer",
    "java developer backend engineer": "backend developer",
    "python developer": "backend developer",
    "junior python developer": "backend developer",
    "nodejs developer": "backend developer",
    "middle node js developer": "backend developer",
    "php developer": "backend developer",
    "php developer laravel": "backend developer",
    "php backend developer": "backend developer",
    "php teamleader": "backend developer",
    "golang developer": "backend developer",
    "golang backend": "backend developer",
    "lap trinh vien golang": "backend developer",
    ".net developer": "backend developer",
    ". net developer": "backend developer",
    "dotnet developer": "backend developer",
    "senior .net developer": "backend developer",
    "junior .net c#": "backend developer",
    "lap trinh vien net": "backend developer",
    "lap trinh vien .net": "backend developer",
    "lap trinh vien .net back end": "backend developer",
    "lap trinh vien .net leader": "backend developer",
    "nhan vien lap trinh .net": "backend developer",
    "nhan vien lap trinh vien backend php": "backend developer",
    "back end developer": "backend developer",
    "backend developer spring boot golang": "backend developer",
    "backend developer golang": "backend developer",
    "backend developer java": "backend developer",
    "backend developer javascript python": "backend developer",
    "backend developer nodejs": "backend developer",
    "backend developer python dev": "backend developer",
    "backend developer asp net core": "backend developer",
    "backend developer .net": "backend developer",
    "backend engineer .net java": "backend developer",

    # =====================
    # FRONTEND DEVELOPER
    # =====================
    "frontend developer": "frontend developer",
    "frontend engineer": "frontend developer",
    "front end developer": "frontend developer",
    "front end engineer": "frontend developer",
    "front end lead": "frontend developer",
    "front end team lead": "frontend developer",
    "junior frontend developer": "frontend developer",
    "middle frontend developer": "frontend developer",
    "middle senior frontend engineer": "frontend developer",
    "react native developer": "frontend developer",
    "reactjs developer": "frontend developer",
    "frontend reactjs developer": "frontend developer",
    "frontend reactjs nextjs": "frontend developer",
    "react native": "frontend developer",
    "angular developer": "frontend developer",
    "chuyen vien lap trinh angular": "frontend developer",
    "lap trinh vien angular": "frontend developer",
    "vuejs developer": "frontend developer",
    "frontend developer nextjs": "frontend developer",
    "lap trinh vien frontend": "frontend developer",
    "lap trinh vien giao dien frontend developer": "frontend developer",
    "lap trinh vien web front end reactjs nextjs": "frontend developer",
    "nhan vien lap trinh frontend": "frontend developer",
    "nhan vien phat trien phan mem frontend developer": "frontend developer",
    "nhan vien phan mem frontend": "frontend developer",

    # =====================
    # FULLSTACK DEVELOPER
    # =====================
    "fullstack developer": "fullstack developer",
    "full stack developer": "fullstack developer",
    "full stack engineer": "fullstack developer",
    "fullstack engineer": "fullstack developer",
    "fullstack software engineer": "fullstack developer",
    "dev nodejs fullstack": "fullstack developer",
    "fullstack java developer": "fullstack developer",
    "java fullstack developer": "fullstack developer",
    "fullstack c# .net": "fullstack developer",
    "talent developer fullstack": "fullstack developer",
    "middle full stack developer": "fullstack developer",
    "middle fullstack developer": "fullstack developer",
    "middle senior full stack developer python": "fullstack developer",
    "nhan vien lap trinh full stack developer": "fullstack developer",
    "nhan vien lap trinh fullstack developer": "fullstack developer",
    "nhan vien lap trinh fullstack": "fullstack developer",
    "lap trinh vien fullstack": "fullstack developer",
    "lap trinh vien full stack": "fullstack developer",
    "lap trinh vien full stack desktop web mobile": "fullstack developer",
    "lap trinh fullstack": "fullstack developer",
    "chuyen vien phat trien phan mem fullstack": "fullstack developer",
    "chuyen vien phat trien phan mem fullstack": "fullstack developer",
    "chuyen vien phat trien phan mem fullstack": "fullstack developer",

    # =====================
    # CLOUD ENGINEER
    # =====================
    "cloud engineer": "cloud engineer",
    "junior cloud engineer": "cloud engineer",
    "aws cloud lead": "cloud engineer",
    "azure api engineer": "cloud engineer",
    "cloud support engineer": "cloud engineer",
    "post sales system & cloud engineer": "cloud engineer",
    "ky su giai phap cloud": "cloud engineer",
    "ky su giai phap dien toan dam may senior cloud engineer": "cloud engineer",
    "ky su he thong cloud engineer": "cloud engineer",
    "it infrastructure system network & cloud": "cloud engineer",
    "cloud engineer aws": "cloud engineer",
    "cloud engineer aws oci azure": "cloud engineer",
    "cloud engineer storage": "cloud engineer",
    "data center storage engineer": "cloud engineer",

    # =====================
    # DATABASE ADMINISTRATOR
    # =====================
    "database administrator": "database administrator",
    "database administrator dba": "database administrator",
    "senior database administrator": "database administrator",
    "dba": "database administrator",
    "quan tri co so du lieu": "database administrator",
    "nhan vien quan tri co so du lieu dba": "database administrator",
    "nhan su quan tri co so du lieu dba": "database administrator",
    "database engineer dba": "database administrator",

    # =====================
    # AUTOMATION TESTER
    # =====================
    "automation tester": "automation tester",
    "automation test engineer": "automation tester",
    "automation test": "automation tester",
    "automation testing": "automation tester",
    "performance tester": "automation tester",
    "manual tester": "automation tester",
    "manual tester junior": "automation tester",
    "manual tester intern": "automation tester",
    "senior tester": "automation tester",
    "middle tester": "automation tester",
    "junior tester": "automation tester",
    "tester": "automation tester",
    "tester product": "automation tester",
    "app mobile tester": "automation tester",
    "intern tester": "automation tester",
    "intern tester tieng nhat": "automation tester",
    "qc tester": "automation tester",
    "uat tester": "automation tester",
    "qa tester": "automation tester",
    "software tester": "automation tester",
    "nhan vien tester": "automation tester",
    "nhan vien tester mobile app": "automation tester",
    "nhan vien kiem thu phan mem": "automation tester",
    "nhan vien kiem thu phan mem software qa": "automation tester",
    "nhan vien kiem thu tester": "automation tester",
    "chuyen vien kiem thu phan mem": "automation tester",
    "chuyen vien kiem thu phan mem auto": "automation tester",
    "chuyen vien kiem thu phan mem manual tester agile": "automation tester",
    "chuyen vien kiem thu phan mem tester": "automation tester",
    "chuyen vien kiem tra chat luong phan mem tester": "automation tester",
    "kiem thu phan mem": "automation tester",
    "kiem thu phan mem tester": "automation tester",
    "ai augmented quality engineer tester": "automation tester",
    "ai quality assurance automation focus": "automation tester",
    "junior qc japanese n4 kiem thu phan mem": "automation tester",

    # =====================
    # DEVOPS ENGINEER
    # =====================
    "devops engineer": "devops engineer",
    "devops": "devops engineer",
    "devops cloud engineer": "devops engineer",
    "senior devops engineer": "devops engineer",
    "devops engineer senior": "devops engineer",
    "devops engineer site reliability engineer": "devops engineer",
    "devops engineer sysops engineer": "devops engineer",
    "devops engineer kubernetes cloud": "devops engineer",
    "devops engineer azure": "devops engineer",
    "devops engineer middle": "devops engineer",
    "devops engineer junior": "devops engineer",
    "fresher devops": "devops engineer",
    "thuc tap sinh devops": "devops engineer",
    "ky su devops": "devops engineer",
    "ky su van hanh va phat trien he thong devops": "devops engineer",
    "ky su phat trien van hanh he thong devops": "devops engineer",
    "nhan vien van hanh ung dung devops": "devops engineer",
    "cloud devops engineer": "devops engineer",
    "cloud devops engineer cds": "devops engineer",
    "devsecops engineer": "devops engineer",
    "appsec devops devsecops engineer": "devops engineer",
    "site reliability engineer sre trong devops team": "devops engineer",
    "chuyen vien sre": "devops engineer",

    # =====================
    # IOT ENGINEER
    # =====================
    "iot engineer": "iot engineer",
    "ky su iot": "iot engineer",
    "ki su iot": "iot engineer",
    "ky su iot iot engineer": "iot engineer",
    "chuyen vien iot iot platform digital twin": "iot engineer",
    "chuyen vien ky thuat iot": "iot engineer",
    "nhan vien ky thuat iot": "iot engineer",
    "nhan vien thiet bi iot": "iot engineer",
    "chuyen vien ky thuat thiet bi dien thoai iot": "iot engineer",
    "ky su tich hop he thong iot va lap trinh nhung": "iot engineer",
    "ky su nhung embedded iot": "iot engineer",
    "ky su nhung iot": "iot engineer",

    # =====================
    # EMBEDDED ENGINEER
    # =====================
    "embedded engineer": "embedded engineer",
    "embedded developer": "embedded engineer",
    "embedded software engineer": "embedded engineer",
    "embedded software development": "embedded engineer",
    "embedded linux engineer": "embedded engineer",
    "embedded systems engineer": "embedded engineer",
    "embedded android developer": "embedded engineer",
    "fresher embedded developer": "embedded engineer",
    "firmware engineer": "embedded engineer",
    "embedded software engineer lead": "embedded engineer",
    "embedded software verification and validation": "embedded engineer",
    "ky su he thong nhung": "embedded engineer",
    "ky su lap trinh nhung": "embedded engineer",
    "ky su lap trinh embedded linux": "embedded engineer",
    "ky su lap trinh firmware": "embedded engineer",
    "ky su phan mem nhung": "embedded engineer",
    "lap trinh nhung": "embedded engineer",
    "lap trinh phan mem nhung": "embedded engineer",
    "lap trinh vien nhung embedded": "embedded engineer",
    "embedded engineer lap trinh nhung": "embedded engineer",
    "ky su chuyen vien phan mem nhung": "embedded engineer",

    # =====================
    # PRODUCT ANALYST
    # =====================
    "product analyst": "product analyst",
    "product analyst research": "product analyst",
    "data product analyst": "product analyst",
    "product executive": "product analyst",
    "product development executive": "product analyst",
    "product management executive": "product analyst",
    "product management specialist": "product analyst",
    "product operator": "product analyst",
    "product operations executive": "product analyst",
    "assistant product manager": "product analyst",
    "business product lead": "product analyst",

    # =====================
    # BUSINESS ANALYST
    # =====================
    "business analyst": "business analyst",
    "it business analyst": "business analyst",
    "junior business analyst": "business analyst",
    "middle business analyst": "business analyst",
    "senior business analyst": "business analyst",
    "business analyst leader": "business analyst",
    "business analyst lead": "business analyst",
    "business analyst intern": "business analyst",
    "business analyst internship": "business analyst",
    "business analyst fresher": "business analyst",
    "business analyst data": "business analyst",
    "data business analyst": "business analyst",
    "erp business analyst": "business analyst",
    "ba business analyst": "business analyst",
    "chuyen vien business analyst": "business analyst",
    "chuyen vien phan tich nghiep vu": "business analyst",
    "nhan vien phan tich nghiep vu": "business analyst",
    "phan tich nghiep vu": "business analyst",
    "phan tich nghiep vu business analyst": "business analyst",
    "chuyen vien phan tich nghiep vu business analyst": "business analyst",
    "chuyen vien phan tich nghiep vu ba": "business analyst",
    "nhan vien phan tich nghiep vu ba": "business analyst",
    "business analyst chuyen vien phan tich nghiep vu": "business analyst",
    "business analyst ba": "business analyst",
    "nhan vien business analyst": "business analyst",
    "nhan vien ba": "business analyst",
    "nhan vien ba business analyst": "business analyst",
    "thuc tap sinh phan tich nghiep vu": "business analyst",
    "thuc tap sinh phan tich nghiep vu business analyst intern": "business analyst",
    "junior ba phan tich yeu cau quy trinh nghiep vu": "business analyst",
    "leader ba it": "business analyst",
    "chuyen vien phan tich he thong nghiep vu": "business analyst",
    "business analyst crm": "business analyst",
    "business analyst erp lead": "business analyst",
    "chuyen vien ba erp tu van trien khai erp phan mem ke toan": "business analyst",
    "chuyen vien phan tich nghiep vu odoo": "business analyst",
    "business analyst banking fintech": "business analyst",
    "business analyst domain bank fintech": "business analyst",

    # =====================
    # PRODUCT OWNER / PRODUCT MANAGER
    # =====================
    "product owner": "product manager",
    "product manager": "product manager",
    "nhan vien product owner": "product manager",
    "product owner po": "product manager",
    "product owner product manager": "product manager",
    "product manager product owner": "product manager",
    "product executive product owner": "product manager",
    "giam doc san pham product manager": "product manager",
    "chuyen gia phat trien san pham po": "product manager",
    "chuyen gia san pham": "product manager",
    "chuyen gia phat trien san pham so khcn product owner": "product manager",
    "fresher product owner": "product manager",
    "intern product owner": "product manager",
    "product owner intern": "product manager",
    "game product owner": "product manager",
    "product owner mobile app": "product manager",
    "product owner rpa product": "product manager",
    "product owner edtech ai game": "product manager",
    "product owner fintech": "product manager",
    "product owner expert": "product manager",
    "product manager ecommerce": "product manager",
    "product manager erp logistics wms": "product manager",
    "product manager mobile app": "product manager",
    "product manager technical solution": "product manager",
}

TITLE_CANONICAL_KEYWORDS = {
    "data analyst": [
        "data analyst", "business data analyst", "bi analyst", "fp&a analyst", "hr data analysis",
        "credit risk analytics", "fraud risk data analytics", "phan tich du lieu",
    ],
    "data engineer": [
        "data engineer", "data integration engineer", "data warehouse engineer", "data platform engineer",
        "big data", "data crawler engineer", "ky su du lieu",
    ],
    "data scientist": [
        "data scientist", "nha khoa hoc du lieu", "khoa hoc du lieu",
    ],
    "ai engineer": [
        "ai engineer", "machine learning engineer", "ml engineer", "ai developer", "agentic engineer",
        "computer vision", "nlp engineer", "ai platform", "prompt engineer",
    ],
    "ai researcher": [
        "ai researcher", "ai research", "research engineer", "nlp research",
    ],
    "data labeling specialist": [
        "data labeling", "label data", "gan nhan du lieu", "xu ly du lieu", "ngon ngu du lieu", "nhap lieu",
    ],
    "software engineer": [
        "software engineer", "software developer", "phat trien phan mem", "phat trien ung dung",
    ],
    "backend developer": [
        "backend developer", "backend engineer", "java developer", "java backend", "python developer",
        "nodejs developer", "php developer", "golang developer", ".net developer", "dotnet developer",
    ],
    "frontend developer": [
        "frontend developer", "frontend engineer", "angular developer", "reactjs developer",
        "react native developer", "vuejs developer", "front end developer",
    ],
    "fullstack developer": [
        "fullstack developer", "full stack developer", "fullstack engineer", "full stack engineer",
    ],
    "cloud engineer": [
        "cloud engineer", "aws cloud", "azure api engineer", "cloud support",
    ],
    "database administrator": [
        "database administrator", "dba", "quan tri co so du lieu",
    ],
    "automation tester": [
        "automation tester", "automation test", "performance tester", "manual tester", "tester",
        "software tester", "qa tester", "uat tester", "kiem thu phan mem",
    ],
    "devops engineer": [
        "devops engineer", "devsecops engineer", "site reliability engineer", "sre", "sysops",
    ],
    "iot engineer": [
        "iot engineer", "ky su iot", "ki su iot", "thiet bi iot",
    ],
    "embedded engineer": [
        "embedded engineer", "embedded developer", "embedded software", "firmware engineer",
        "lap trinh nhung", "he thong nhung",
    ],
    "product analyst": [
        "product analyst", "product executive", "product operations", "assistant product manager",
    ],
    "business analyst": [
        "business analyst", "phan tich nghiep vu", "ba", "it business analyst", "erp business analyst",
    ],
    "product manager": [
        "product owner", "product manager", "po",
    ],
}

JOB_FAMILY_RULES = {
    "data_analytics": [
        "data analyst", "product analyst",
    ],
    "data_engineering": [
        "data engineer", "database administrator",
    ],
    "data_science_ml": [
        "data scientist", "ai engineer", "ai researcher",
    ],
    "software_engineering": [
        "software engineer", "backend developer", "frontend developer", "fullstack developer",
        "embedded engineer", "iot engineer",
    ],
    "cloud_devops_sre": [
        "cloud engineer", "devops engineer",
    ],
    "qa_testing": [
        "automation tester",
    ],
    "product_project_ba": [
        "business analyst", "product manager", "product analyst",
    ],
    "data_governance_quality": [
        "data labeling specialist",
    ],
    "database_platform": [
        "database administrator",
    ],
}

JOB_FAMILY_DESCRIPTION_HINTS = {
    "data_analytics": [
        "dashboard", "report", "business intelligence", "power bi", "tableau", "sql", "kpi", "insight",
        "phan tich du lieu",
    ],
    "data_engineering": [
        "etl", "pipeline", "data pipeline", "data warehouse", "dwh", "airflow", "spark",
        "kafka", "hadoop", "lakehouse", "fabric", "batch", "streaming", "database administrator",
    ],
    "data_science_ml": [
        "machine learning", "deep learning", "nlp", "llm", "rag", "computer vision", "model training",
        "tensorflow", "pytorch", "generative ai", "genai", "feature engineering",
    ],
    "software_engineering": [
        "software engineer", "backend developer", "frontend developer", "fullstack developer",
        "backend", "frontend", "fullstack", "api", "microservice", "rest", "graphql",
        "react", "nodejs", "java", ".net", "web development", "wordpress",
        "embedded", "firmware", "iot",
    ],
    "cloud_devops_sre": [
        "cloud engineer", "devops engineer",
        "cloud", "aws", "azure", "gcp", "kubernetes", "docker", "terraform", "ci/cd",
        "monitoring", "observability", "infrastructure as code",
    ],
    "qa_testing": [
        "automation tester", "manual tester", "software tester",
        "automation test", "test automation", "qa", "quality assurance", "selenium", "cypress",
        "playwright", "regression test", "performance test",
    ],
    "product_project_ba": [
        "product analyst", "business analyst", "phan tich nghiep vu",
        "product owner", "product manager",
        "user story", "brd", "frd", "stakeholder", "scrum", "agile",
        "project management", "product roadmap", "backlog", "prioritization",
    ],
    "data_governance_quality": [
        "data governance", "data quality", "master data", "data steward", "label", "labeling",
        "gan nhan", "data validation", "metadata", "lineage", "xu ly du lieu", "nhap lieu",
    ],
    "database_platform": [
        "database administrator", "dba",
        "database", "sql server", "postgresql", "mysql", "query tuning", "index", "backup", "replication",
    ],
}

DISPLAY_REPLACEMENTS = [
    (r"\bBA\b", "Business Analyst"),
    (r"\bPO\b", "Product Owner"),
    (r"\bPM\b", "Project Manager"),
    (r"\bQA\b", "QA"),
    (r"\bQC\b", "QC"),
    (r"\bAI\b", "AI"),
    (r"\bNLP\b", "NLP"),
    (r"\bLLM\b", "LLM"),
    (r"\bRAG\b", "RAG"),
    (r"\bIoT\b", "IoT"),
    (r"\bDBA\b", "DBA"),
    (r"\bDevOps\b", "DevOps"),
    (r"\bNodejs\b", "NodeJS"),
    (r"\bReactjs\b", "ReactJS"),
    (r"\bVuejs\b", "VueJS"),
    (r"\bDotnet\b", ".NET"),
]

def strip_bracket_noise(text):
    text = safe_str(text)
    if not text:
        return ""

    matches_round = re.findall(r"\((.*?)\)", text)
    for m in matches_round:
        normalized = normalize_for_match(m)
        if normalized in BRACKET_NOISE_KEYWORDS:
            text = text.replace(f"({m})", " ")

    matches_square = re.findall(r"\[(.*?)\]", text)
    for m in matches_square:
        normalized = normalize_for_match(m)
        if normalized in BRACKET_NOISE_KEYWORDS:
            text = text.replace(f"[{m}]", " ")

    return re.sub(r"\s+", " ", text).strip()

def remove_internal_title_codes(text):
    text = safe_str(text)
    if not text:
        return ""
    text = re.sub(r"\b(?:TA\d+|HO\d{2}\.\d+|HOLT\.\d+|IT\d+)\b", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" -|_")
    return text

def split_title_segments(text):
    text = safe_str(text)
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"\s*-\s*", text) if p.strip()]
    return parts

def normalize_segment_for_dedup(segment):
    s = normalize_for_match(segment)
    s = re.sub(r"\b(junior|middle|senior|lead|leader|team lead|fresher|intern)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_noise_segment(segment):
    s = normalize_for_match(segment)
    if not s:
        return True
    for pat in TITLE_SEGMENT_NOISE_PATTERNS:
        if re.search(pat, s, flags=re.I):
            return True
    return False

def should_keep_bracket_content(content):
    s = normalize_for_match(content)
    if not s:
        return False

    # Drop purely noisy brackets
    if s in BRACKET_NOISE_KEYWORDS:
        return False

    # Keep meaningful technical/specialization cues
    keep_keywords = [
        "business analyst", "ba", "product owner", "product manager",
        "python", "java", "golang", "go", "node", "nodejs", "php", "ruby",
        "react", "reactjs", "react native", "angular", "vue", "vuejs", "nextjs",
        ".net", "net", "c#", "c++", "kotlin", "swift",
        "aws", "azure", "gcp", "cloud", "kubernetes", "docker",
        "nlp", "llm", "rag", "computer vision", "cv", "agentic", "generative ai",
        "automation", "manual tester", "tester", "dba", "mysql", "sql", "spring boot",
        "embedded", "firmware", "iot",
    ]
    return any(k in s for k in keep_keywords)

def clean_bracket_content(text):
    text = safe_str(text)
    if not text:
        return ""

    def _replace_round(match):
        content = match.group(1).strip()
        return f"({content})" if should_keep_bracket_content(content) else " "

    def _replace_square(match):
        content = match.group(1).strip()
        return f"[{content}]" if should_keep_bracket_content(content) else " "

    text = re.sub(r"\((.*?)\)", _replace_round, text)
    text = re.sub(r"\[(.*?)\]", _replace_square, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_title_global_noise(text):
    text = safe_str(text)
    if not text:
        return ""

    normalized = normalize_for_match(text)

    # remove common noisy long tails directly on raw text by patterns aligned to normalized content
    replacements = [
        (r"(?i)\b(?:lương|thu nhập|salary|upto|up to|offer)\b.*$", ""),
        (r"(?i)\b(?:đi làm ngay|ob sớm|urgent|hot)\b.*$", ""),
        (r"(?i)\b(?:từ|from)\s*\d+\s*năm.*$", ""),
        (r"(?i)\b\d+\s*năm\s*kinh\s*nghiệm.*$", ""),
        (r"(?i)\b(?:english required|good at english|tiếng anh tốt|n\d\+|jlpt.*)$", ""),
    ]
    for pat, repl in replacements:
        text = re.sub(pat, repl, text)

    text = re.sub(r"\s+", " ", text).strip(" -|_")
    return text

def deduplicate_segments(segments):
    deduped = []
    seen = set()
    for seg in segments:
        key = normalize_segment_for_dedup(seg)
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(seg.strip())
    return deduped

def normalize_display_title(text):
    text = safe_str(text)
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text).strip(" -|_")
    text = re.sub(r"\s*([/,+])\s*", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # mild title case only for obvious lowercase english tokens
    text = text.replace("BackEend", "Backend")
    text = text.replace("Devops", "DevOps")
    text = text.replace("FullStack", "Fullstack")
    text = text.replace("Full Stack", "Fullstack")
    text = text.replace("FrontEnd", "Frontend")
    text = text.replace("BackEnd", "Backend")
    text = text.replace("Ai ", "AI ")
    text = text.replace(" Ai", " AI")
    text = text.replace("DotNet", ".NET")
    text = text.replace(". Net", ".NET")
    text = text.replace(". net", ".NET")
    text = text.replace("Reactjs", "ReactJS")
    text = text.replace("Vuejs", "VueJS")
    text = text.replace("Nodejs", "NodeJS")
    text = text.replace("Golang", "Golang")
    text = text.replace("Npl", "NLP")

    for pat, repl in DISPLAY_REPLACEMENTS:
        text = re.sub(pat, repl, text)

    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s+", " ", text).strip(" -|_")
    return text

def normalize_title_surface(text):
    text = clean_text_light(text)
    if not text:
        return ""

    text = clean_title_global_noise(text)
    text = strip_bracket_noise(text)
    text = clean_bracket_content(text)
    text = remove_internal_title_codes(text)

    segments = split_title_segments(text)
    segments = [seg.strip() for seg in segments if seg.strip()]
    kept_segments = [seg for seg in segments if not is_noise_segment(seg)]

    if kept_segments:
        kept_segments = deduplicate_segments(kept_segments)
        text = " - ".join(kept_segments)
    else:
        text = text.strip()

    # remove repeated canonical phrases joined by hyphen
    text = re.sub(r"\s+", " ", text).strip(" -|_")
    text = normalize_display_title(text)
    return text

def canonicalize_title(surface_text):
    s = normalize_for_match(surface_text)
    if not s:
        return ""

    s = re.sub(r"\b(junior|middle|senior|lead|leader|team lead|fresher|intern)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # direct exact lookup
    if s in TITLE_SYNONYM_MAP:
        return TITLE_SYNONYM_MAP[s]

    # exact phrase containment prioritizing longest synonym key
    for k in sorted(TITLE_SYNONYM_MAP.keys(), key=len, reverse=True):
        if k == s:
            return TITLE_SYNONYM_MAP[k]

    for k in sorted(TITLE_SYNONYM_MAP.keys(), key=len, reverse=True):
        if k in s:
            return TITLE_SYNONYM_MAP[k]

    # fallback by broader canonical keyword rules
    for canonical, keywords in TITLE_CANONICAL_KEYWORDS.items():
        for kw in keywords:
            if kw in s:
                return canonical

    # sensible defaults for uncaptured cases
    if "backend" in s or "java developer" in s or "python developer" in s or "nodejs" in s or "php developer" in s or ".net" in s or "dotnet" in s:
        return "backend developer"
    if "frontend" in s or "reactjs" in s or "angular" in s or "vuejs" in s or "react native" in s:
        return "frontend developer"
    if "fullstack" in s or "full stack" in s:
        return "fullstack developer"
    if "devops" in s or "devsecops" in s or "sre" in s:
        return "devops engineer"
    if "tester" in s or "kiem thu" in s or "qa" in s or "qc" in s:
        return "automation tester"
    if "business analyst" in s or re.search(r"\bba\b", s):
        return "business analyst"
    if "product owner" in s or "product manager" in s:
        return "product manager"
    if "data analyst" in s or "phan tich du lieu" in s:
        return "data analyst"
    if "data engineer" in s or "du lieu" in s and "engineer" in s:
        return "data engineer"
    if "data scientist" in s or "khoa hoc du lieu" in s:
        return "data scientist"
    if "ai" in s and ("engineer" in s or "developer" in s):
        return "ai engineer"
    if "research" in s and "ai" in s:
        return "ai researcher"
    if "cloud" in s:
        return "cloud engineer"
    if "dba" in s or "database administrator" in s or "quan tri co so du lieu" in s:
        return "database administrator"
    if "embedded" in s or "firmware" in s or "lap trinh nhung" in s:
        return "embedded engineer"
    if "iot" in s:
        return "iot engineer"
    if "label" in s or "gan nhan" in s or "xu ly du lieu" in s or "ngon ngu du lieu" in s:
        return "data labeling specialist"

    return s

def normalize_job_title(text):
    surface = normalize_title_surface(text)
    canonical = canonicalize_title(surface)
    return canonical

def infer_job_family_from_title(job_title_canonical):
    t = safe_str(job_title_canonical).lower().strip()
    if not t:
        return "unknown"

    for family, keywords in JOB_FAMILY_RULES.items():
        if any(k in t for k in keywords):
            return family
    return "unknown"

def infer_job_family_from_description(text):
    t = normalize_for_match(text)
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


# =========================
# LOCATION
# =========================

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
    "tp hcm": "TP. Hồ Chí Minh",
    "tphcm": "TP. Hồ Chí Minh",
    "hcm": "TP. Hồ Chí Minh",
    "sài gòn": "TP. Hồ Chí Minh",
    "sai gon": "TP. Hồ Chí Minh",
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

CITY_LABEL_PATTERN = r"(Cao Bằng|Cao Bang|Sơn La|Son La|Lai Châu|Lai Chau|Lạng Sơn|Lang Son|Tuyên Quang|Tuyen Quang|Lào Cai|Lao Cai|Thái Nguyên|Thai Nguyen|Điện Biên|Dien Bien|Phú Thọ|Phu Tho|Bắc Ninh|Bac Ninh|Hà Nội|Ha Noi|Quảng Ninh|Quang Ninh|Hải Phòng|Hai Phong|Hưng Yên|Hung Yen|Ninh Bình|Ninh Binh|Thanh Hóa|Thanh Hoa|Nghệ An|Nghe An|Hà Tĩnh|Ha Tinh|Quảng Trị|Quang Tri|Huế|Hue|Đà Nẵng|Da Nang|Quảng Ngãi|Quang Ngai|Gia Lai|Đắk Lắk|Dak Lak|Khánh Hòa|Khanh Hoa|Lâm Đồng|Lam Dong|Đồng Nai|Dong Nai|Hồ Chí Minh|Ho Chi Minh|TP\.?\s*HCM|TPHCM|HCM|Sài Gòn|Sai Gon|Tây Ninh|Tay Ninh|Đồng Tháp|Dong Thap|Vĩnh Long|Vinh Long|Cần Thơ|Can Tho|An Giang|Cà Mau|Ca Mau)\s*:"

def detect_city_from_text(text):
    t = normalize_for_match(text)
    hits = []

    for alias, canonical in VIETNAM_CITY_ALIASES.items():
        alias_norm = normalize_for_match(alias)
        if alias_norm and re.search(rf"(?<!\w){re.escape(alias_norm)}(?!\w)", t):
            hits.append(canonical)

    hits = list(dict.fromkeys(hits))
    return ", ".join(hits) if hits else None

def has_multi_location(text):
    t = normalize_for_match(text)
    hits = set()

    for alias, canonical in VIETNAM_CITY_ALIASES.items():
        alias_norm = normalize_for_match(alias)
        if alias_norm and re.search(rf"(?<!\w){re.escape(alias_norm)}(?!\w)", t):
            hits.add(canonical)

    return len(hits) >= 2

def clean_working_address_text(raw_text):
    text = safe_str(raw_text)
    if not text:
        return ""

    text = text.replace("\\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # bỏ note editorial đầu dòng
    text = re.sub(
        r"^\(\s*đã được cập nhật theo danh mục hành chính mới.*?\)\s*[-:–•]*\s*",
        "",
        text,
        flags=re.I
    )

    # bỏ chú thích đơn vị hành chính cũ
    text = re.sub(
        r"\(\s*(quận|huyện|thị xã|thành phố|tp\.?)\s+[^)]*?\s+cũ\s*\)",
        "",
        text,
        flags=re.I
    )

    # bỏ "(Tất cả phường)"
    text = re.sub(
        r"\(\s*tất cả phường\s*\)",
        "",
        text,
        flags=re.I
    )

    # bỏ "...và N địa điểm khác Thu gọn"
    text = re.sub(
        r"\.\.\.\s*và\s*\d+\s*địa điểm khác\s*thu gọn",
        "",
        text,
        flags=re.I
    )

    # bỏ chuỗi "Thu gọn" còn sót
    text = re.sub(r"\bThu gọn\b", "", text, flags=re.I)

    # chuẩn bullet / dash đầu dòng
    text = re.sub(r"^\s*[-–•]+\s*", "", text)

    # chuẩn khoảng trắng
    text = re.sub(r"\s+", " ", text).strip(" -,:;")
    return text

def split_multiple_addresses(text):
    text = safe_str(text)
    if not text:
        return []

    matches = list(re.finditer(CITY_LABEL_PATTERN, text, flags=re.I))
    if not matches:
        return [text]

    parts = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip(" -|,;")
        if chunk:
            parts.append(chunk)

    return parts

def normalize_single_address_part(part):
    part = safe_str(part)
    if not part:
        return ""

    # bỏ lặp city label trong cùng 1 part nếu có
    part = re.sub(r"\s+", " ", part).strip(" -,:;")
    part = re.sub(r"\s*-\s*", " - ", part)
    part = re.sub(r"\s*,\s*", ", ", part)

    # bỏ các cụm vô nghĩa còn sót
    part = re.sub(r"\(\s*tất cả phường\s*\)", "", part, flags=re.I)
    part = re.sub(r"\bThu gọn\b", "", part, flags=re.I)

    # chuẩn lại khoảng trắng
    part = re.sub(r"\s+", " ", part).strip(" -,:;")
    return part

def is_meaningful_address_part(part):
    p = normalize_for_match(part)
    if not p:
        return False

    # loại part quá rỗng / vô nghĩa
    if p in {"tat ca phuong"}:
        return False

    # loại part chỉ còn city label
    city_only_patterns = [
        r"^(cao bang|son la|lai chau|lang son|tuyen quang|lao cai|thai nguyen|dien bien|phu tho|bac ninh|ha noi|quang ninh|hai phong|hung yen|ninh binh|thanh hoa|nghe an|ha tinh|quang tri|hue|da nang|quang ngai|gia lai|dak lak|khanh hoa|lam dong|dong nai|ho chi minh|tp hcm|tphcm|hcm|sai gon|tay ninh|dong thap|vinh long|can tho|an giang|ca mau)\s*:?$"
    ]
    if any(re.search(pat, p) for pat in city_only_patterns):
        return False

    return True

def deduplicate_preserve_order(items):
    seen = set()
    results = []
    for item in items:
        key = normalize_for_match(item)
        if not key or key in seen:
            continue
        seen.add(key)
        results.append(item)
    return results

def parse_working_address(raw_text):
    cleaned = clean_working_address_text(raw_text)
    parts = split_multiple_addresses(cleaned)
    parts = [normalize_single_address_part(p) for p in parts]
    parts = [p for p in parts if is_meaningful_address_part(p)]
    parts = deduplicate_preserve_order(parts)

    location_city = detect_city_from_text(raw_text)

    return {
        "working_address_clean_list": parts,
        "working_address_clean": " | ".join(parts) if parts else "",
        "location_city": location_city,
        "is_multi_location": len(parts) >= 2 or has_multi_location(raw_text),
    }

# =========================
# WORK MODE
# =========================

WORK_MODE_RULES = {
    "hybrid": [
        r"\bhybrid\b",
        r"\bhybird\b",
        r"\bhybrid working\b",
        r"\blinh hoat\b",
        r"\bket hop onsite va remote\b",
        r"\bket hop online va offline\b",
        r"\bket hop truc tiep va tu xa\b",
        r"\bremote \d+ days/month\b",
        r"\bremote \d+ ngay/tu[a-z]*\b",
        r"\b[0-9]+\s*days/month\b",
        r"\b[0-9]+\s*ngay/tu[a-z]*\b",
    ],
    "remote": [
        r"\bremote\b",
        r"\bwfh\b",
        r"\bwork from home\b",
        r"\bfully remote\b",
        r"\blam viec tu xa\b",
        r"\blam viec online\b",
        r"\blam online\b",
        r"\blam viec tai nha\b",
        r"\btai nha\b",
    ],
    "onsite": [
        r"\bonsite\b",
        r"\bon site\b",
        r"\bwork onsite\b",
        r"\bwork on site\b",
        r"\bin office\b",
        r"\btai van phong\b",
        r"\blam viec tai cong ty\b",
        r"\blam viec truc tiep\b",
        r"\blam viec co dinh tai vp\b",
    ],
}

def infer_work_mode(*texts):
    merged = " ".join([safe_str(t) for t in texts if safe_str(t)])
    merged_norm = normalize_for_match(merged)

    if not merged_norm:
        return "unknown"

    for mode in ["hybrid", "remote", "onsite"]:
        for p in WORK_MODE_RULES[mode]:
            if re.search(p, merged_norm, flags=re.I):
                return mode

    return "unknown"

def infer_work_mode_with_source(*texts):
    merged = " ".join([safe_str(t) for t in texts if safe_str(t)])
    merged_norm = normalize_for_match(merged)

    if not merged_norm:
        return "unknown", "none"

    for mode in ["hybrid", "remote", "onsite"]:
        for p in WORK_MODE_RULES[mode]:
            if re.search(p, merged_norm, flags=re.I):
                return mode, p

    return "unknown", "none"

# =========================
# DEADLINE
# =========================

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

    m = re.search(r"(\d+)\s*ngay", normalize_for_match(text))
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

# =========================
# SALARY
# =========================

def clean_salary(raw):
    text = clean_text_light(raw).lower()
    text = text.replace("thoả", "thỏa")
    text = re.sub(r"\s+", " ", text).strip()
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


def detect_salary_multiplier(text):
    t = safe_str(text).lower()

    # USD giữ nguyên base number, sẽ quy đổi sau
    if "usd" in t or "$" in t:
        return 1.0, "usd"

    if "triệu" in t or "trieu" in t or re.search(r"\b\d+(?:[.,]\d+)?\s*(tr|trieu)\b", t):
        return 1_000_000, "vnd"

    if "nghìn" in t or "ngan" in t or re.search(r"\b\d+(?:[.,]\d+)?\s*k\b", t):
        return 1_000, "vnd"

    return 1.0, "vnd"


def make_salary_bucket(min_vnd, max_vnd, salary_type, negotiable):
    if negotiable:
        return "negotiable"

    ref = max_vnd if max_vnd is not None else min_vnd
    if ref is None:
        return "unknown"

    if ref < 10_000_000:
        return "under_10m"
    if ref < 20_000_000:
        return "10m_20m"
    if ref < 30_000_000:
        return "20m_30m"
    if ref < 50_000_000:
        return "30m_50m"
    return "50m_plus"


def parse_salary_range(raw):
    text = clean_salary(raw)
    if not text:
        return {
            "salary_clean": "",
            "salary_type": "unknown",
            "salary_is_negotiable": None,
            "salary_min_vnd_month": None,
            "salary_max_vnd_month": None,
            "salary_bucket": "unknown",
        }

    negotiable = ("thỏa thuận" in text) or ("negotiable" in text)

    nums = re.findall(r"\d+(?:[.,]\d+)?", text)
    nums = [parse_numeric_token(x) for x in nums]
    nums = [x for x in nums if x is not None]

    multiplier, currency = detect_salary_multiplier(text)

    salary_min = None
    salary_max = None
    salary_type = "unknown"

    if negotiable and not nums:
        salary_type = "negotiable"

    elif "up to" in text or "upto" in text or "tối đa" in text:
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

    # Quy đổi toàn bộ về VND/tháng để filter thống nhất
    if currency == "usd":
        fx = 25000
        if salary_min is not None:
            salary_min *= fx
        if salary_max is not None:
            salary_max *= fx

    salary_bucket = make_salary_bucket(
        salary_min,
        salary_max,
        salary_type,
        negotiable
    )

    return {
        "salary_clean": text,
        "salary_type": salary_type,
        "salary_is_negotiable": negotiable,
        "salary_min_vnd_month": salary_min,
        "salary_max_vnd_month": salary_max,
        "salary_bucket": salary_bucket,
    }

# =========================
# EXPERIENCE + EDUCATION + EMPLOYMENT TYPE + JOB LEVEL
# =========================

def clean_experience(text):
    return normalize_for_match(text)


def _to_years(value, unit):
    if value is None:
        return None
    unit = safe_str(unit).lower()
    if "thang" in unit:
        return value / 12.0
    return value


def parse_experience_range(raw):
    text = clean_experience(raw)
    if not text:
        return {
            "experience_clean": "",
            "experience_min_years": None,
            "experience_max_years": None,
            "experience_type": "unknown",
        }

    # no experience
    if any(kw in text for kw in [
        "khong yeu cau kinh nghiem",
        "khong can kinh nghiem",
        "khong yeu cau",
        "no experience",
        "khong co kinh nghiem",
    ]):
        return {
            "experience_clean": text,
            "experience_min_years": 0.0,
            "experience_max_years": 0.0,
            "experience_type": "no_experience",
        }

    # range with explicit units, e.g. "6 thang - 1 nam", "tu 2 den 3 nam"
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*(thang|nam)\s*(?:-|den|toi)\s*(\d+(?:\.\d+)?)\s*(thang|nam)",
        text
    )
    if m:
        v1 = float(m.group(1))
        u1 = m.group(2)
        v2 = float(m.group(3))
        u2 = m.group(4)
        y1 = _to_years(v1, u1)
        y2 = _to_years(v2, u2)
        return {
            "experience_clean": text,
            "experience_min_years": min(y1, y2),
            "experience_max_years": max(y1, y2),
            "experience_type": "range",
        }

    # minimum explicit unit: "tu 6 thang", "tren 2 nam", "it nhat 1 nam", "2+ nam"
    m = re.search(
        r"(?:tu|tren|hon|it nhat|toi thieu)?\s*(\d+(?:\.\d+)?)\s*(\+)?\s*(thang|nam)",
        text
    )
    if m:
        val = float(m.group(1))
        plus = m.group(2)
        unit = m.group(3)
        years = _to_years(val, unit)

        if any(x in text for x in ["tu ", "tren ", "hon ", "it nhat", "toi thieu"]) or plus:
            return {
                "experience_clean": text,
                "experience_min_years": years,
                "experience_max_years": None,
                "experience_type": "minimum",
            }

        if "duoi" in text:
            return {
                "experience_clean": text,
                "experience_min_years": 0.0,
                "experience_max_years": years,
                "experience_type": "maximum",
            }

        return {
            "experience_clean": text,
            "experience_min_years": years,
            "experience_max_years": years,
            "experience_type": "fixed",
        }

    # range without repeated unit at end: "2-3 nam", "2 den 3 nam"
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:-|den|toi)\s*(\d+(?:\.\d+)?)\s*(thang|nam)",
        text
    )
    if m:
        v1 = float(m.group(1))
        v2 = float(m.group(2))
        unit = m.group(3)
        y1 = _to_years(v1, unit)
        y2 = _to_years(v2, unit)
        return {
            "experience_clean": text,
            "experience_min_years": min(y1, y2),
            "experience_max_years": max(y1, y2),
            "experience_type": "range",
        }

    # fallback numeric parse
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    nums = [float(x) for x in nums] if nums else []

    if len(nums) >= 2:
        return {
            "experience_clean": text,
            "experience_min_years": min(nums[0], nums[1]),
            "experience_max_years": max(nums[0], nums[1]),
            "experience_type": "range",
        }

    if len(nums) == 1:
        val = nums[0]
        if any(x in text for x in ["tu", "+", "tren", "it nhat", "toi thieu", "tro len"]):
            return {
                "experience_clean": text,
                "experience_min_years": val,
                "experience_max_years": None,
                "experience_type": "minimum",
            }
        if "duoi" in text:
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
    "phd": ["tien si", "phd", "doctor", "doctoral"],
    "master": ["thac si", "master", "mba","cao hoc"],
    "bachelor": ["dai hoc", "cu nhan", "bachelor", "university"],
    "college": ["cao dang", "college","trung cap"],
    "high_school": ["trung hoc", "high school", "pho thong"],
}

EDUCATION_RANK = {
    "unknown": 0,
    "high_school": 1,
    "college": 2,
    "bachelor": 3,
    "master": 4,
    "phd": 5,
}

EMPLOYMENT_TYPE_MAP = {
    "full_time": [
        "toan thoi gian", "full time", "full-time", "fulltime", "chinh thuc"
    ],
    "part_time": [
        "ban thoi gian", "part time", "part-time", "parttime"
    ],
    "internship": [
        "thuc tap", "thuc tap sinh", "internship", "intern"
    ],
    "contract": [
        "hop dong", "contract"
    ],
    "freelance": [
        "freelance", "freelancer", "cong tac vien", "ctv", "cong tac"
    ],
    "temporary": [
        "temporary", "thoi vu", "ngan han"
    ],
}


def normalize_education_level(text):
    t = normalize_for_match(text)
    if not t:
        return "unknown"

    matched_levels = []
    for level, kws in EDUCATION_MAP.items():
        if any(k in t for k in kws):
            matched_levels.append(level)

    if not matched_levels:
        return "unknown"

    return max(matched_levels, key=lambda x: EDUCATION_RANK.get(x, 0))


def normalize_employment_type(text):
    t = normalize_for_match(text)
    if not t:
        return "unknown"

    for level, kws in EMPLOYMENT_TYPE_MAP.items():
        if any(k in t for k in kws):
            return level
    return "unknown"

# =========================
# JOB LEVEL (PRIORITY FILL)
# =========================

JOB_LEVEL_RULES = {
    "director": [
        r"\bdirector\b", r"\bhead\b", r"giam doc"
    ],
    "manager": [
        r"\bmanager\b", r"truong phong", r"pho phong", r"quan ly", r"giam sat"
    ],
    "lead": [
        r"\blead\b", r"\bleader\b", r"team lead", r"truong nhom"
    ],
    "senior": [
        r"\bsenior\b", r"\bsr\b", r"cao cap", r"chuyen vien cao cap"
    ],
    "middle": [
        r"\bmiddle\b", r"\bmid\b", r"mid senior", r"middle senior"
    ],
    "junior": [
        r"\bjunior\b", r"\bjr\b", r"nhan vien"
    ],
    "fresher": [
        r"\bfresher\b", r"moi tot nghiep"
    ],
    "intern": [
        r"\bintern\b", r"\binternship\b", r"thuc tap", r"thuc tap sinh"
    ],
}

JOB_LEVEL_PRIORITY = [
    "director", "manager", "lead", "senior", "middle", "junior", "fresher", "intern"
]

def extract_job_level_from_text(text):
    t = normalize_for_match(text)
    if not t:
        return "unknown"

    for lvl in JOB_LEVEL_PRIORITY:
        for p in JOB_LEVEL_RULES[lvl]:
            if re.search(p, t, flags=re.I):
                return lvl
    return "unknown"

def infer_job_level_from_experience(experience_min_years):
    if experience_min_years is None:
        return "unknown"

    if experience_min_years <= 0.5:
        return "fresher"
    if experience_min_years < 2.5:
        return "junior"
    if experience_min_years < 4:
        return "middle"
    return "senior"

def resolve_job_level(job_title_raw, experience_min_years, job_level_raw):
    # Priority 1: title
    lvl_from_title = extract_job_level_from_text(job_title_raw)
    if lvl_from_title != "unknown":
        return lvl_from_title, "job_title"

    # Priority 2: experience
    lvl_from_exp = infer_job_level_from_experience(experience_min_years)
    if lvl_from_exp != "unknown":
        return lvl_from_exp, "experience_min_years"

    # Priority 3: job_level_raw
    lvl_from_raw = extract_job_level_from_text(job_level_raw)
    if lvl_from_raw != "unknown":
        return lvl_from_raw, "job_level_raw"

    return "unknown", "unknown"

# =========================
# TAGS
# =========================

TAG_ROLE_MAP = {
    "backend developer": "backend developer",
    "back end developer": "backend developer",
    "backend engineer": "backend developer",
    "software engineer": "software engineer",
    "software developer": "software engineer",
    "business analyst": "business analyst",
    "business analyst phan tich nghiep vu": "business analyst",
    "phan tich nghiep vu": "business analyst",
    "software tester": "software tester",
    "tester": "software tester",
    "qa tester": "software tester",
    "fullstack developer": "fullstack developer",
    "full stack developer": "fullstack developer",
    "frontend developer": "frontend developer",
    "front end developer": "frontend developer",
    "data analyst": "data analyst",
    "data engineer": "data engineer",
    "data scientist": "data scientist",
    "ai engineer": "ai engineer",
    "product owner": "product owner",
    "product manager": "product manager",
    "devops engineer": "devops engineer",
    "cloud engineer": "cloud engineer",
    "database administrator": "database administrator",
    "dba": "database administrator",
    "embedded engineer": "embedded engineer",
    "iot engineer": "iot engineer",
}

TAG_DOMAIN_KEYWORDS = {
    "it - phan mem": "it phan mem",
    "it phan mem": "it phan mem",
    "phan mem": "it phan mem",
    "software": "it phan mem",
    "it - phan cung": "it phan cung",
    "phan cung": "it phan cung",
    "du lieu": "du lieu",
    "data": "du lieu",
    "ai": "ai ml",
    "machine learning": "ai ml",
    "ml": "ai ml",
    "tai chinh": "tai chinh ngan hang",
    "ngan hang": "tai chinh ngan hang",
    "banking": "tai chinh ngan hang",
    "finance": "tai chinh ngan hang",
    "thuong mai dien tu": "thuong mai dien tu",
    "ecommerce": "thuong mai dien tu",
    "e commerce": "thuong mai dien tu",
    "edtech": "edtech",
    "fintech": "fintech",
    "healthcare": "healthcare",
    "y te": "healthcare",
    "game": "game",
}

TAG_SPECIALTY_KEYWORDS = {
    "automation": "automation",
    "manual": "manual",
    "api": "api",
    "mobile": "mobile",
    "web": "web",
    "backend": "backend",
    "frontend": "frontend",
    "fullstack": "fullstack",
    "full stack": "fullstack",
    "cloud": "cloud",
    "devops": "devops",
    "data": "data",
    "ai": "ai",
    "machine learning": "machine learning",
    "ml": "machine learning",
    "nlp": "nlp",
    "llm": "llm",
    "rag": "rag",
    "computer vision": "computer vision",
    "cv": "computer vision",
    "embedded": "embedded",
    "iot": "iot",
    "database": "database",
    "dba": "database",
    "product": "product",
    "business analysis": "business analysis",
    "phan tich nghiep vu": "business analysis",
}

def split_tag_parts(text):
    text = clean_text_light(text)
    if not text:
        return []
    return [p.strip() for p in re.split(r"[;\n]+", text) if p.strip()]

def normalize_single_tag(part):
    part = clean_text_light(part)
    if not part:
        return ""
    part = re.sub(r"\s+", " ", part).strip(" -,:;/|")
    return part

def extract_parenthetical_text(tag_text):
    matches = re.findall(r"\((.*?)\)", safe_str(tag_text))
    return [clean_text_light(m) for m in matches if clean_text_light(m)]

def strip_parenthetical_text(tag_text):
    t = safe_str(tag_text)
    t = re.sub(r"\(.*?\)", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return clean_text_light(t)

def extract_role_from_tag(tag_text):
    candidates = []

    core = strip_parenthetical_text(tag_text)
    if core:
        candidates.append(core)

    candidates.extend(extract_parenthetical_text(tag_text))

    for cand in candidates:
        c = normalize_for_match(cand)
        for k, v in sorted(TAG_ROLE_MAP.items(), key=lambda x: len(x[0]), reverse=True):
            if k in c:
                return v
    return None

def extract_domain_from_tag(tag_text):
    t = normalize_for_match(tag_text)

    for kw, canonical in sorted(TAG_DOMAIN_KEYWORDS.items(), key=lambda x: len(x[0]), reverse=True):
        if kw in t:
            return canonical
    return None

def extract_specialties_from_tag(tag_text):
    found = []

    base_text = normalize_for_match(tag_text)
    parenthetical_text = " ".join(normalize_for_match(x) for x in extract_parenthetical_text(tag_text))
    merged = f"{base_text} {parenthetical_text}".strip()

    for kw, canonical in sorted(TAG_SPECIALTY_KEYWORDS.items(), key=lambda x: len(x[0]), reverse=True):
        if kw in merged:
            found.append(canonical)

    return deduplicate_list(found)

def normalize_tags_structured(text):
    parts = split_tag_parts(text)
    parts_clean = [normalize_single_tag(p) for p in parts]
    parts_clean = [p for p in parts_clean if p]
    parts_clean = deduplicate_list(parts_clean)

    roles = []
    domains = []
    specialties = []

    for p in parts_clean:
        role = extract_role_from_tag(p)
        if role:
            roles.append(role)

        domain = extract_domain_from_tag(p)
        if domain:
            domains.append(domain)

        specialties.extend(extract_specialties_from_tag(p))

    roles = deduplicate_list(roles)
    domains = deduplicate_list(domains)
    specialties = deduplicate_list(specialties)

    # text cho model: giữ dấu tiếng Việt từ tag clean gốc
    tags_text_for_model = ", ".join(parts_clean)

    # text canonical cho rule/debug/scoring
    tags_text_canonical = ", ".join(deduplicate_list(roles + specialties + domains))

    return {
        "tags_list": parts_clean,
        "tags_role_list": roles,
        "tags_domain_list": domains,
        "tags_specialty_list": specialties,
        "tags_text_for_model": clean_text_for_phobert(tags_text_for_model),
        "tags_text_canonical": tags_text_canonical,
    }

# =========================
# SKILL TAXONOMY
# =========================

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
    "r": {"aliases": ["r", "ngôn ngữ r"], "group": "programming"},
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
    alias = safe_str(alias).strip()
    if not alias:
        return None

    alias_norm = normalize_for_match(alias)

    # các alias quá ngắn cần boundary chặt hơn
    if alias_norm == "r":
        return r"(?<!\w)r(?!\w)"
    if alias_norm == "ml":
        return r"(?<!\w)ml(?!\w)"
    if alias_norm == "cv":
        return r"(?<!\w)cv(?!\w)"

    return r"(?<!\w)" + re.escape(alias_norm) + r"(?!\w)"


SKILL_PATTERNS = {}
for skill, meta in SKILL_TAXONOMY.items():
    pats = []
    for alias in meta["aliases"]:
        pat = alias_to_regex(alias)
        if pat:
            pats.append(re.compile(pat, flags=re.I))
    SKILL_PATTERNS[skill] = pats


def infer_skill_importance(segment, source_field):
    s = normalize_for_match(segment)

    if any(h in s for h in PREFERRED_HINTS):
        return "preferred"

    if source_field == "requirements":
        return "required"

    if any(h in s for h in REQUIRED_HINTS):
        return "required"

    return "mentioned"


def split_skill_segments(text):
    text = clean_text_preserve_structure(text)
    if not text:
        return []

    segments = [seg.strip() for seg in re.split(r"[\n•\-;]+", text) if seg.strip()]
    return segments


def extract_skill_records_from_text(text, source_field="unknown"):
    segments = split_skill_segments(text)
    if not segments:
        return []

    records = []

    for seg in segments:
        seg_norm = normalize_for_match(seg)

        for skill, patterns in SKILL_PATTERNS.items():
            if any(p.search(seg_norm) for p in patterns):
                records.append({
                    "skill": skill,
                    "skill_group": SKILL_TAXONOMY[skill]["group"],
                    "source_field": source_field,
                    "importance": infer_skill_importance(seg, source_field),
                    "excerpt": seg[:300],
                })

    # dedup theo skill + source + importance
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
        if importance_filter is not None and r.get("importance") != importance_filter:
            continue
        if r.get(key):
            vals.append(r[key])

    return deduplicate_list(vals)


def extract_job_skill_records(row):
    title_records = extract_skill_records_from_text(
        safe_str(row.get("job_title_surface")) + "\n" + safe_str(row.get("job_title_canonical")),
        source_field="title"
    )

    tag_records = extract_skill_records_from_text(
        safe_str(row.get("tags_text_canonical")),
        source_field="tags"
    )

    req_records = extract_skill_records_from_text(
        safe_str(row.get("requirements_clean_strict")),
        source_field="requirements"
    )

    desc_records = extract_skill_records_from_text(
        safe_str(row.get("description_clean_strict")),
        source_field="description"
    )

    benefit_records = extract_skill_records_from_text(
        safe_str(row.get("benefits_clean_strict")),
        source_field="benefits"
    )

    return merge_skill_records(
        title_records,
        tag_records,
        req_records,
        desc_records,
        benefit_records,
    )


def build_skill_text(xs):
    if not xs:
        return ""
    return clean_text_for_phobert(", ".join(xs))

# =========================
# FINAL JOB TEXT BUILDERS
# =========================

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


def format_experience_brief(row):
    mn = row.get("experience_min_years")
    mx = row.get("experience_max_years")
    exp_type = safe_str(row.get("experience_type"))

    if exp_type == "no_experience":
        return "không yêu cầu kinh nghiệm"

    if pd.notna(mn) and pd.notna(mx):
        if mn == mx:
            return f"{mn:g} năm"
        return f"{mn:g}-{mx:g} năm"

    if pd.notna(mn):
        return f"từ {mn:g} năm"

    if pd.notna(mx):
        return f"dưới {mx:g} năm"

    return ""


def build_job_text_sparse(row):
    parts = []

    for field in [
        row.get("job_title_canonical"),
        row.get("job_family"),
        row.get("job_level_norm"),
        row.get("employment_type_norm"),
        row.get("education_level_norm"),
        row.get("tags_text_canonical"),
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
    job_level = row.get("job_level_norm")
    employment_type = row.get("employment_type_norm")
    education_level = row.get("education_level_norm")
    required_skills = row.get("skills_required_text_phobert")
    preferred_skills = row.get("skills_preferred_text_phobert")
    tags_text = row.get("tags_text_for_model")
    exp_brief = format_experience_brief(row)
    location = row.get("location_city")
    work_mode = row.get("work_mode")
    req = truncate_by_words(row.get("requirements_clean_phobert"), 160)

    if title:
        parts.append(f"Vị trí: {title}")
    if family and family != "unknown":
        parts.append(f"Nhóm nghề: {family}")
    if job_level and job_level != "unknown":
        parts.append(f"Cấp độ công việc: {job_level}")
    if employment_type and employment_type != "unknown":
        parts.append(f"Hình thức tuyển dụng: {employment_type}")
    if education_level and education_level != "unknown":
        parts.append(f"Học vấn: {education_level}")
    if required_skills:
        parts.append(f"Kỹ năng bắt buộc: {required_skills}")
    if preferred_skills:
        parts.append(f"Kỹ năng ưu tiên: {preferred_skills}")
    if tags_text:
        parts.append(f"Tag công việc: {tags_text}")
    if exp_brief:
        parts.append(f"Kinh nghiệm: {exp_brief}")
    if location:
        parts.append(f"Địa điểm: {location}")
    if work_mode and work_mode != "unknown":
        parts.append(f"Hình thức làm việc: {work_mode}")
    if req:
        parts.append(f"Yêu cầu chính: {req}")

    return "\n".join(parts).strip()


def build_job_text_phobert_chatbot(row):
    parts = []

    title = row.get("job_title_for_phobert")
    family = row.get("job_family")
    job_level = row.get("job_level_norm")
    employment_type = row.get("employment_type_norm")
    education_level = row.get("education_level_norm")
    location = row.get("location_city")
    work_mode = row.get("work_mode")
    salary_brief = format_salary_brief(row)
    exp_brief = format_experience_brief(row)
    tags_text = row.get("tags_text_for_model")
    working_address = row.get("working_address_clean")
    required_skills = row.get("skills_required_text_phobert")
    preferred_skills = row.get("skills_preferred_text_phobert")
    requirements_text = truncate_by_words(row.get("requirements_clean_phobert"), 220)
    description_text = truncate_by_words(row.get("description_clean_phobert"), 220)
    benefits_text = truncate_by_words(row.get("benefits_clean_phobert"), 120)

    if title:
        parts.append(f"Vị trí tuyển dụng: {title}")
    if family and family != "unknown":
        parts.append(f"Nhóm công việc: {family}")
    if job_level and job_level != "unknown":
        parts.append(f"Cấp độ công việc: {job_level}")
    if employment_type and employment_type != "unknown":
        parts.append(f"Loại hình tuyển dụng: {employment_type}")
    if education_level and education_level != "unknown":
        parts.append(f"Yêu cầu học vấn: {education_level}")
    if location:
        parts.append(f"Địa điểm: {location}")
    if working_address:
        parts.append(f"Nơi làm việc: {working_address}")
    if work_mode and work_mode != "unknown":
        parts.append(f"Hình thức làm việc: {work_mode}")
    if salary_brief:
        parts.append(f"Mức lương: {salary_brief}")
    if exp_brief:
        parts.append(f"Kinh nghiệm: {exp_brief}")
    if tags_text:
        parts.append(f"Tags liên quan: {tags_text}")
    if required_skills:
        parts.append(f"Kỹ năng bắt buộc: {required_skills}")
    if preferred_skills:
        parts.append(f"Kỹ năng ưu tiên: {preferred_skills}")
    if requirements_text:
        parts.append(f"Yêu cầu:\n{requirements_text}")
    if description_text:
        parts.append(f"Mô tả công việc:\n{description_text}")
    if benefits_text:
        parts.append(f"Quyền lợi:\n{benefits_text}")

    return "\n\n".join(parts).strip()

# =========================
# SECTION CHUNKING + SECTION RECORDS
# =========================

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

                    # overlap thật sự
                    if end >= len(para):
                        break
                    start = max(0, end - overlap)

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
    job_level = safe_str(row.get("job_level_norm"))
    location = safe_str(row.get("location_city"))
    work_mode = safe_str(row.get("work_mode"))
    required_skills = safe_str(row.get("skills_required_text_phobert"))

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
    if job_level and job_level != "unknown":
        parts.append(f"Cấp độ: {job_level}")
    if location:
        parts.append(f"Địa điểm: {location}")
    if work_mode and work_mode != "unknown":
        parts.append(f"Hình thức làm việc: {work_mode}")
    if required_skills and section_type in {"title", "requirements", "description"}:
        parts.append(f"Kỹ năng chính: {required_skills}")

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
                "job_level_norm": row.get("job_level_norm"),
                "location_city": row.get("location_city"),
                "work_mode": row.get("work_mode"),
                "section_type": section_type,
                "chunk_order": chunk_order,
                "section_priority": SECTION_PRIORITY.get(section_type, 1.0),
                "chunk_text_raw": chunk_text,
                "chunk_text_phobert": build_chunk_text_phobert(row, section_type, chunk_text),
            })

    return rows

# =========================
# PHOBERT PREP + EMBEDDING UTILS
# =========================

def maybe_segment_vi_text(text):
    text = safe_str(text).strip()
    if not text:
        return ""

    # tránh segment các chuỗi quá ngắn / quá ít từ
    if len(text.split()) <= 2:
        return text

    if HAS_UNDERTHESEA:
        try:
            segmented = word_tokenize(text, format="text")
            segmented = safe_str(segmented).strip()
            return segmented if segmented else text
        except Exception:
            return text

    return text


def prepare_phobert_text(text: str) -> str:
    text = safe_str(text)
    if not text.strip():
        return "[EMPTY]"

    text = clean_text_for_phobert(text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return "[EMPTY]"

    text = maybe_segment_vi_text(text)
    text = re.sub(r"\s+", " ", text).strip()

    return text if text else "[EMPTY]"


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    masked_sum = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    mask_sum = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    return masked_sum / mask_sum


def l2_normalize_embeddings(x):
    if isinstance(x, torch.Tensor):
        return torch.nn.functional.normalize(x, p=2, dim=1)
    x = np.asarray(x)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def cosine_similarity_matrix(query_emb, doc_embs):
    query_emb = np.asarray(query_emb)
    doc_embs = np.asarray(doc_embs)

    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)
    if doc_embs.ndim == 1:
        doc_embs = doc_embs.reshape(1, -1)

    query_emb = l2_normalize_embeddings(query_emb)
    doc_embs = l2_normalize_embeddings(doc_embs)

    return np.dot(query_emb, doc_embs.T)

# =========================
# DENSE RETRIEVAL
# =========================

def encode_query_for_matching(query: str, max_length=PHOBERT_MAX_LENGTH_MATCH):
    q = encode_phobert_texts([query], batch_size=1, max_length=max_length, already_prepared=False)
    return q[0]


def retrieve_top_jobs(query: str, top_k: int = 10):
    if job_dense_embeddings is None:
        raise RuntimeError("job_dense_embeddings is None. Hãy bật RUN_EMBEDDING=True.")

    q_emb = encode_query_for_matching(query, max_length=PHOBERT_MAX_LENGTH_MATCH)
    scores = job_dense_embeddings @ q_emb
    top_idx = np.argsort(scores)[::-1][:top_k]

    cols = [
        "job_url",
        "job_title_display",
        "job_family",
        "job_level_norm",
        "location_city",
        "work_mode",
        "skills_required",
        "skills_preferred",
        "job_text_phobert_match",
    ]
    out = df_clean.iloc[top_idx][cols].copy()
    out["cosine_score"] = scores[top_idx]
    return out.reset_index(drop=True)


def retrieve_top_sections(query: str, top_k: int = 10):
    if section_dense_embeddings is None:
        raise RuntimeError("section_dense_embeddings is None. Hãy bật RUN_SECTION_EMBEDDING=True.")

    q_emb = encode_query_for_matching(query, max_length=PHOBERT_MAX_LENGTH_CHUNK)
    scores = section_dense_embeddings @ q_emb
    top_idx = np.argsort(scores)[::-1][:top_k]

    cols = [
        "job_url",
        "job_title_display",
        "job_family",
        "job_level_norm",
        "location_city",
        "section_type",
        "chunk_order",
        "section_priority",
        "chunk_text_raw",
    ]
    out = job_sections_df.iloc[top_idx][cols].copy()
    out["cosine_score"] = scores[top_idx]
    return out.reset_index(drop=True)


# =========================
# PIPELINE STEPS
# =========================

def run_load_raw():
    global df_raw
    df_raw = load_raw_data(RAW_INPUT_PATH)


def run_merge_schema():
    global df
    df = merge_semantic_columns(df_raw)


def run_data_audit():
    global audit_df, missing_df
    audit_rows = []
    combined_text = (
        get_series(df, "job_title_raw", "").fillna("") + " " +
        get_series(df, "description_raw", "").fillna("") + " " +
        get_series(df, "requirements_raw", "").fillna("")
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


def run_base_clean():
    global df_clean
    df_clean = df.copy()
    for col in df_clean.columns:
        df_clean[col] = df_clean[col].apply(normalize_empty_value)

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


def run_title_processing():
    df_clean["job_title_surface"] = df_clean["job_title_raw"].apply(normalize_title_surface)
    df_clean["job_title_match"] = df_clean["job_title_surface"].apply(normalize_for_match)
    df_clean["job_title_canonical"] = df_clean["job_title_surface"].apply(canonicalize_title)
    df_clean["job_title_display"] = df_clean["job_title_surface"].fillna("")

    df_clean["job_family_from_title"] = df_clean["job_title_canonical"].apply(infer_job_family_from_title)
    df_clean["job_family_from_description"] = df_clean["description_clean_strict"].apply(infer_job_family_from_description)

    family_resolved = [
        resolve_job_family(a, b)
        for a, b in zip(df_clean["job_family_from_title"], df_clean["job_family_from_description"])
    ]
    df_clean["job_family"] = [x[0] for x in family_resolved]
    df_clean["job_family_source"] = [x[1] for x in family_resolved]

    df_clean["job_title_for_phobert"] = (
        df_clean["job_title_surface"].fillna("").str.strip()
        + " | "
        + df_clean["job_title_canonical"].fillna("").str.strip()
    ).str.strip(" |").apply(clean_text_for_phobert)


def run_location_processing():
    address_parsed = df_clean["working_addresses_raw"].apply(parse_working_address)
    address_df = pd.DataFrame(address_parsed.tolist(), index=df_clean.index)
    for c in address_df.columns:
        df_clean[c] = address_df[c]


def run_work_mode_processing():
    work_mode_parsed = [
        infer_work_mode_with_source(a, b, c, d)
        for a, b, c, d in zip(
            df_clean["job_title_raw"],
            df_clean["description_raw"],
            df_clean["benefits_raw"],
            df_clean["requirements_raw"]
        )
    ]
    df_clean["work_mode"] = [x[0] for x in work_mode_parsed]
    df_clean["work_mode_source"] = [x[1] for x in work_mode_parsed]


def run_deadline_processing():
    deadline_parsed = df_clean["deadline_raw"].apply(parse_deadline)
    deadline_df = pd.DataFrame(deadline_parsed.tolist(), index=df_clean.index)
    for c in deadline_df.columns:
        df_clean[c] = deadline_df[c]


def run_salary_processing():
    salary_parsed = df_clean["salary_raw"].apply(parse_salary_range)
    salary_df = pd.DataFrame(salary_parsed.tolist(), index=df_clean.index)
    for c in salary_df.columns:
        df_clean[c] = salary_df[c]


def run_experience_education_employment_level():
    exp_parsed = df_clean["experience_raw"].apply(parse_experience_range)
    exp_df = pd.DataFrame(exp_parsed.tolist(), index=df_clean.index)
    for c in exp_df.columns:
        df_clean[c] = exp_df[c]

    df_clean["education_level_norm"] = df_clean["education_level_raw"].apply(normalize_education_level)
    df_clean["employment_type_norm"] = df_clean["employment_type_raw"].apply(normalize_employment_type)

    job_level_parsed = [
        resolve_job_level(a, b, c)
        for a, b, c in zip(
            df_clean["job_title_raw"],
            df_clean["experience_min_years"],
            df_clean["job_level_raw"]
        )
    ]
    df_clean["job_level_norm"] = [x[0] for x in job_level_parsed]
    df_clean["job_level_source"] = [x[1] for x in job_level_parsed]


def run_tags_processing():
    tags_parsed = df_clean["tags_raw"].apply(normalize_tags_structured)
    tags_df = pd.DataFrame(tags_parsed.tolist(), index=df_clean.index)
    for c in tags_df.columns:
        df_clean[c] = tags_df[c]


def run_skill_taxonomy():
    global job_skill_map_df, role_skill_stats_df

    df_clean["skill_records"] = df_clean.apply(extract_job_skill_records, axis=1)
    df_clean["skills_extracted"] = df_clean["skill_records"].apply(lambda rs: list_from_records(rs, "skill"))
    df_clean["skill_groups_extracted"] = df_clean["skill_records"].apply(lambda rs: list_from_records(rs, "skill_group"))
    df_clean["skills_required"] = df_clean["skill_records"].apply(lambda rs: list_from_records(rs, "skill", importance_filter="required"))
    df_clean["skills_preferred"] = df_clean["skill_records"].apply(lambda rs: list_from_records(rs, "skill", importance_filter="preferred"))
    df_clean["skills_mentioned"] = df_clean["skill_records"].apply(lambda rs: list_from_records(rs, "skill", importance_filter="mentioned"))

    df_clean["skills_text_phobert"] = df_clean["skills_extracted"].apply(build_skill_text)
    df_clean["skills_required_text_phobert"] = df_clean["skills_required"].apply(build_skill_text)
    df_clean["skills_preferred_text_phobert"] = df_clean["skills_preferred"].apply(build_skill_text)

    job_skill_rows = []
    for _, row in df_clean.iterrows():
        for r in row["skill_records"]:
            job_skill_rows.append({
                "job_url": row.get("job_url"),
                "job_title_display": row.get("job_title_display"),
                "job_title_canonical": row.get("job_title_canonical"),
                "job_family": row.get("job_family"),
                "location_city": row.get("location_city"),
                "skill": r["skill"],
                "skill_group": r["skill_group"],
                "source_field": r["source_field"],
                "importance": r["importance"],
                "excerpt": r["excerpt"],
            })
    job_skill_map_df = pd.DataFrame(job_skill_rows)

    if len(job_skill_map_df) > 0:
        role_skill_stats_df = (
            job_skill_map_df.groupby(["job_family", "skill", "importance"])
            .size()
            .reset_index(name="job_count")
            .sort_values(["job_family", "job_count"], ascending=[True, False])
        )
    else:
        role_skill_stats_df = pd.DataFrame(columns=["job_family", "skill", "importance", "job_count"])


def run_text_builders():
    df_clean["job_text_sparse"] = df_clean.apply(build_job_text_sparse, axis=1)
    df_clean["job_text_phobert_match"] = df_clean.apply(build_job_text_phobert_match, axis=1)
    df_clean["job_text_phobert_chatbot"] = df_clean.apply(build_job_text_phobert_chatbot, axis=1)
    df_clean["dense_encoder_route"] = "phobert"
    df_clean["dense_similarity_metric"] = "cosine"


def run_section_records():
    global job_sections_df
    section_rows = []
    for _, row in df_clean.iterrows():
        section_rows.extend(build_job_section_records(row))
    job_sections_df = pd.DataFrame(section_rows)


def run_prepare_texts():
    df_clean["job_text_phobert_match_prepared"] = df_clean["job_text_phobert_match"].apply(prepare_phobert_text)
    df_clean["job_text_phobert_chatbot_prepared"] = df_clean["job_text_phobert_chatbot"].apply(prepare_phobert_text)
    if len(job_sections_df) > 0:
        job_sections_df["chunk_text_phobert_prepared"] = job_sections_df["chunk_text_phobert"].apply(prepare_phobert_text)
    else:
        job_sections_df["chunk_text_phobert_prepared"] = ""


def run_load_encoder():
    global tokenizer, model, encode_phobert_texts

    if RUN_EMBEDDING:
        if USE_LOCAL_ONLY and not LOCAL_MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Local model directory not found: {LOCAL_MODEL_DIR}. "
                "Set PREPROCESS_LOCAL_MODEL_DIR to a valid local model path."
            )
        model_source = str(LOCAL_MODEL_DIR) if USE_LOCAL_ONLY else PHOBERT_MODEL_NAME
        pretrained_kwargs = {
            "cache_dir": str(HF_HUB_CACHE_DIR),
            "local_files_only": USE_LOCAL_ONLY,
        }
        tokenizer = AutoTokenizer.from_pretrained(model_source, **pretrained_kwargs)
        model = AutoModel.from_pretrained(model_source, **pretrained_kwargs)
        model.to(DEVICE)
        model.eval()

        def encode_phobert_texts(
            texts,
            batch_size=PHOBERT_BATCH_SIZE,
            max_length=PHOBERT_MAX_LENGTH_MATCH,
            already_prepared=False
        ):
            texts = list(texts) if texts is not None else []
            if len(texts) == 0:
                return np.empty((0, model.config.hidden_size), dtype=np.float32)

            if already_prepared:
                prepared = [safe_str(t).strip() if safe_str(t).strip() else "[EMPTY]" for t in texts]
            else:
                prepared = [prepare_phobert_text(t) for t in texts]

            embeddings = []
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
                    output = model(input_ids=input_ids, attention_mask=attention_mask)

                batch_emb = mean_pooling(output, attention_mask)
                if NORMALIZE_EMBEDDINGS:
                    batch_emb = F.normalize(batch_emb, p=2, dim=1)
                embeddings.append(batch_emb.cpu().numpy())

            return np.vstack(embeddings)
    else:
        tokenizer = None
        model = None

        def encode_phobert_texts(*args, **kwargs):
            raise RuntimeError("RUN_EMBEDDING=False -> encoder is disabled")


def run_embeddings():
    global section_dense_embeddings, job_dense_embeddings

    section_dense_embeddings = None
    job_dense_embeddings = None

    if RUN_EMBEDDING and RUN_SECTION_EMBEDDING and len(job_sections_df) > 0:
        job_sections_df["section_embedding_row_id"] = np.arange(len(job_sections_df))
        section_dense_embeddings = encode_phobert_texts(
            job_sections_df["chunk_text_phobert_prepared"].fillna("").tolist(),
            batch_size=PHOBERT_BATCH_SIZE,
            max_length=PHOBERT_MAX_LENGTH_CHUNK,
            already_prepared=True,
        )
        job_sections_df["has_dense_embedding"] = 1
    else:
        job_sections_df["has_dense_embedding"] = 0

    df_clean["job_embedding_row_id"] = np.arange(len(df_clean))
    if RUN_EMBEDDING and len(df_clean) > 0:
        job_dense_embeddings = encode_phobert_texts(
            df_clean["job_text_phobert_match_prepared"].fillna("").tolist(),
            batch_size=PHOBERT_BATCH_SIZE,
            max_length=PHOBERT_MAX_LENGTH_MATCH,
            already_prepared=True,
        )
        df_clean["has_dense_embedding"] = 1
    else:
        df_clean["has_dense_embedding"] = 0


def run_downstream_tables():
    global DOWNSTREAM_FIELD_GUIDE, df_matching_ready, df_chatbot_ready, job_sections_ready

    DOWNSTREAM_FIELD_GUIDE = {
        "tfidf_input": "job_text_sparse",
        "phobert_matching_input": "job_text_phobert_match",
        "phobert_chatbot_input": "job_text_phobert_chatbot",
        "chatbot_chunk_table": "job_sections_df",
        "chatbot_chunk_text_field": "chunk_text_phobert",
        "skill_table": "job_skill_map_df",
        "role_skill_stats": "role_skill_stats_df",
        "retrieval_metric": "cosine_similarity_on_l2_normalized_embeddings",
        "job_embedding_id_field": "job_embedding_row_id",
        "section_embedding_id_field": "section_embedding_row_id",
        "location_field": "location_city",
        "working_address_field": "working_address_clean",
    }

    matching_cols = [
        "job_url", "job_id",
        "job_title_display", "job_title_canonical", "job_family",
        "job_level_norm",
        "location_city", "working_address_clean", "is_multi_location", "work_mode",
        "salary_min_vnd_month", "salary_max_vnd_month", "salary_is_negotiable", "salary_bucket",
        "experience_min_years", "experience_max_years", "experience_type",
        "education_level_norm", "employment_type_norm",
        "skills_extracted", "skills_required", "skills_preferred", "skill_groups_extracted",
        "tags_role_list", "tags_domain_list", "tags_specialty_list",
        "job_text_sparse", "job_text_phobert_match", "job_text_phobert_chatbot",
        "job_embedding_row_id", "has_dense_embedding", "dense_encoder_route", "dense_similarity_metric",
    ]
    matching_cols = [c for c in matching_cols if c in df_clean.columns]
    df_matching_ready = df_clean[matching_cols].copy()
    df_matching_ready["dense_model_name"] = PHOBERT_MODEL_NAME
    df_matching_ready["dense_similarity_metric"] = "cosine"

    chatbot_cols = [
        "job_url", "job_id",
        "job_title_display", "job_title_canonical", "job_family", "job_level_norm",
        "location_city", "working_address_clean", "is_multi_location", "work_mode",
        "salary_min_vnd_month", "salary_max_vnd_month", "salary_is_negotiable", "salary_bucket",
        "experience_min_years", "experience_max_years", "experience_type",
        "education_level_norm", "employment_type_norm",
        "skills_extracted", "skills_required", "skills_preferred",
        "tags_role_list", "tags_domain_list", "tags_specialty_list",
        "requirements_clean_phobert", "description_clean_phobert", "benefits_clean_phobert",
        "job_text_phobert_chatbot",
        "job_embedding_row_id", "has_dense_embedding"
    ]
    chatbot_cols = [c for c in chatbot_cols if c in df_clean.columns]
    df_chatbot_ready = df_clean[chatbot_cols].copy()

    section_cols = [
        "section_embedding_row_id",
        "job_url", "job_title_display", "job_title_canonical",
        "job_family", "job_level_norm", "location_city", "work_mode",
        "section_type", "chunk_order", "section_priority",
        "chunk_text_raw", "chunk_text_phobert",
        "has_dense_embedding"
    ]
    section_cols = [c for c in section_cols if c in job_sections_df.columns]
    job_sections_ready = job_sections_df[section_cols].copy()


def run_save_artifacts():
    global artifact_paths
    artifact_paths = {}

    if SAVE_INTERMEDIATE:
        artifact_paths["jobs_matching_ready"] = save_table(df_matching_ready, ARTIFACT_DIR / artifact_name("jobs_matching_ready"))
        artifact_paths["jobs_chatbot_ready"] = save_table(df_chatbot_ready, ARTIFACT_DIR / artifact_name("jobs_chatbot_ready"))
        artifact_paths["jobs_chatbot_sections"] = save_table(job_sections_ready, ARTIFACT_DIR / artifact_name("jobs_chatbot_sections"))
        artifact_paths["job_skill_map"] = save_table(job_skill_map_df, ARTIFACT_DIR / artifact_name("job_skill_map"))
        artifact_paths["role_skill_stats"] = save_table(role_skill_stats_df, ARTIFACT_DIR / artifact_name("role_skill_stats"))

        artifact_paths["job_embedding_index"] = save_table(
            df_clean[[
                "job_embedding_row_id", "job_url", "job_title_display", "job_title_canonical",
                "job_family", "job_level_norm", "location_city", "work_mode", "job_text_phobert_match"
            ]],
            ARTIFACT_DIR / artifact_name("job_embedding_index")
        )

        if len(job_sections_df) > 0:
            artifact_paths["job_section_embedding_index"] = save_table(
                job_sections_df[[
                    "section_embedding_row_id", "job_url", "job_title_display", "job_title_canonical",
                    "job_family", "job_level_norm", "location_city", "section_type",
                    "chunk_order", "chunk_text_raw", "chunk_text_phobert"
                ]],
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


def run_manifest():
    global manifest
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
            "job_text_prepared_field": "job_text_phobert_match_prepared" if "job_text_phobert_match_prepared" in df_clean.columns else None,
            "chatbot_text_field": "job_text_phobert_chatbot",
            "chatbot_text_prepared_field": "job_text_phobert_chatbot_prepared" if "job_text_phobert_chatbot_prepared" in df_clean.columns else None,
            "section_text_field": "chunk_text_phobert",
            "section_text_prepared_field": "chunk_text_phobert_prepared" if "chunk_text_phobert_prepared" in job_sections_df.columns else None,
            "job_embedding_id_field": "job_embedding_row_id",
            "section_embedding_id_field": "section_embedding_row_id",
            "max_length_match": PHOBERT_MAX_LENGTH_MATCH,
            "max_length_chatbot": PHOBERT_MAX_LENGTH_CHATBOT,
            "max_length_chunk": PHOBERT_MAX_LENGTH_CHUNK,
            "batch_size": PHOBERT_BATCH_SIZE,
            "segmentation": "underthesea_if_available",
        },
        "core_fields": {
            "title_display": "job_title_display",
            "title_canonical": "job_title_canonical",
            "job_family": "job_family",
            "job_level": "job_level_norm",
            "location": "location_city",
            "working_address": "working_address_clean",
            "work_mode": "work_mode",
            "salary_min": "salary_min_vnd_month",
            "salary_max": "salary_max_vnd_month",
            "experience_min": "experience_min_years",
            "experience_max": "experience_max_years",
            "education_level": "education_level_norm",
            "employment_type": "employment_type_norm",
            "skills": "skills_extracted",
            "skills_required": "skills_required",
            "skills_preferred": "skills_preferred",
            "tags_model_text": "tags_text_for_model",
        },
        "downstream_field_guide": DOWNSTREAM_FIELD_GUIDE,
        "artifacts": artifact_paths,
    }

    if SAVE_INTERMEDIATE:
        manifest_path = ARTIFACT_DIR / f"{artifact_name('manifest_phobert')}.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)


def run_length_stats():
    global length_stats, section_stats
    length_stats = pd.DataFrame({
        "job_text_sparse_len": df_clean["job_text_sparse"].fillna("").str.len(),
        "job_text_phobert_match_len": df_clean["job_text_phobert_match"].fillna("").str.len(),
        "job_text_phobert_chatbot_len": df_clean["job_text_phobert_chatbot"].fillna("").str.len(),
        "tags_text_for_model_len": df_clean["tags_text_for_model"].fillna("").str.len() if "tags_text_for_model" in df_clean.columns else 0,
        "skills_text_phobert_len": df_clean["skills_text_phobert"].fillna("").str.len() if "skills_text_phobert" in df_clean.columns else 0,
        "skills_required_text_phobert_len": df_clean["skills_required_text_phobert"].fillna("").str.len() if "skills_required_text_phobert" in df_clean.columns else 0,
    })

    if len(job_sections_df) > 0:
        section_stats = (
            job_sections_df.assign(chunk_len=job_sections_df["chunk_text_phobert"].fillna("").str.len())
            .groupby("section_type")["chunk_len"]
            .describe()
            .reset_index()
        )
    else:
        section_stats = pd.DataFrame()


def run_final_qa():
    global qa_summary
    def pct_missing(series):
        return float(series.isna().mean()) if len(series) > 0 else 0.0

    def pct_unknown(series):
        s = series.fillna("").astype(str).str.strip().str.lower()
        return float((s == "unknown").mean()) if len(series) > 0 else 0.0

    qa_summary = {
        "n_jobs": int(len(df_clean)),
        "n_sections": int(len(job_sections_df)),
        "pct_missing_job_title_display": pct_missing(df_clean["job_title_display"]) if "job_title_display" in df_clean.columns else None,
        "pct_missing_job_title_canonical": pct_missing(df_clean["job_title_canonical"]) if "job_title_canonical" in df_clean.columns else None,
        "pct_unknown_job_family": pct_unknown(df_clean["job_family"]) if "job_family" in df_clean.columns else None,
        "pct_missing_location_city": pct_missing(df_clean["location_city"]) if "location_city" in df_clean.columns else None,
        "pct_true_is_multi_location": float(df_clean["is_multi_location"].fillna(False).mean()) if "is_multi_location" in df_clean.columns else None,
        "pct_unknown_work_mode": pct_unknown(df_clean["work_mode"]) if "work_mode" in df_clean.columns else None,
        "pct_missing_salary_min": pct_missing(df_clean["salary_min_vnd_month"]) if "salary_min_vnd_month" in df_clean.columns else None,
        "pct_missing_salary_max": pct_missing(df_clean["salary_max_vnd_month"]) if "salary_max_vnd_month" in df_clean.columns else None,
        "pct_negotiable_salary": float(df_clean["salary_is_negotiable"].fillna(False).mean()) if "salary_is_negotiable" in df_clean.columns else None,
        "pct_missing_experience_min": pct_missing(df_clean["experience_min_years"]) if "experience_min_years" in df_clean.columns else None,
        "pct_missing_experience_max": pct_missing(df_clean["experience_max_years"]) if "experience_max_years" in df_clean.columns else None,
        "pct_unknown_experience_type": pct_unknown(df_clean["experience_type"]) if "experience_type" in df_clean.columns else None,
        "pct_unknown_education_level": pct_unknown(df_clean["education_level_norm"]) if "education_level_norm" in df_clean.columns else None,
        "pct_unknown_employment_type": pct_unknown(df_clean["employment_type_norm"]) if "employment_type_norm" in df_clean.columns else None,
        "pct_unknown_job_level": pct_unknown(df_clean["job_level_norm"]) if "job_level_norm" in df_clean.columns else None,
        "avg_tags_per_job": float(df_clean["tags_list"].apply(len).mean()) if "tags_list" in df_clean.columns else None,
        "avg_skills_per_job": float(df_clean["skills_extracted"].apply(len).mean()) if "skills_extracted" in df_clean.columns else None,
        "avg_required_skills_per_job": float(df_clean["skills_required"].apply(len).mean()) if "skills_required" in df_clean.columns else None,
        "avg_preferred_skills_per_job": float(df_clean["skills_preferred"].apply(len).mean()) if "skills_preferred" in df_clean.columns else None,
        "pct_jobs_without_skills": float((df_clean["skills_extracted"].apply(len) == 0).mean()) if "skills_extracted" in df_clean.columns else None,
        "avg_job_text_sparse_len": float(df_clean["job_text_sparse"].fillna("").str.len().mean()) if "job_text_sparse" in df_clean.columns else None,
        "avg_job_text_match_len": float(df_clean["job_text_phobert_match"].fillna("").str.len().mean()) if "job_text_phobert_match" in df_clean.columns else None,
        "avg_job_text_chatbot_len": float(df_clean["job_text_phobert_chatbot"].fillna("").str.len().mean()) if "job_text_phobert_chatbot" in df_clean.columns else None,
        "avg_section_chunk_len": float(job_sections_df["chunk_text_phobert"].fillna("").str.len().mean()) if "chunk_text_phobert" in job_sections_df.columns and len(job_sections_df) > 0 else None,
        "job_has_dense_embedding_rate": float(df_clean["has_dense_embedding"].fillna(0).mean()) if "has_dense_embedding" in df_clean.columns else None,
        "section_has_dense_embedding_rate": float(job_sections_df["has_dense_embedding"].fillna(0).mean()) if "has_dense_embedding" in job_sections_df.columns and len(job_sections_df) > 0 else None,
    }


def main():
    steps = [
        ("Load raw data", run_load_raw),
        ("Merge schema", run_merge_schema),
        ("Data audit", run_data_audit),
        ("Base cleaning", run_base_clean),
        ("Title processing", run_title_processing),
        ("Location processing", run_location_processing),
        ("Work mode processing", run_work_mode_processing),
        ("Deadline processing", run_deadline_processing),
        ("Salary processing", run_salary_processing),
        ("Experience/education/employment/job-level processing", run_experience_education_employment_level),
        ("Tags processing", run_tags_processing),
        ("Skill taxonomy processing", run_skill_taxonomy),
        ("Build job text fields", run_text_builders),
        ("Build section records", run_section_records),
        ("Prepare PhoBERT texts", run_prepare_texts),
        ("Load PhoBERT encoder", run_load_encoder),
        ("Create embeddings", run_embeddings),
        ("Build downstream tables", run_downstream_tables),
        ("Save intermediate artifacts", run_save_artifacts),
        ("Build manifest", run_manifest),
        ("Build length statistics", run_length_stats),
        ("Build final QA summary", run_final_qa),
    ]
    for name, fn in steps:
        run_step(name, fn)

    log_info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()

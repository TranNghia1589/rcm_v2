import glob
import os
import re
import json
import unicodedata
from datetime import datetime

import pandas as pd


# =========================
# CẤU HÌNH ĐƯỜNG DẪN
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_TOPCV_DIR = os.path.join(BASE_DIR, "data", "raw", "jobs")

# NOTE: Raw file name includes a timestamp. Automatically pick the latest CSV/XLSX.
def find_latest_raw_file(data_dir: str) -> str:
    patterns = [os.path.join(data_dir, "topcv_all_fields_merged_*.csv"),
                os.path.join(data_dir, "topcv_all_fields_merged_*.xlsx")]
    candidates = []
    for p in patterns:
        candidates.extend(glob.glob(p))
    if not candidates:
        raise FileNotFoundError(f"Không tìm thấy file raw trong {data_dir}")
    # sort by modified time descending
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]

RAW_INPUT_PATH = find_latest_raw_file(DATA_TOPCV_DIR)

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")

# =========================
# CẤU HÌNH SKILL DICTIONARY
# Có thể tách ra file json sau này
# =========================
SKILL_DICT = {
    "python": ["python"],
    "sql": ["sql", "mysql", "postgresql", "sql server", "mssql"],
    "excel": ["excel", "microsoft excel"],
    "power_bi": ["power bi", "powerbi"],
    "tableau": ["tableau"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "scikit_learn": ["scikit-learn", "sklearn"],
    "machine_learning": ["machine learning", "ml"],
    "deep_learning": ["deep learning", "dl"],
    "spark": ["spark", "apache spark"],
    "hadoop": ["hadoop"],
    "airflow": ["airflow"],
    "docker": ["docker"],
    "git": ["git", "github", "gitlab"],
    "statistics": ["thống kê", "statistics", "statistical"],
    "data_visualization": ["trực quan hóa dữ liệu", "data visualization", "visualization"],
}


# =========================
# HÀM ĐỌC FILE RAW
# Nhiệm vụ:
# - đọc file Excel/CSV đầu vào
# - trả về DataFrame raw
# =========================
def load_raw_data(input_path: str) -> pd.DataFrame:
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".xlsx":
        df = pd.read_excel(input_path)
    elif ext == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Định dạng file không hỗ trợ: {ext}")

    print(f"[INFO] Loaded raw data: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


# =========================
# HÀM CHUẨN HÓA GIÁ TRỊ RỖNG
# Nhiệm vụ:
# - biến các chuỗi rỗng / 'nan' / 'None' thành None
# =========================
def normalize_empty_value(val):
    if pd.isna(val):
        return None

    val_str = str(val).strip()
    if not val_str:
        return None

    if val_str.lower() in {"nan", "none", "null"}:
        return None

    return val_str


# =========================
# HÀM CHỌN GIÁ TRỊ ĐẦU TIÊN KHÔNG RỖNG
# Nhiệm vụ:
# - dùng để gộp các cột cùng ý nghĩa
# - ưu tiên cột detail trước, rồi fallback sang cột search/company
# =========================
def first_non_empty(*values):
    for v in values:
        v = normalize_empty_value(v)
        if v is not None:
            return v
    return None


# =========================
# HÀM GỘP CÁC CỘT CÙNG Ý NGHĨA
# Nhiệm vụ:
# - rút gọn schema raw về schema chuẩn để dễ clean
# - chỉ giữ 1 cột đại diện cho mỗi loại thông tin
# =========================
def merge_semantic_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()

    out["job_url"] = df.get("job_url")
    out["source_field_name"] = df.get("source_field_name")
    out["field_count"] = df.get("field_count")

    # title
    out["job_title_raw"] = [
        first_non_empty(a, b)
        for a, b in zip(df.get("detail_title", [None] * len(df)),
                        df.get("title", [None] * len(df)))
    ]

    # company name
    out["company_name_raw"] = [
        first_non_empty(a, b)
        for a, b in zip(df.get("company_name_full", [None] * len(df)),
                        df.get("company_name", [None] * len(df)))
    ]

    # company url
    out["company_url"] = [
        first_non_empty(a, b)
        for a, b in zip(df.get("company_url_from_job", [None] * len(df)),
                        df.get("company_url", [None] * len(df)))
    ]

    # salary
    out["salary_raw"] = [
        first_non_empty(a, b)
        for a, b in zip(df.get("detail_salary", [None] * len(df)),
                        df.get("salary_list", [None] * len(df)))
    ]

    # location
    out["location_raw"] = [
        first_non_empty(a, b, c)
        for a, b, c in zip(
            df.get("detail_location", [None] * len(df)),
            df.get("address_list", [None] * len(df)),
            df.get("working_addresses", [None] * len(df)),
        )
    ]

    # working addresses / working times
    out["working_addresses_raw"] = df.get("working_addresses")
    out["working_times_raw"] = df.get("working_times")

    # experience
    out["experience_raw"] = [
        first_non_empty(a, b)
        for a, b in zip(df.get("detail_experience", [None] * len(df)),
                        df.get("exp_list", [None] * len(df)))
    ]

    # description / requirements / benefits
    out["description_raw"] = df.get("desc_mota")
    out["requirements_raw"] = df.get("desc_yeucau")
    out["benefits_raw"] = df.get("desc_quyenloi")

    # các trường khác
    out["job_level_raw"] = df.get("job_level")
    out["education_level_raw"] = df.get("education_level")
    out["employment_type_raw"] = df.get("employment_type")
    out["deadline_raw"] = df.get("deadline")
    out["company_scale_raw"] = [
        first_non_empty(a, b)
        for a, b in zip(df.get("company_scale_from_job", [None] * len(df)),
                        df.get("company_scale", [None] * len(df)))
    ]
    out["company_address_raw"] = [
        first_non_empty(a, b)
        for a, b in zip(df.get("company_address_from_job", [None] * len(df)),
                        df.get("company_address", [None] * len(df)))
    ]
    out["company_description_raw"] = df.get("company_description")

    print(f"[INFO] After merging semantic columns: {out.shape[0]} rows x {out.shape[1]} cols")
    return out


# =========================
# HÀM CHUẨN HÓA UNICODE
# Nhiệm vụ:
# - đưa text về dạng unicode chuẩn
# =========================
def normalize_unicode(text: str) -> str:
    if text is None:
        return ""
    return unicodedata.normalize("NFC", str(text))


# =========================
# HÀM CLEAN TEXT
# Nhiệm vụ:
# - chuẩn hóa text cho NLP
# - lowercase
# - bỏ html, ký tự thừa, khoảng trắng thừa
# - giữ tiếng Việt có dấu
# =========================
def clean_text(text: str) -> str:
    text = normalize_empty_value(text)
    if text is None:
        return ""

    text = normalize_unicode(text)
    text = text.lower()

    # bỏ html tag
    text = re.sub(r"<[^>]+>", " ", text)

    # bỏ xuống dòng / tab
    text = re.sub(r"[\r\n\t]+", " ", text)

    # bỏ bullet unicode phổ biến
    text = text.replace("•", " ").replace("▪", " ").replace("✅", " ")

    # chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================
# HÀM CLEAN WORKING ADDRESSES
# Nhiệm vụ:
# - loại bỏ nội dung trong ngoặc (ví dụ: "(quận Tân Bình cũ)")
# - giữ nguyên phần địa chỉ chính
# =========================
def clean_working_addresses(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ""

    # xóa các phần trong ngoặc đơn/ngoặc vuông/ngoặc nhọn
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\{[^}]*\}", " ", text)

    # chuẩn hóa khoảng trắng sau khi xóa
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================
# HÀM CHUẨN HÓA LOCATION
# Nhiệm vụ:
# - gom các cách viết location khác nhau về 1 chuẩn
# =========================
def normalize_location(text: str) -> str:
    text = clean_text(text)

    if not text:
        return ""

    mapping = {
        "tp.hcm": "ho chi minh",
        "tphcm": "ho chi minh",
        "tp hcm": "ho chi minh",
        "hcm": "ho chi minh",
        "hồ chí minh": "ho chi minh",
        "sài gòn": "ho chi minh",
        "hn": "ha noi",
        "hà nội": "ha noi",
        "đà nẵng": "da nang",
    }

    for k, v in mapping.items():
        if k in text:
            return v

    return text


# =========================
# HÀM PARSE SALARY
# Nhiệm vụ:
# - tách salary_raw thành salary_min / salary_max / salary_type
# - hiện tại là bản heuristic cơ bản
# =========================
def parse_salary(text: str):
    text = clean_text(text)

    if not text:
        return None, None, "unknown"

    if "thỏa thuận" in text or "thuong luong" in text or "cạnh tranh" in text:
        return None, None, "negotiable"

    nums = re.findall(r"\d+(?:[.,]\d+)?", text)
    nums = [float(n.replace(",", ".")) for n in nums]

    if "triệu" in text or "trieu" in text:
        nums = [int(n * 1_000_000) for n in nums]
    elif "usd" in text:
        nums = [int(n) for n in nums]
    else:
        nums = [int(n) for n in nums]

    if len(nums) >= 2:
        return nums[0], nums[1], "range"
    if len(nums) == 1:
        return nums[0], nums[0], "fixed"

    return None, None, "unknown"


# =========================
# HÀM PARSE EXPERIENCE
# Nhiệm vụ:
# - chuẩn hóa experience_raw thành mức kinh nghiệm dễ dùng hơn
# =========================
def parse_experience(text: str):
    text = clean_text(text)

    if not text:
        return None, None, "unknown"

    if "không yêu cầu" in text:
        return 0, 0, "no_requirement"

    if "thực tập" in text or "intern" in text or "fresher" in text:
        return 0, 1, "intern_fresher"

    nums = re.findall(r"\d+", text)
    nums = [int(x) for x in nums]

    if len(nums) >= 2:
        return nums[0], nums[1], "range"
    if len(nums) == 1:
        if "+" in text or "trên" in text:
            return nums[0], None, "min_only"
        return nums[0], nums[0], "fixed"

    return None, None, "unknown"


# =========================
# HÀM EXTRACT SKILLS
# Nhiệm vụ:
# - tìm skill trong text dựa trên dictionary
# - trả về danh sách skill chuẩn hóa
# =========================
def extract_skills(text: str, skill_dict: dict) -> list:
    text = clean_text(text)
    found = []

    for canonical_skill, aliases in skill_dict.items():
        for alias in aliases:
            alias_clean = clean_text(alias)
            pattern = rf"(?<!\w){re.escape(alias_clean)}(?!\w)"
            if re.search(pattern, text):
                found.append(canonical_skill)
                break

    return sorted(set(found))


# =========================
# HÀM TẠO JOB_TEXT
# Nhiệm vụ:
# - tạo 1 text tổng hợp để dùng cho TF-IDF / PhoBERT
# - gom title + requirements + description + skill
# =========================
def build_job_text(row: pd.Series) -> str:
    parts = []

    if row.get("job_title_clean"):
        parts.append(f"[title] {row['job_title_clean']}")

    if row.get("requirements_clean"):
        parts.append(f"[requirements] {row['requirements_clean']}")

    if row.get("description_clean"):
        parts.append(f"[description] {row['description_clean']}")

    if row.get("skills_normalized"):
        parts.append(f"[skills] {' | '.join(row['skills_normalized'])}")

    return " ".join(parts).strip()


# =========================
# HÀM CHỌN CÁC CỘT CUỐI CÙNG
# Nhiệm vụ:
# - chỉ giữ lại các cột phục vụ DB + NLP
# =========================
def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    final_cols = [
        "job_url",
        "source_field_name",
        "field_count",
        "job_title_raw",
        "job_title_clean",
        "company_name_raw",
        "company_url",
        "salary_raw",
        "salary_min",
        "salary_max",
        "salary_type",
        "location_raw",
        "location_normalized",
        "working_addresses_raw",
        "working_addresses_clean",
        "working_times_raw",
        "working_times_clean",
        "experience_raw",
        "experience_min_years",
        "experience_max_years",
        "experience_type",
        "job_level_raw",
        "education_level_raw",
        "employment_type_raw",
        "deadline_raw",
        "company_scale_raw",
        "company_address_raw",
        "company_description_raw",
        "description_raw",
        "description_clean",
        "requirements_raw",
        "requirements_clean",
        "benefits_raw",
        "benefits_clean",
        "skills_normalized",
        "skills_normalized_str",
        "job_text",
    ]

    existing_cols = [c for c in final_cols if c in df.columns]
    return df[existing_cols].copy()


# =========================
# HÀM LƯU FILE OUTPUT
# Nhiệm vụ:
# - lưu bản processed ra CSV + Excel
# =========================
def save_processed_data(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    csv_path = os.path.join(output_dir, f"jobs_nlp_ready_{timestamp}.csv")
    xlsx_path = os.path.join(output_dir, f"jobs_nlp_ready_{timestamp}.xlsx")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved CSV  -> {csv_path}")

    try:
        df.to_excel(xlsx_path, index=False)
        print(f"[INFO] Saved XLSX -> {xlsx_path}")
    except Exception as e:
        print(f"[WARN] Không lưu được Excel: {e}")


# =========================
# HÀM CHÍNH PREPROCESS
# Nhiệm vụ:
# - điều phối toàn bộ pipeline tiền xử lý
# =========================
def main():
    print("[STEP 1] Load raw data")
    raw_df = load_raw_data(RAW_INPUT_PATH)

    print("[STEP 2] Merge semantic columns")
    df = merge_semantic_columns(raw_df)

    print("[STEP 3] Normalize basic text columns")
    df["job_title_clean"] = df["job_title_raw"].apply(clean_text)
    df["description_clean"] = df["description_raw"].apply(clean_text)
    df["requirements_clean"] = df["requirements_raw"].apply(clean_text)
    df["benefits_clean"] = df["benefits_raw"].apply(clean_text)

    # Làm sạch địa chỉ làm việc: loại bỏ các chú thích trong ngoặc
    df["working_addresses_clean"] = df["working_addresses_raw"].apply(clean_working_addresses)
    df["working_times_clean"] = df["working_times_raw"].apply(clean_text)

    print("[STEP 4] Normalize structured columns")
    df["location_normalized"] = df["location_raw"].apply(normalize_location)

    salary_parsed = df["salary_raw"].apply(parse_salary)
    df["salary_min"] = salary_parsed.apply(lambda x: x[0])
    df["salary_max"] = salary_parsed.apply(lambda x: x[1])
    df["salary_type"] = salary_parsed.apply(lambda x: x[2])

    exp_parsed = df["experience_raw"].apply(parse_experience)
    df["experience_min_years"] = exp_parsed.apply(lambda x: x[0])
    df["experience_max_years"] = exp_parsed.apply(lambda x: x[1])
    df["experience_type"] = exp_parsed.apply(lambda x: x[2])

    print("[STEP 5] Extract skills")
    skill_source_text = (
        df["job_title_clean"].fillna("") + " " +
        df["requirements_clean"].fillna("") + " " +
        df["description_clean"].fillna("")
    )
    df["skills_normalized"] = skill_source_text.apply(lambda x: extract_skills(x, SKILL_DICT))
    df["skills_normalized_str"] = df["skills_normalized"].apply(
        lambda x: " | ".join(x) if isinstance(x, list) else ""
    )

    print("[STEP 6] Build job_text for NLP")
    df["job_text"] = df.apply(build_job_text, axis=1)

    print("[STEP 7] Select final columns")
    final_df = select_final_columns(df)

    print(f"[INFO] Final processed shape: {final_df.shape[0]} rows x {final_df.shape[1]} cols")

    print("[STEP 8] Save processed data")
    save_processed_data(final_df, OUTPUT_DIR)

    print("[DONE] Preprocessing completed.")


if __name__ == "__main__":
    main()

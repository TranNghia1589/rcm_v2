# src/data_processing/merge_jobs.py

from __future__ import annotations
import glob
import os
import re
from typing import List
import pandas as pd


RAW_DIR = "data/raw"
OUTPUT_PATH = "data/processed/jobs_merged_cleaned.csv"
ROLE_MAP = {
    "data_analyst": "Data Analyst",
    "data_engineer": "Data Engineer",
    "ai_engineer": "AI Engineer",
    "ai_researcher": "AI Researcher",
    "data_labeling": "Data Labeling",
    "gan_nhan_du_lieu": "Data Labeling",
}


TEXT_COLUMNS = [
    "title",
    "detail_title",
    "company",
    "company_name_full",
    "salary_list",
    "detail_salary",
    "detail_location",
    "detail_experience",
    "tags",
    "job_level",
    "education_level",
    "employment_type",
    "desc_mota",
    "desc_yeucau",
    "desc_quyenloi",
    "company_field",
    "company_description",
]


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text)

    # bỏ html cơ bản
    text = re.sub(r"<[^>]+>", " ", text)

    # chuẩn hóa khoảng trắng
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()

    return text


def infer_role_from_filename(filename: str) -> str:
    lowered = filename.lower()
    for key, value in ROLE_MAP.items():
        if key in lowered:
            return value
    return "Unknown"


def safe_get(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([""] * len(df))


def build_job_text(row: pd.Series) -> str:
    parts = [
        f"Role: {row.get('source_role', '')}",
        f"Title: {row.get('title', '')}",
        f"Detail title: {row.get('detail_title', '')}",
        f"Company: {row.get('company_name_full', '') or row.get('company', '')}",
        f"Location: {row.get('detail_location', '')}",
        f"Experience: {row.get('detail_experience', '')}",
        f"Salary: {row.get('salary_list', '') or row.get('detail_salary', '')}",
        f"Tags: {row.get('tags', '')}",
        f"Job level: {row.get('job_level', '')}",
        f"Education: {row.get('education_level', '')}",
        f"Employment type: {row.get('employment_type', '')}",
        f"Description: {row.get('desc_mota', '')}",
        f"Requirements: {row.get('desc_yeucau', '')}",
        f"Benefits: {row.get('desc_quyenloi', '')}",
        f"Company field: {row.get('company_field', '')}",
        f"Company description: {row.get('company_description', '')}",
    ]
    return "\n".join([p for p in parts if p.split(": ", 1)[1].strip()])


def normalize_tags(tag_value: str) -> str:
    if not tag_value:
        return ""
    text = clean_text(tag_value)

    # tách bằng dấu phẩy / chấm phẩy / slash / pipe
    chunks = re.split(r"[,;/|]", text)
    chunks = [c.strip() for c in chunks if c.strip()]

    # loại trùng nhưng giữ thứ tự
    seen = set()
    result = []
    for item in chunks:
        lowered = item.lower()
        if lowered not in seen:
            seen.add(lowered)
            result.append(item)

    return ", ".join(result)


def load_job_files() -> List[str]:
    csv_files = glob.glob(os.path.join(RAW_DIR, "topcv_*.csv"))
    # bỏ file resume nếu có
    csv_files = [f for f in csv_files if "resume" not in os.path.basename(f).lower()]
    return sorted(csv_files)


def main() -> None:
    files = load_job_files()
    if not files:
        raise FileNotFoundError("Không tìm thấy file topcv_*.csv trong /mnt/data")

    all_dfs = []

    for file_path in files:
        print(f"Loading: {file_path}")
        df = pd.read_csv(file_path)

        # chỉ giữ các cột cần nếu có
        selected = pd.DataFrame()
        for col in TEXT_COLUMNS:
            selected[col] = safe_get(df, col)

        selected["job_url"] = safe_get(df, "job_url")
        selected["deadline"] = safe_get(df, "deadline")
        selected["source_file"] = os.path.basename(file_path)
        selected["source_role"] = infer_role_from_filename(os.path.basename(file_path))

        # clean text
        for col in selected.columns:
            selected[col] = selected[col].apply(clean_text)

        selected["tags"] = selected["tags"].apply(normalize_tags)

        # loại job không có title lẫn yêu cầu
        selected = selected[
            (selected["title"].str.len() > 0)
            | (selected["desc_yeucau"].str.len() > 0)
            | (selected["desc_mota"].str.len() > 0)
        ].copy()

        all_dfs.append(selected)

    merged = pd.concat(all_dfs, ignore_index=True)

    # bỏ trùng cơ bản
    merged["dedup_key"] = (
        merged["title"].str.lower().fillna("")
        + "||"
        + merged["company_name_full"].str.lower().fillna("")
        + "||"
        + merged["detail_location"].str.lower().fillna("")
    )
    merged = merged.drop_duplicates(subset=["dedup_key"]).drop(columns=["dedup_key"])

    merged["job_text"] = merged.apply(build_job_text, axis=1)

    print("Merged rows:", len(merged))
    print("Roles distribution:")
    print(merged["source_role"].value_counts(dropna=False))

    merged.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
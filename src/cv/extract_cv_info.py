# src/cv_processing/extract_cv_info.py

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List

import fitz  # PyMuPDF
from docx import Document


BASE_DIR = Path(__file__).resolve().parents[2]
SKILL_CATALOG_PATH = BASE_DIR / "data" / "skill_catalog.json"

SKILL_VOCAB = [
    "python", "sql", "excel", "power bi", "powerbi", "tableau",
    "pandas", "numpy", "machine learning", "deep learning",
    "pytorch", "tensorflow", "scikit-learn", "sklearn",
    "spark", "hadoop", "airflow", "etl", "nlp", "computer vision",
    "statistics", "data visualization", "dashboard", "dashboarding",
    "git", "docker", "linux", "mysql", "postgresql", "postgres",
    "mongodb", "aws", "azure", "gcp", "llm", "rag", "langchain",
    "streamlit", "flask", "fastapi", "power query"
]


ROLE_KEYWORDS = {
    "Data Analyst": [
        "data analyst",
        "bi analyst",
        "business intelligence",
        "power bi",
        "tableau",
        "dashboard"
    ],
    "Data Engineer": [
        "data engineer",
        "etl",
        "data pipeline",
        "data warehouse",
        "airflow",
        "spark"
    ],
    "AI Engineer": [
        "ai engineer",
        "machine learning engineer",
        "ml engineer",
        "model deployment",
        "llm"
    ],
    "AI Researcher": [
        "ai researcher",
        "research scientist",
        "research assistant",
        "paper",
        "experiment"
    ],
    "Data Scientist": [
        "data scientist",
        "machine learning",
        "predictive modeling",
        "statistical modeling"
    ]
}
EDUCATION_KEYWORDS = [
    "university", "college", "đại học", "cao đẳng", "bachelor", "master",
    "cử nhân", "thạc sĩ", "khoa học dữ liệu", "data science", "computer science",
    "information technology", "công nghệ thông tin"
]


def read_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    pages = [page.get_text() for page in doc]
    return "\n".join(pages)


def read_docx(file_path: str) -> str:
    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)


def read_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_cv_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return read_pdf(file_path)
    if ext == ".docx":
        return read_docx(file_path)
    if ext == ".txt":
        return read_txt(file_path)
    raise ValueError(f"Unsupported CV format: {ext}")


def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_email(text: str) -> str:
    match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return match.group(0) if match else ""


def extract_phone(text: str) -> str:
    match = re.search(r"(\+?\d[\d\s\-().]{8,}\d)", text)
    return match.group(0).strip() if match else ""

def load_skill_catalog() -> Dict[str, List[str]]:
    with open(SKILL_CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_skills(text: str, skill_catalog: Dict[str, List[str]]) -> List[str]:
    lowered = text.lower()
    found = []

    for canonical_skill, aliases in skill_catalog.items():
        for alias in aliases:
            pattern = r"\b" + re.escape(alias.lower()) + r"\b"
            if re.search(pattern, lowered):
                found.append(canonical_skill)
                break

    unique = []
    seen = set()
    for item in found:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique


def guess_target_role(text: str, skills: List[str]) -> str:
    lowered = text.lower()
    role_scores = {}

    for role, keywords in ROLE_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw in lowered:
                score += 2
        role_scores[role] = score

    best_role = max(role_scores, key=role_scores.get)
    best_score = role_scores[best_role]

    # Nếu không có tín hiệu đủ mạnh thì trả Unknown
    if best_score < 2:
        return "Unknown"

    return best_role


def extract_education_signals(text: str) -> List[str]:
    lowered = text.lower()
    found = [kw for kw in EDUCATION_KEYWORDS if kw in lowered]
    # loại trùng
    result = []
    seen = set()
    for item in found:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def guess_experience_years(text: str) -> str:
    lowered = text.lower()
    
    # tìm pattern dạng 1 year / 2 years / 3 năm
    patterns = [
        r"(\d+)\+?\s*(?:years|year)",
        r"(\d+)\+?\s*năm",
    ]
    values = []
    for pattern in patterns:
        matches = re.findall(pattern, lowered)
        for m in matches:
            try:
                values.append(int(m))
            except ValueError:
                pass

    if values:
        return str(max(values))

    # heuristic cơ bản
    if any(x in lowered for x in ["intern", "fresher", "sinh viên", "new graduate", "recent graduate"]):
        return "0"

    return "Unknown"


def summarize_projects(text: str) -> List[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    result = []

    project_keywords = ["project", "dự án", "dashboard"]
    noisy_keywords = ["project management", "managed projects", "scheduling"]

    for line in lines:
        lowered = line.lower()

        if any(k in lowered for k in noisy_keywords):
            continue

        if any(k in lowered for k in project_keywords):
            if 15 <= len(line) <= 180:
                result.append(line)

    unique = []
    seen = set()
    for item in result:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique[:5]
def extract_relevant_skill_text(raw_text: str) -> str:
    """
    Ưu tiên lấy text từ các section có khả năng chứa skill.
    Nếu không tìm thấy, fallback về toàn văn bản.
    """
    text = raw_text.replace("\r", "\n")
    lowered = text.lower()

    section_keywords = [
        "skills", "technical skills", "core competencies",
        "kỹ năng", "công cụ", "technologies", "tools"
    ]

    lines = text.splitlines()
    selected_lines = []

    capture = False
    for line in lines:
        line_clean = line.strip()
        line_lower = line_clean.lower()

        if any(k in line_lower for k in section_keywords):
            capture = True
            selected_lines.append(line_clean)
            continue

        # dừng khi gặp section mới khá rõ
        if capture and any(
            x in line_lower for x in [
                "experience", "education", "summary", "about me",
                "work history", "projects", "certifications",
                "học vấn", "kinh nghiệm", "dự án"
            ]
        ):
            capture = False

        if capture and line_clean:
            selected_lines.append(line_clean)

    if selected_lines:
        return "\n".join(selected_lines)

    return raw_text

def extract_cv_info(file_path: str) -> Dict:
    raw_text = load_cv_text(file_path)
    cleaned_text = normalize_text(raw_text)

    skill_catalog = load_skill_catalog()
    skill_text = extract_relevant_skill_text(raw_text)
    skills = extract_skills(skill_text, skill_catalog)
    target_role = guess_target_role(cleaned_text, skills)
    education_signals = extract_education_signals(cleaned_text)
    experience_years = guess_experience_years(cleaned_text)
    projects = summarize_projects(raw_text)

    result = {
        "file_name": os.path.basename(file_path),
        "email": extract_email(cleaned_text),
        "phone": extract_phone(cleaned_text),
        "skills": skills,
        "target_role": target_role,
        "experience_years": experience_years,
        "education_signals": education_signals,
        "projects": projects,
        "raw_text_preview": cleaned_text[:1000],
    }
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cv_path", required=True, help="Path to CV file (.pdf, .docx, .txt)")
    parser.add_argument("--output_path", default="", help="Optional path to save extracted JSON")
    args = parser.parse_args()

    result = extract_cv_info(args.cv_path)

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON to: {args.output_path}")


if __name__ == "__main__":
    main()
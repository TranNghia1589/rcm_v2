from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "processed" / "jobs_merged_cleaned.csv"
OUTPUT_PATH = BASE_DIR / "data" / "role_profiles" / "role_profiles.json"

TOP_KEYWORDS = 30

STOPWORDS = {
    "và", "là", "có", "cho", "trong", "với", "các", "một", "được", "tại",
    "the", "and", "for", "with", "from", "you", "your", "our", "will",
    "yêu", "cầu", "mô", "tả", "quyền", "lợi", "kinh", "nghiệm", "việc",
    "công", "ty", "ứng", "viên", "khả", "năng", "làm", "năm", "job",
    "company", "work", "team", "experience", "skills", "data"
}

KNOWN_SKILLS = {
    "python": "Python",
    "sql": "SQL",
    "excel": "Excel",
    "power bi": "Power BI",
    "powerbi": "Power BI",
    "tableau": "Tableau",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "machine learning": "Machine Learning",
    "deep learning": "Deep Learning",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "scikit-learn": "Scikit-learn",
    "sklearn": "Scikit-learn",
    "spark": "Spark",
    "airflow": "Airflow",
    "etl": "ETL",
    "nlp": "NLP",
    "computer vision": "Computer Vision",
    "statistics": "Statistics",
    "statistical analysis": "Statistics",
    "data visualization": "Data Visualization",
    "dashboard": "Dashboarding",
    "dashboarding": "Dashboarding",
    "git": "Git",
    "docker": "Docker",
    "linux": "Linux",
    "mysql": "MySQL",
    "postgresql": "PostgreSQL",
    "postgres": "PostgreSQL",
    "mongodb": "MongoDB",
    "aws": "AWS",
    "azure": "Azure",
    "gcp": "GCP",
    "llm": "LLM",
    "rag": "RAG",
    "langchain": "LangChain",
    "streamlit": "Streamlit",
    "flask": "Flask",
    "fastapi": "FastAPI",
    "data analysis": "Data Analysis",
    "analysis": "Data Analysis",
    "annotation": "Annotation",
    "data labeling": "Data Labeling",
    "quality control": "Quality Control",
    "research": "Research",
    "experiment": "Experiment Design",
    "paper reading": "Paper Reading",
    "data warehouse": "Data Warehouse",
    "cloud": "Cloud Computing",
    "cloud computing": "Cloud Computing",
}

ROLE_DEFAULT_SKILLS = {
    "Data Analyst": ["SQL", "Excel", "Power BI", "Python", "Statistics", "Dashboarding"],
    "Data Engineer": ["Python", "SQL", "ETL", "Airflow", "Spark", "Data Warehouse"],
    "AI Engineer": ["Python", "Machine Learning", "Deep Learning", "PyTorch", "TensorFlow", "LLM"],
    "AI Researcher": ["Research", "Paper Reading", "Experiment Design", "Machine Learning", "Deep Learning", "Statistics"],
    "Data Scientist": ["Python", "Machine Learning", "Statistics", "Pandas", "NumPy"],
    "Data Labeling": ["Annotation", "Data Labeling", "Quality Control"],
}


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[^a-zA-ZÀ-ỹ0-9+#./ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    text = clean_text(text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


def split_tags(tag_text: str) -> List[str]:
    if not tag_text or pd.isna(tag_text):
        return []
    parts = re.split(r"[,;/|]", str(tag_text))
    return [p.strip() for p in parts if p.strip()]


def normalize_tag_to_skill(tag: str) -> str:
    key = clean_text(tag)
    return KNOWN_SKILLS.get(key, "")


def infer_recommended_next_skills(common_skills: List[str], role_name: str) -> List[str]:
    defaults = ROLE_DEFAULT_SKILLS.get(role_name, [])
    result = []
    seen = set()

    for item in common_skills + defaults:
        k = item.lower()
        if k not in seen:
            seen.add(k)
            result.append(item)

    return result[:8]


def main() -> None:
    print("Reading input from:", INPUT_PATH)
    print("Writing output to:", OUTPUT_PATH)

    df = pd.read_csv(INPUT_PATH)

    df["profile_text"] = (
        df["desc_mota"].fillna("") + " " +
        df["desc_yeucau"].fillna("") + " " +
        df["tags"].fillna("")
    )

    role_profiles: Dict[str, Dict] = {}

    for role, group in df.groupby("source_role"):
        tag_counter = Counter()
        keyword_counter = Counter()
        exp_counter = Counter()

        for _, row in group.iterrows():
            # Chỉ lấy tag nếu map được sang skill thật
            tags = split_tags(row.get("tags", ""))
            for tag in tags:
                skill = normalize_tag_to_skill(tag)
                if skill:
                    tag_counter[skill] += 1

            # Keyword để mô tả role, không phải skill
            tokens = tokenize(row.get("profile_text", ""))
            for token in tokens:
                if token not in KNOWN_SKILLS:
                    keyword_counter[token] += 1

            exp = str(row.get("detail_experience", "")).strip()
            if exp:
                exp_counter[exp] += 1

        common_skills = [k for k, _ in tag_counter.most_common()]
        if not common_skills:
            common_skills = ROLE_DEFAULT_SKILLS.get(role, [])

        common_keywords = [k for k, _ in keyword_counter.most_common(TOP_KEYWORDS)]
        common_experience = [k for k, _ in exp_counter.most_common(5)]

        role_profiles[role] = {
            "role_name": role,
            "job_count": int(len(group)),
            "common_skills": common_skills,
            "common_keywords": common_keywords,
            "common_experience_patterns": common_experience,
            "recommended_next_skills": infer_recommended_next_skills(common_skills, role),
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(role_profiles, f, ensure_ascii=False, indent=2)

    print(f"Saved role profiles to: {OUTPUT_PATH}")
    print(json.dumps(role_profiles, ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    main()
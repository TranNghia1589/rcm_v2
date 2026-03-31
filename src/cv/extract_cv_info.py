from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

import fitz  # PyMuPDF
from docx import Document

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data_contracts.schemas import CVExtractedRecord
from src.data_contracts.validators import normalize_cv_record, validate_cv_record

SKILL_CATALOG_PATH = BASE_DIR / "data" / "reference" / "skill_catalog.json"

ROLE_KEYWORDS = {
    "Data Analyst": ["data analyst", "bi analyst", "business intelligence", "power bi", "tableau", "dashboard"],
    "Data Engineer": ["data engineer", "etl", "data pipeline", "data warehouse", "airflow", "spark"],
    "AI Engineer": ["ai engineer", "machine learning engineer", "ml engineer", "model deployment", "llm"],
    "AI Researcher": ["ai researcher", "research scientist", "research assistant", "paper", "experiment"],
    "Data Scientist": ["data scientist", "machine learning", "predictive modeling", "statistical modeling"],
}

EDUCATION_KEYWORDS = [
    "university",
    "college",
    "bachelor",
    "master",
    "phd",
    "data science",
    "computer science",
    "information technology",
]

SECTION_KEYWORDS: dict[str, list[str]] = {
    "summary": ["summary", "profile", "objective", "career objective", "about me"],
    "skills": ["skills", "technical skills", "core competencies", "technologies", "tools"],
    "education": ["education", "academic", "qualification", "study"],
    "experience": ["experience", "employment", "work history", "professional experience"],
    "projects": ["project", "projects"],
    "languages": ["language", "languages"],
    "certifications": ["certification", "certifications", "certificate", "licenses"],
    "activities": ["activities", "volunteer", "extracurricular", "leadership"],
}

KNOWN_LANGUAGES = {
    "english",
    "vietnamese",
    "japanese",
    "korean",
    "chinese",
    "french",
    "german",
    "spanish",
    "thai",
}

DATE_RANGE_PATTERN = re.compile(
    r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{4}|\d{1,2}/\d{4}|\d{4})"
    r"\s*(?:-|to|until|–|—)\s*"
    r"((?:present|current|now)|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{4}|\d{1,2}/\d{4}|\d{4})",
    flags=re.IGNORECASE,
)

YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
URL_PATTERN = re.compile(r"https?://[^\s)>,]+", flags=re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_PATTERN = re.compile(r"(\+?\d[\d\s\-().]{8,}\d)")


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
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _uniq(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        val = str(item).strip(" -\t")
        if not val:
            continue
        key = val.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(val)
    return out


def _split_lines(raw_text: str) -> list[str]:
    lines = [x.strip() for x in normalize_text(raw_text).split("\n")]
    return [x for x in lines if x]


def _is_header(line: str) -> bool:
    txt = line.strip().lower().rstrip(":")
    if not txt or len(txt) > 50:
        return False
    return any(txt == kw or txt.startswith(f"{kw} ") for kws in SECTION_KEYWORDS.values() for kw in kws)


def _header_to_section(line: str) -> str | None:
    txt = line.strip().lower().rstrip(":")
    for section, keywords in SECTION_KEYWORDS.items():
        for kw in keywords:
            if txt == kw or txt.startswith(f"{kw} "):
                return section
    return None


def split_sections(raw_text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {"all": _split_lines(raw_text)}
    current = "all"
    for line in sections["all"]:
        if _is_header(line):
            sec = _header_to_section(line)
            if sec:
                current = sec
                sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return sections


def extract_email(text: str) -> str:
    m = EMAIL_PATTERN.search(text)
    return m.group(0) if m else ""


def extract_phone(text: str) -> str:
    m = PHONE_PATTERN.search(text)
    return m.group(0).strip() if m else ""


def load_skill_catalog() -> Dict[str, List[str]]:
    with open(SKILL_CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_skills(text: str, skill_catalog: Dict[str, List[str]]) -> List[str]:
    lowered = text.lower()
    found: list[str] = []
    for canonical_skill, aliases in skill_catalog.items():
        for alias in aliases:
            pattern = r"\b" + re.escape(alias.lower()) + r"\b"
            if re.search(pattern, lowered):
                found.append(canonical_skill)
                break
    return _uniq(found)


def guess_target_role(text: str, skills: List[str]) -> str:
    lowered = text.lower()
    role_scores: dict[str, int] = {}
    for role, keywords in ROLE_KEYWORDS.items():
        score = sum(2 for kw in keywords if kw in lowered)
        role_scores[role] = score
    best_role = max(role_scores, key=role_scores.get)
    return best_role if role_scores[best_role] >= 2 else "Unknown"


def extract_education_signals(text: str) -> List[str]:
    lowered = text.lower()
    return _uniq([kw for kw in EDUCATION_KEYWORDS if kw in lowered])


def guess_experience_years(text: str) -> str:
    lowered = text.lower()
    values: list[int] = []
    for p in [r"(\d+)\+?\s*(?:years|year)", r"(\d+)\+?\s*năm"]:
        for m in re.findall(p, lowered):
            try:
                values.append(int(m))
            except ValueError:
                pass
    if values:
        return str(max(values))
    if any(x in lowered for x in ["intern", "fresher", "new graduate", "recent graduate"]):
        return "0"
    return "Unknown"


def summarize_projects(lines: list[str]) -> List[str]:
    result: list[str] = []
    for line in lines:
        l = line.lower()
        if "project management" in l:
            continue
        if any(k in l for k in ["project", "dashboard", "system", "application"]):
            if 15 <= len(line) <= 220:
                result.append(line)
    return _uniq(result)[:8]


def _extract_links(lines: list[str]) -> list[str]:
    links: list[str] = []
    for line in lines:
        links.extend(URL_PATTERN.findall(line))
        low = line.lower()
        if "linkedin.com/" in low or "github.com/" in low:
            links.append(line.strip())
    return _uniq(links)


def _extract_address(lines: list[str]) -> str:
    for line in lines[:25]:
        low = line.lower()
        if any(x in low for x in ["street", "city", "district", "ward", "ho chi minh", "ha noi", "hanoi"]):
            return line
        if "," in line and any(ch.isdigit() for ch in line):
            return line
    return ""


def _extract_education_fields(lines: list[str]) -> dict[str, list[str]]:
    institutions: list[str] = []
    degrees: list[str] = []
    majors: list[str] = []
    years: list[str] = []
    edu_results: list[str] = []
    result_types: list[str] = []

    degree_pat = re.compile(r"\b(bachelor|master|phd|doctor|engineer|b\.?sc|m\.?sc|mba)\b", re.IGNORECASE)
    institution_pat = re.compile(r"\b(university|college|academy|institute|school|polytechnic)\b", re.IGNORECASE)
    major_pat = re.compile(r"\b(major|specialization|specialised in|specialized in|in)\b", re.IGNORECASE)
    gpa_pat = re.compile(r"\b(gpa|cgpa)\b", re.IGNORECASE)

    for line in lines:
        if institution_pat.search(line):
            institutions.append(line)
        if degree_pat.search(line):
            degrees.append(line)
        if major_pat.search(line) and len(line) <= 140:
            majors.append(line)
        if gpa_pat.search(line):
            edu_results.append(line)
            result_types.append("GPA")
        years.extend(m.group(0) for m in YEAR_PATTERN.finditer(line))

    return {
        "educational_institution_name": _uniq(institutions)[:6],
        "degree_names": _uniq(degrees)[:8],
        "major_field_of_studies": _uniq(majors)[:8],
        "passing_years": _uniq(years)[:8],
        "educational_results": _uniq(edu_results)[:8],
        "result_types": _uniq(result_types)[:8],
    }


def _extract_experience_fields(lines: list[str], skill_catalog: Dict[str, List[str]]) -> dict[str, list[str]]:
    companies: list[str] = []
    positions: list[str] = []
    starts: list[str] = []
    ends: list[str] = []
    locations: list[str] = []
    responsibilities: list[str] = []

    company_pat = re.compile(r"\b(inc|llc|ltd|limited|corp|corporation|company|co\.|jsc|group|solutions|technology|tech|software|bank)\b", re.IGNORECASE)
    position_pat = re.compile(r"\b(engineer|developer|analyst|intern|manager|lead|specialist|consultant|architect|tester|qa|devops|scientist|programmer)\b", re.IGNORECASE)
    location_pat = re.compile(r"\b(city|district|remote|hanoi|ha noi|ho chi minh|hcm|da nang|vietnam|viet nam)\b", re.IGNORECASE)

    for line in lines:
        if company_pat.search(line) and len(line) <= 120:
            companies.append(line)
        if position_pat.search(line) and len(line) <= 140:
            positions.append(line)
        if location_pat.search(line) and len(line) <= 120:
            locations.append(line)
        if line.startswith(("-", "*", "•")) or line.lower().startswith(("responsible", "built", "developed", "implemented", "designed", "optimized")):
            responsibilities.append(line)

        for s, e in DATE_RANGE_PATTERN.findall(line):
            starts.append(s)
            ends.append(e)

    related = extract_skills("\n".join(lines), skill_catalog)

    return {
        "professional_company_names": _uniq(companies)[:10],
        "positions": _uniq(positions)[:12],
        "start_dates": _uniq(starts)[:12],
        "end_dates": _uniq(ends)[:12],
        "locations": _uniq(locations)[:12],
        "responsibilities": _uniq(responsibilities)[:20],
        "related_skils_in_job": _uniq(related)[:20],
    }


def _extract_languages(lines: list[str]) -> tuple[list[str], list[str]]:
    langs: list[str] = []
    profs: list[str] = []

    for line in lines:
        chunks = re.split(r"[,;|/]", line)
        for chunk in chunks:
            low = chunk.lower()
            for lang in KNOWN_LANGUAGES:
                if re.search(rf"\b{re.escape(lang)}\b", low):
                    langs.append(lang.title())
                    if any(k in low for k in ["native", "fluent", "intermediate", "basic", "advanced", "ielts", "toeic"]):
                        profs.append(chunk.strip())

    return _uniq(langs), _uniq(profs)


def _extract_certifications(lines: list[str], skill_catalog: Dict[str, List[str]]) -> dict[str, list[str]]:
    providers: list[str] = []
    cert_skills: list[str] = []
    issue_dates: list[str] = []
    expiry_dates: list[str] = []

    provider_pat = re.compile(r"\b(coursera|udemy|edx|aws|azure|google|oracle|microsoft|ibm|meta|datacamp)\b", re.IGNORECASE)

    for line in lines:
        if provider_pat.search(line):
            providers.append(line)
        cert_skills.extend(extract_skills(line, skill_catalog))
        years = [m.group(0) for m in YEAR_PATTERN.finditer(line)]
        if years:
            issue_dates.append(years[0])
            if len(years) > 1:
                expiry_dates.append(years[-1])

    return {
        "certification_providers": _uniq(providers)[:10],
        "certification_skills": _uniq(cert_skills)[:15],
        "issue_dates": _uniq(issue_dates)[:10],
        "expiry_dates": _uniq(expiry_dates)[:10],
    }


def _extract_career_objective(sections: dict[str, list[str]]) -> str:
    for key in ["summary", "all"]:
        lines = sections.get(key, [])
        if not lines:
            continue
        for line in lines[:10]:
            if 30 <= len(line) <= 350:
                return line
    return ""


def extract_cv_info(file_path: str) -> Dict:
    raw_text = load_cv_text(file_path)
    cleaned_text = normalize_text(raw_text)
    sections = split_sections(raw_text)

    skill_catalog = load_skill_catalog()
    skills = extract_skills("\n".join(sections.get("skills", []) or sections.get("all", [])), skill_catalog)
    target_role = guess_target_role(cleaned_text, skills)
    education_signals = extract_education_signals(cleaned_text)
    experience_years = guess_experience_years(cleaned_text)

    edu_lines = sections.get("education", [])
    exp_lines = sections.get("experience", [])
    proj_lines = sections.get("projects", []) or sections.get("all", [])
    lang_lines = sections.get("languages", [])
    cert_lines = sections.get("certifications", [])
    act_lines = sections.get("activities", [])

    edu_fields = _extract_education_fields(edu_lines)
    exp_fields = _extract_experience_fields(exp_lines, skill_catalog)
    langs, profs = _extract_languages(lang_lines)
    cert_fields = _extract_certifications(cert_lines, skill_catalog)

    links = _extract_links(sections.get("all", []))
    companies_as_urls = [x for x in links if "linkedin.com" not in x.lower() and "github.com" not in x.lower()]

    projects = summarize_projects(proj_lines)
    career_objective = _extract_career_objective(sections)

    file_name = os.path.basename(file_path)
    cv_id = hashlib.md5(file_name.encode("utf-8")).hexdigest()[:12]

    record = CVExtractedRecord(
        cv_id=cv_id,
        file_name=file_name,
        source_path=str(Path(file_path).resolve()),
        email=extract_email(cleaned_text),
        phone=extract_phone(cleaned_text),
        address=_extract_address(sections.get("all", [])),
        career_objective=career_objective,
        skills=skills,
        target_role=target_role,
        experience_years=experience_years,
        projects=projects,
        raw_text_preview=cleaned_text[:1000],
        education_signals=education_signals,
        educational_institution_name=edu_fields["educational_institution_name"],
        degree_names=edu_fields["degree_names"],
        passing_years=edu_fields["passing_years"],
        educational_results=edu_fields["educational_results"],
        result_types=edu_fields["result_types"],
        major_field_of_studies=edu_fields["major_field_of_studies"],
        professional_company_names=exp_fields["professional_company_names"],
        company_urls=companies_as_urls,
        start_dates=exp_fields["start_dates"],
        end_dates=exp_fields["end_dates"],
        positions=exp_fields["positions"],
        locations=exp_fields["locations"],
        responsibilities=exp_fields["responsibilities"],
        related_skils_in_job=exp_fields["related_skils_in_job"],
        extra_curricular_activity_types=_uniq(act_lines)[:8],
        extra_curricular_organization_names=_uniq(act_lines)[:8],
        extra_curricular_organization_links=[],
        role_positions=_uniq(exp_fields["positions"])[:8],
        languages=langs,
        proficiency_levels=profs,
        certification_providers=cert_fields["certification_providers"],
        certification_skills=cert_fields["certification_skills"],
        online_links=links,
        issue_dates=cert_fields["issue_dates"],
        expiry_dates=cert_fields["expiry_dates"],
        job_position_name=target_role,
        responsibilities_job=" ".join(exp_fields["responsibilities"][:3]),
        skills_required=skills,
        metadata={"extractor": "heuristic_sections_v2"},
    )

    normalized = normalize_cv_record(record.to_dict())
    errors = validate_cv_record(normalized)
    if errors:
        normalized.setdefault("metadata", {})
        normalized["metadata"]["validation_errors"] = errors
    return normalized


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cv_path", required=True, help="Path to CV file (.pdf, .docx, .txt)")
    parser.add_argument("--output_path", default="", help="Optional path to save extracted JSON")
    args = parser.parse_args()

    result = extract_cv_info(args.cv_path)

    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON to: {args.output_path}")

    if sys.platform == "win32":
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        except Exception:
            pass
    try:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except UnicodeEncodeError:
        print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

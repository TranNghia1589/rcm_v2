from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = "cv_extracted.v1"


@dataclass
class CVExtractedRecord:
    # Core identity
    schema_version: str = SCHEMA_VERSION
    cv_id: str = ""
    file_name: str = ""
    source_path: str = ""

    # Contact
    email: str = ""
    phone: str = ""
    address: str = ""

    # Core profile
    career_objective: str = ""
    skills: list[str] = field(default_factory=list)
    target_role: str = "Unknown"
    experience_years: str = "Unknown"
    projects: list[str] = field(default_factory=list)
    raw_text_preview: str = ""

    # Education
    education_signals: list[str] = field(default_factory=list)
    educational_institution_name: list[str] = field(default_factory=list)
    degree_names: list[str] = field(default_factory=list)
    passing_years: list[str] = field(default_factory=list)
    educational_results: list[str] = field(default_factory=list)
    result_types: list[str] = field(default_factory=list)
    major_field_of_studies: list[str] = field(default_factory=list)

    # Experience
    professional_company_names: list[str] = field(default_factory=list)
    company_urls: list[str] = field(default_factory=list)
    start_dates: list[str] = field(default_factory=list)
    end_dates: list[str] = field(default_factory=list)
    positions: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    responsibilities: list[str] = field(default_factory=list)
    related_skils_in_job: list[str] = field(default_factory=list)

    # Extra activities
    extra_curricular_activity_types: list[str] = field(default_factory=list)
    extra_curricular_organization_names: list[str] = field(default_factory=list)
    extra_curricular_organization_links: list[str] = field(default_factory=list)
    role_positions: list[str] = field(default_factory=list)

    # Languages / certificates
    languages: list[str] = field(default_factory=list)
    proficiency_levels: list[str] = field(default_factory=list)
    certification_providers: list[str] = field(default_factory=list)
    certification_skills: list[str] = field(default_factory=list)
    online_links: list[str] = field(default_factory=list)
    issue_dates: list[str] = field(default_factory=list)
    expiry_dates: list[str] = field(default_factory=list)

    # Optional job-match style fields (kept for compatibility with external datasets)
    job_position_name: str = ""
    educational_requirements: str = ""
    experience_requirement: str = ""
    age_requirement: str = ""
    responsibilities_job: str = ""
    skills_required: list[str] = field(default_factory=list)
    matched_score: float | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

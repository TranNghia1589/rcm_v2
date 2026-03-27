from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UserNode:
    user_id: int
    email: str
    full_name: str
    phone: str


@dataclass
class CVNode:
    cv_id: int
    user_id: int
    file_name: str
    target_role: str
    experience_years: float | None


@dataclass
class SkillNode:
    skill_id: int
    canonical_name: str
    skill_group: str


@dataclass
class JobNode:
    job_id: int
    title: str
    company_name: str
    location: str
    job_family: str
    experience_min_years: float | None
    experience_max_years: float | None

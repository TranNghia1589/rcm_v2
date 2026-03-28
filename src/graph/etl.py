from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.infrastructure.db.neo4j_client import Neo4jClient, Neo4jConfig
from src.infrastructure.db.postgres_client import PostgresClient, PostgresConfig


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return [text]
    return []


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


@dataclass
class GraphETLStats:
    users: int = 0
    cvs: int = 0
    skills: int = 0
    jobs: int = 0
    cv_skill_links: int = 0
    job_skill_links: int = 0
    role_links: int = 0
    lack_skill_links: int = 0
    education_links: int = 0
    experience_links: int = 0
    project_links: int = 0
    certification_links: int = 0
    language_links: int = 0


class GraphETL:
    def __init__(self, postgres_cfg_path: str | Path, neo4j_cfg_path: str | Path) -> None:
        self.postgres_cfg_path = postgres_cfg_path
        self.neo4j_cfg_path = neo4j_cfg_path

    def _apply_schema(self, neo: Neo4jClient, root_dir: Path) -> None:
        constraints = root_dir / "database" / "neo4j" / "schema" / "constraints.cypher"
        indexes = root_dir / "database" / "neo4j" / "schema" / "indexes.cypher"
        for file_path in [constraints, indexes]:
            if not file_path.exists():
                continue
            content = file_path.read_text(encoding="utf-8")
            statements = [x.strip() for x in content.split(";") if x.strip()]
            for stmt in statements:
                neo.execute(stmt)

    def _upsert_users(self, pg: PostgresClient, neo: Neo4jClient) -> int:
        rows = pg.fetch_all(
            """
            SELECT user_id, email, full_name, phone
            FROM users
            """
        )
        count = 0
        for user_id, email, full_name, phone in rows:
            neo.execute(
                """
                MERGE (u:User {user_id: $user_id})
                SET u.email = $email,
                    u.full_name = $full_name,
                    u.phone = $phone
                """,
                {
                    "user_id": int(user_id),
                    "email": _safe_str(email),
                    "full_name": _safe_str(full_name),
                    "phone": _safe_str(phone),
                },
            )
            count += 1
        return count

    def _upsert_cvs(self, pg: PostgresClient, neo: Neo4jClient) -> int:
        rows = pg.fetch_all(
            """
            SELECT cv_id, user_id, file_name, target_role, experience_years,
                   address, career_objective, seniority_level, schema_version
            FROM cv_profiles
            """
        )
        count = 0
        for cv_id, user_id, file_name, target_role, experience_years, address, career_objective, seniority_level, schema_version in rows:
            neo.execute(
                """
                MERGE (cv:CV {cv_id: $cv_id})
                SET cv.file_name = $file_name,
                    cv.target_role = $target_role,
                    cv.experience_years = $experience_years,
                    cv.address = $address,
                    cv.career_objective = $career_objective,
                    cv.seniority_level = $seniority_level,
                    cv.schema_version = $schema_version
                WITH cv
                MATCH (u:User {user_id: $user_id})
                MERGE (u)-[:HAS_CV]->(cv)
                """,
                {
                    "cv_id": int(cv_id),
                    "user_id": int(user_id),
                    "file_name": _safe_str(file_name),
                    "target_role": _safe_str(target_role),
                    "experience_years": _safe_float(experience_years),
                    "address": _safe_str(address),
                    "career_objective": _safe_str(career_objective),
                    "seniority_level": _safe_str(seniority_level),
                    "schema_version": _safe_str(schema_version),
                },
            )
            count += 1
        return count

    def _upsert_skills(self, pg: PostgresClient, neo: Neo4jClient) -> int:
        rows = pg.fetch_all(
            """
            SELECT skill_id, canonical_name, skill_group
            FROM skills
            """
        )
        count = 0
        for skill_id, canonical_name, skill_group in rows:
            neo.execute(
                """
                MERGE (s:Skill {skill_id: $skill_id})
                SET s.canonical_name = $canonical_name,
                    s.skill_group = $skill_group
                """,
                {
                    "skill_id": int(skill_id),
                    "canonical_name": _safe_str(canonical_name),
                    "skill_group": _safe_str(skill_group),
                },
            )
            count += 1
        return count

    def _upsert_jobs(self, pg: PostgresClient, neo: Neo4jClient) -> int:
        rows = pg.fetch_all(
            """
            SELECT job_id, title, company_name, location, job_family, experience_min_years, experience_max_years
            FROM jobs
            """
        )
        count = 0
        for job_id, title, company_name, location, job_family, exp_min, exp_max in rows:
            neo.execute(
                """
                MERGE (j:Job {job_id: $job_id})
                SET j.title = $title,
                    j.company_name = $company_name,
                    j.location = $location,
                    j.job_family = $job_family,
                    j.experience_min_years = $exp_min,
                    j.experience_max_years = $exp_max
                """,
                {
                    "job_id": int(job_id),
                    "title": _safe_str(title),
                    "company_name": _safe_str(company_name),
                    "location": _safe_str(location),
                    "job_family": _safe_str(job_family),
                    "exp_min": _safe_float(exp_min),
                    "exp_max": _safe_float(exp_max),
                },
            )
            if _safe_str(company_name):
                neo.execute(
                    """
                    MERGE (c:Company {name: $company_name})
                    WITH c
                    MATCH (j:Job {job_id: $job_id})
                    MERGE (j)-[:AT_COMPANY]->(c)
                    """,
                    {
                        "company_name": _safe_str(company_name),
                        "job_id": int(job_id),
                    },
                )
            count += 1
        return count

    def _link_cv_skills(self, pg: PostgresClient, neo: Neo4jClient) -> int:
        rows = pg.fetch_all(
            """
            SELECT cv_id, skill_id, source, confidence
            FROM cv_skills
            """
        )
        count = 0
        for cv_id, skill_id, source, confidence in rows:
            neo.execute(
                """
                MATCH (cv:CV {cv_id: $cv_id})
                MATCH (s:Skill {skill_id: $skill_id})
                MERGE (cv)-[r:HAS_SKILL]->(s)
                SET r.source = $source,
                    r.confidence = $confidence
                """,
                {
                    "cv_id": int(cv_id),
                    "skill_id": int(skill_id),
                    "source": _safe_str(source),
                    "confidence": _safe_float(confidence),
                },
            )
            count += 1
        return count

    def _link_job_skills(self, pg: PostgresClient, neo: Neo4jClient) -> int:
        rows = pg.fetch_all(
            """
            SELECT job_id, skill_id, source_field, importance
            FROM job_skills
            """
        )
        count = 0
        for job_id, skill_id, source_field, importance in rows:
            neo.execute(
                """
                MATCH (j:Job {job_id: $job_id})
                MATCH (s:Skill {skill_id: $skill_id})
                MERGE (j)-[r:REQUIRES_SKILL]->(s)
                SET r.source_field = $source_field,
                    r.importance = $importance
                """,
                {
                    "job_id": int(job_id),
                    "skill_id": int(skill_id),
                    "source_field": _safe_str(source_field),
                    "importance": _safe_str(importance),
                },
            )
            count += 1
        return count

    def _link_roles_and_gaps(self, pg: PostgresClient, neo: Neo4jClient) -> tuple[int, int]:
        rows = pg.fetch_all(
            """
            SELECT cv_id, target_role_from_cv, best_fit_roles, missing_skills
            FROM cv_gap_reports
            ORDER BY gap_report_id DESC
            """
        )
        role_links = 0
        gap_links = 0

        for cv_id, target_role, best_fit_roles, missing_skills in rows:
            role_name = _safe_str(target_role)
            if role_name:
                neo.execute(
                    """
                    MERGE (r:Role {name: $role_name})
                    WITH r
                    MATCH (cv:CV {cv_id: $cv_id})
                    MERGE (cv)-[:TARGETS_ROLE]->(r)
                    """,
                    {"role_name": role_name, "cv_id": int(cv_id)},
                )
                role_links += 1

            best_roles = _to_list(best_fit_roles)
            for rank, role in enumerate(best_roles, start=1):
                role_n = _safe_str(role)
                if not role_n:
                    continue
                neo.execute(
                    """
                    MERGE (r:Role {name: $role_name})
                    WITH r
                    MATCH (cv:CV {cv_id: $cv_id})
                    MERGE (cv)-[rel:BEST_FIT_ROLE]->(r)
                    SET rel.rank = $rank
                    """,
                    {
                        "role_name": role_n,
                        "cv_id": int(cv_id),
                        "rank": int(rank),
                    },
                )
                role_links += 1

            miss = _to_list(missing_skills)
            for ms in miss:
                name = _safe_str(ms)
                if not name:
                    continue
                neo.execute(
                    """
                    MERGE (s:Skill {canonical_name: $name})
                    WITH s
                    MATCH (cv:CV {cv_id: $cv_id})
                    MERGE (cv)-[:LACKS_SKILL]->(s)
                    """,
                    {"name": name, "cv_id": int(cv_id)},
                )
                gap_links += 1

        return role_links, gap_links

    def _link_cv_educations(self, pg: PostgresClient, neo: Neo4jClient) -> int:
        rows = pg.fetch_all(
            """
            SELECT cv_education_id, cv_id, institution_name, degree_name, major_field, passing_year, educational_result, result_type
            FROM cv_educations
            """
        )
        count = 0
        for edu_id, cv_id, institution_name, degree_name, major_field, passing_year, educational_result, result_type in rows:
            inst_name = _safe_str(institution_name) or "Unknown Institution"
            inst_key = f"{inst_name.lower()}"
            neo.execute(
                """
                MERGE (i:Institution {inst_key: $inst_key})
                SET i.name = $institution_name
                WITH i
                MATCH (cv:CV {cv_id: $cv_id})
                MERGE (cv)-[r:STUDIED_AT]->(i)
                SET r.degree_name = $degree_name,
                    r.major_field = $major_field,
                    r.passing_year = $passing_year,
                    r.educational_result = $educational_result,
                    r.result_type = $result_type,
                    r.cv_education_id = $cv_education_id
                """,
                {
                    "inst_key": inst_key,
                    "institution_name": inst_name,
                    "cv_id": int(cv_id),
                    "degree_name": _safe_str(degree_name),
                    "major_field": _safe_str(major_field),
                    "passing_year": _safe_str(passing_year),
                    "educational_result": _safe_str(educational_result),
                    "result_type": _safe_str(result_type),
                    "cv_education_id": int(edu_id),
                },
            )
            count += 1
        return count

    def _link_cv_experiences(self, pg: PostgresClient, neo: Neo4jClient) -> int:
        rows = pg.fetch_all(
            """
            SELECT cv_experience_id, cv_id, company_name, position_name, start_date, end_date, location, responsibilities, related_skills
            FROM cv_experiences
            """
        )
        count = 0
        for exp_id, cv_id, company_name, position_name, start_date, end_date, location, responsibilities, related_skills in rows:
            exp_key = f"cv{int(cv_id)}-exp{int(exp_id)}"
            neo.execute(
                """
                MERGE (e:Experience {exp_key: $exp_key})
                SET e.company_name = $company_name,
                    e.position_name = $position_name,
                    e.start_date = $start_date,
                    e.end_date = $end_date,
                    e.location = $location
                WITH e
                MATCH (cv:CV {cv_id: $cv_id})
                MERGE (cv)-[:HAS_EXPERIENCE]->(e)
                """,
                {
                    "exp_key": exp_key,
                    "company_name": _safe_str(company_name),
                    "position_name": _safe_str(position_name),
                    "start_date": _safe_str(start_date),
                    "end_date": _safe_str(end_date),
                    "location": _safe_str(location),
                    "cv_id": int(cv_id),
                },
            )
            company = _safe_str(company_name)
            if company:
                neo.execute(
                    """
                    MERGE (c:Company {name: $company_name})
                    WITH c
                    MATCH (e:Experience {exp_key: $exp_key})
                    MERGE (e)-[:AT_COMPANY]->(c)
                    """,
                    {"company_name": company, "exp_key": exp_key},
                )

            for skill in _to_list(related_skills):
                s = _safe_str(skill)
                if not s:
                    continue
                neo.execute(
                    """
                    MERGE (sk:Skill {canonical_name: $skill_name})
                    WITH sk
                    MATCH (e:Experience {exp_key: $exp_key})
                    MERGE (e)-[:USED_SKILL]->(sk)
                    """,
                    {"skill_name": s, "exp_key": exp_key},
                )
            _ = responsibilities
            count += 1
        return count

    def _link_cv_projects(self, pg: PostgresClient, neo: Neo4jClient) -> int:
        rows = pg.fetch_all(
            """
            SELECT cv_project_id, cv_id, project_name, project_description, tech_stack, impact_summary
            FROM cv_projects
            """
        )
        count = 0
        for project_id, cv_id, project_name, project_description, tech_stack, impact_summary in rows:
            p_name = _safe_str(project_name) or f"Project {project_id}"
            p_key = f"cv{int(cv_id)}-project{int(project_id)}"
            neo.execute(
                """
                MERGE (p:Project {project_key: $project_key})
                SET p.name = $name,
                    p.description = $description,
                    p.impact_summary = $impact_summary
                WITH p
                MATCH (cv:CV {cv_id: $cv_id})
                MERGE (cv)-[:HAS_PROJECT]->(p)
                """,
                {
                    "project_key": p_key,
                    "name": p_name,
                    "description": _safe_str(project_description),
                    "impact_summary": _safe_str(impact_summary),
                    "cv_id": int(cv_id),
                },
            )

            for skill in _to_list(tech_stack):
                s = _safe_str(skill)
                if not s:
                    continue
                neo.execute(
                    """
                    MERGE (sk:Skill {canonical_name: $skill_name})
                    WITH sk
                    MATCH (p:Project {project_key: $project_key})
                    MERGE (p)-[:USES_SKILL]->(sk)
                    """,
                    {"skill_name": s, "project_key": p_key},
                )
            count += 1
        return count

    def _link_cv_certifications(self, pg: PostgresClient, neo: Neo4jClient) -> int:
        rows = pg.fetch_all(
            """
            SELECT cv_certification_id, cv_id, certification_name, provider, issue_date, expiry_date, certification_skills
            FROM cv_certifications
            """
        )
        count = 0
        for cert_id, cv_id, certification_name, provider, issue_date, expiry_date, cert_skills in rows:
            c_name = _safe_str(certification_name) or f"Certification {cert_id}"
            cert_key = f"cv{int(cv_id)}-cert{int(cert_id)}"
            neo.execute(
                """
                MERGE (c:Certification {cert_key: $cert_key})
                SET c.name = $name,
                    c.provider = $provider,
                    c.issue_date = $issue_date,
                    c.expiry_date = $expiry_date
                WITH c
                MATCH (cv:CV {cv_id: $cv_id})
                MERGE (cv)-[:HAS_CERTIFICATION]->(c)
                """,
                {
                    "cert_key": cert_key,
                    "name": c_name,
                    "provider": _safe_str(provider),
                    "issue_date": _safe_str(issue_date),
                    "expiry_date": _safe_str(expiry_date),
                    "cv_id": int(cv_id),
                },
            )

            for skill in _to_list(cert_skills):
                s = _safe_str(skill)
                if not s:
                    continue
                neo.execute(
                    """
                    MERGE (sk:Skill {canonical_name: $skill_name})
                    WITH sk
                    MATCH (c:Certification {cert_key: $cert_key})
                    MERGE (c)-[:CERTIFIES_SKILL]->(sk)
                    """,
                    {"skill_name": s, "cert_key": cert_key},
                )
            count += 1
        return count

    def _link_cv_languages(self, pg: PostgresClient, neo: Neo4jClient) -> int:
        rows = pg.fetch_all(
            """
            SELECT cv_id, language_name, proficiency_level
            FROM cv_languages
            """
        )
        count = 0
        for cv_id, language_name, proficiency_level in rows:
            lang = _safe_str(language_name)
            if not lang:
                continue
            neo.execute(
                """
                MERGE (l:Language {name: $language_name})
                WITH l
                MATCH (cv:CV {cv_id: $cv_id})
                MERGE (cv)-[r:SPEAKS_LANGUAGE]->(l)
                SET r.proficiency = $proficiency
                """,
                {
                    "language_name": lang,
                    "cv_id": int(cv_id),
                    "proficiency": _safe_str(proficiency_level),
                },
            )
            count += 1
        return count

    def run(self, *, reset_graph: bool = False) -> GraphETLStats:
        root_dir = Path(__file__).resolve().parents[2]
        pg_conf = PostgresConfig.from_yaml(self.postgres_cfg_path)
        neo_conf = Neo4jConfig.from_yaml(self.neo4j_cfg_path)
        stats = GraphETLStats()

        with PostgresClient(pg_conf) as pg, Neo4jClient(neo_conf) as neo:
            self._apply_schema(neo, root_dir)

            if reset_graph:
                neo.execute("MATCH (n) DETACH DELETE n")
                self._apply_schema(neo, root_dir)

            stats.users = self._upsert_users(pg, neo)
            stats.cvs = self._upsert_cvs(pg, neo)
            stats.skills = self._upsert_skills(pg, neo)
            stats.jobs = self._upsert_jobs(pg, neo)
            stats.cv_skill_links = self._link_cv_skills(pg, neo)
            stats.job_skill_links = self._link_job_skills(pg, neo)
            stats.role_links, stats.lack_skill_links = self._link_roles_and_gaps(pg, neo)
            stats.education_links = self._link_cv_educations(pg, neo)
            stats.experience_links = self._link_cv_experiences(pg, neo)
            stats.project_links = self._link_cv_projects(pg, neo)
            stats.certification_links = self._link_cv_certifications(pg, neo)
            stats.language_links = self._link_cv_languages(pg, neo)

        return stats

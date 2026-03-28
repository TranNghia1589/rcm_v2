CREATE INDEX user_email_idx IF NOT EXISTS
FOR (u:User) ON (u.email);

CREATE INDEX job_title_idx IF NOT EXISTS
FOR (j:Job) ON (j.title);

CREATE INDEX job_job_family_idx IF NOT EXISTS
FOR (j:Job) ON (j.job_family);

CREATE INDEX cv_target_role_idx IF NOT EXISTS
FOR (c:CV) ON (c.target_role);

CREATE INDEX cv_seniority_idx IF NOT EXISTS
FOR (c:CV) ON (c.seniority_level);

CREATE INDEX project_name_idx IF NOT EXISTS
FOR (p:Project) ON (p.name);

CREATE INDEX certification_name_idx IF NOT EXISTS
FOR (c:Certification) ON (c.name);

CREATE INDEX experience_company_idx IF NOT EXISTS
FOR (e:Experience) ON (e.company_name);

CREATE INDEX user_email_idx IF NOT EXISTS
FOR (u:User) ON (u.email);

CREATE INDEX job_title_idx IF NOT EXISTS
FOR (j:Job) ON (j.title);

CREATE INDEX job_job_family_idx IF NOT EXISTS
FOR (j:Job) ON (j.job_family);

CREATE INDEX cv_target_role_idx IF NOT EXISTS
FOR (c:CV) ON (c.target_role);

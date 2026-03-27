CREATE CONSTRAINT user_user_id_unique IF NOT EXISTS
FOR (u:User) REQUIRE u.user_id IS UNIQUE;

CREATE CONSTRAINT cv_cv_id_unique IF NOT EXISTS
FOR (c:CV) REQUIRE c.cv_id IS UNIQUE;

CREATE CONSTRAINT skill_skill_id_unique IF NOT EXISTS
FOR (s:Skill) REQUIRE s.skill_id IS UNIQUE;

CREATE CONSTRAINT skill_name_unique IF NOT EXISTS
FOR (s:Skill) REQUIRE s.canonical_name IS UNIQUE;

CREATE CONSTRAINT job_job_id_unique IF NOT EXISTS
FOR (j:Job) REQUIRE j.job_id IS UNIQUE;

CREATE CONSTRAINT company_name_unique IF NOT EXISTS
FOR (c:Company) REQUIRE c.name IS UNIQUE;

CREATE CONSTRAINT role_name_unique IF NOT EXISTS
FOR (r:Role) REQUIRE r.name IS UNIQUE;

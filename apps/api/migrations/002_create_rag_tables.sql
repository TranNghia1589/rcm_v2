-- Fresh schema for a new PostgreSQL database (RAG + CV matching + recommendation)

-- 1) Pipeline / ETL tracking
CREATE TABLE IF NOT EXISTS etl_runs (
    etl_run_id BIGSERIAL PRIMARY KEY,
    pipeline_name TEXT NOT NULL,
    data_version TEXT,
    source_path TEXT,
    status TEXT NOT NULL DEFAULT 'success' CHECK (status IN ('success','failed','running')),
    metadata JSONB,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ
);

-- 2) Users and CV profiles
CREATE TABLE IF NOT EXISTS users (
    user_id BIGSERIAL PRIMARY KEY,
    user_key TEXT UNIQUE,
    email TEXT UNIQUE,
    full_name TEXT,
    phone TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cv_profiles (
    cv_id BIGSERIAL PRIMARY KEY,
    cv_key TEXT NOT NULL UNIQUE,
    user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    file_name TEXT,
    source_path TEXT,
    source_type TEXT NOT NULL DEFAULT 'upload' CHECK (source_type IN ('upload','import','manual')),
    schema_version TEXT NOT NULL DEFAULT 'cv_extracted.v1',
    address TEXT,
    career_objective TEXT,
    seniority_level TEXT,
    location_preference TEXT,
    work_mode_preference TEXT,
    raw_text TEXT,
    parsed_json JSONB,
    target_role TEXT,
    experience_years NUMERIC(5,2),
    education_signals JSONB,
    etl_run_id BIGINT REFERENCES etl_runs(etl_run_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cv_educations (
    cv_education_id BIGSERIAL PRIMARY KEY,
    cv_id BIGINT NOT NULL REFERENCES cv_profiles(cv_id) ON DELETE CASCADE,
    institution_name TEXT,
    degree_name TEXT,
    major_field TEXT,
    passing_year TEXT,
    educational_result TEXT,
    result_type TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cv_experiences (
    cv_experience_id BIGSERIAL PRIMARY KEY,
    cv_id BIGINT NOT NULL REFERENCES cv_profiles(cv_id) ON DELETE CASCADE,
    company_name TEXT,
    position_name TEXT,
    start_date TEXT,
    end_date TEXT,
    location TEXT,
    responsibilities JSONB,
    related_skills JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cv_projects (
    cv_project_id BIGSERIAL PRIMARY KEY,
    cv_id BIGINT NOT NULL REFERENCES cv_profiles(cv_id) ON DELETE CASCADE,
    project_name TEXT,
    project_description TEXT,
    tech_stack JSONB,
    impact_summary TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cv_languages (
    cv_language_id BIGSERIAL PRIMARY KEY,
    cv_id BIGINT NOT NULL REFERENCES cv_profiles(cv_id) ON DELETE CASCADE,
    language_name TEXT NOT NULL,
    proficiency_level TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cv_certifications (
    cv_certification_id BIGSERIAL PRIMARY KEY,
    cv_id BIGINT NOT NULL REFERENCES cv_profiles(cv_id) ON DELETE CASCADE,
    certification_name TEXT,
    provider TEXT,
    issue_date TEXT,
    expiry_date TEXT,
    certification_skills JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cv_links (
    cv_link_id BIGSERIAL PRIMARY KEY,
    cv_id BIGINT NOT NULL REFERENCES cv_profiles(cv_id) ON DELETE CASCADE,
    link_url TEXT NOT NULL,
    link_type TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3) Skills dictionary and links
CREATE TABLE IF NOT EXISTS skills (
    skill_id BIGSERIAL PRIMARY KEY,
    canonical_name TEXT NOT NULL UNIQUE,
    skill_group TEXT,
    aliases JSONB,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cv_skills (
    cv_skill_id BIGSERIAL PRIMARY KEY,
    cv_id BIGINT NOT NULL REFERENCES cv_profiles(cv_id) ON DELETE CASCADE,
    skill_id BIGINT NOT NULL REFERENCES skills(skill_id) ON DELETE RESTRICT,
    source TEXT NOT NULL DEFAULT 'extractor' CHECK (source IN ('extractor','manual','inferred')),
    confidence NUMERIC(5,4),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (cv_id, skill_id, source)
);

-- 4) Jobs and skill mapping
CREATE TABLE IF NOT EXISTS jobs (
    job_id BIGSERIAL PRIMARY KEY,
    external_job_id TEXT UNIQUE,
    job_url TEXT UNIQUE,
    title TEXT NOT NULL,
    title_canonical TEXT,
    company_name TEXT,
    location TEXT,
    work_mode TEXT,
    job_family TEXT,
    salary_min_vnd_month NUMERIC(14,2),
    salary_max_vnd_month NUMERIC(14,2),
    experience_min_years NUMERIC(5,2),
    experience_max_years NUMERIC(5,2),
    employment_type_norm TEXT,
    education_level_norm TEXT,
    job_level_norm TEXT,
    description TEXT,
    requirements TEXT,
    metadata JSONB,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    etl_run_id BIGINT REFERENCES etl_runs(etl_run_id),
    published_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS job_skills (
    job_skill_id BIGSERIAL PRIMARY KEY,
    job_id BIGINT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    skill_id BIGINT NOT NULL REFERENCES skills(skill_id) ON DELETE RESTRICT,
    source_field TEXT,
    importance TEXT CHECK (importance IN ('required','preferred','mentioned')),
    excerpt TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (job_id, skill_id, source_field, importance)
);

-- 5) RAG source documents/chunks/embeddings
CREATE TABLE IF NOT EXISTS rag_documents (
    document_id BIGSERIAL PRIMARY KEY,
    source_type TEXT NOT NULL CHECK (source_type IN ('job','cv','faq','guide','other')),
    source_id TEXT,
    title TEXT,
    body TEXT NOT NULL,
    metadata JSONB,
    etl_run_id BIGINT REFERENCES etl_runs(etl_run_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (source_type, source_id)
);

CREATE TABLE IF NOT EXISTS rag_chunks (
    chunk_id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES rag_documents(document_id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    token_count INT,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (document_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS rag_embeddings (
    embedding_id BIGSERIAL PRIMARY KEY,
    chunk_id BIGINT NOT NULL UNIQUE REFERENCES rag_chunks(chunk_id) ON DELETE CASCADE,
    embedding VECTOR(768) NOT NULL,
    embedding_model TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Optional cache for CV embeddings
CREATE TABLE IF NOT EXISTS cv_embeddings (
    cv_id BIGINT PRIMARY KEY REFERENCES cv_profiles(cv_id) ON DELETE CASCADE,
    embedding VECTOR(768) NOT NULL,
    embedding_model TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 6) Gap analysis and recommendation results
CREATE TABLE IF NOT EXISTS cv_gap_reports (
    gap_report_id BIGSERIAL PRIMARY KEY,
    cv_id BIGINT NOT NULL REFERENCES cv_profiles(cv_id) ON DELETE CASCADE,
    domain_fit TEXT,
    target_role_from_cv TEXT,
    best_fit_roles JSONB,
    strengths JSONB,
    missing_skills JSONB,
    top_role_result JSONB,
    role_ranking JSONB,
    market_gap_json JSONB,
    model_name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS recommendation_runs (
    run_id BIGSERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(user_id) ON DELETE CASCADE,
    cv_id BIGINT REFERENCES cv_profiles(cv_id) ON DELETE SET NULL,
    gap_report_id BIGINT REFERENCES cv_gap_reports(gap_report_id) ON DELETE SET NULL,
    query_text TEXT,
    filters JSONB,
    strategy TEXT DEFAULT 'hybrid_v1',
    model_name TEXT,
    latency_ms INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS recommendation_results (
    result_id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES recommendation_runs(run_id) ON DELETE CASCADE,
    job_id BIGINT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    rank INT,
    score NUMERIC(10,6),
    matched_skills JSONB,
    missing_skills JSONB,
    reasons JSONB,
    evidence_chunk_ids BIGINT[],
    explanation TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, job_id),
    UNIQUE (run_id, rank)
);

-- 7) Chat sessions/history with grounding metadata
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id BIGINT REFERENCES users(user_id) ON DELETE CASCADE,
    cv_id BIGINT REFERENCES cv_profiles(cv_id) ON DELETE SET NULL,
    gap_report_id BIGINT REFERENCES cv_gap_reports(gap_report_id) ON DELETE SET NULL,
    title TEXT,
    model_name TEXT,
    internal_only BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chat_messages (
    message_id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('system','user','assistant')),
    content TEXT NOT NULL,
    retrieved_chunk_ids BIGINT[],
    grounded BOOLEAN,
    latency_ms INTEGER,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

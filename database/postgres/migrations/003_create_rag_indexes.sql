-- Query performance indexes for RAG + recommendation workflow

-- users/cv
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_cv_profiles_user_id ON cv_profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_cv_profiles_target_role ON cv_profiles(target_role);
CREATE INDEX IF NOT EXISTS idx_cv_profiles_updated_at ON cv_profiles(updated_at DESC);

-- skills
CREATE INDEX IF NOT EXISTS idx_skills_group ON skills(skill_group);
CREATE INDEX IF NOT EXISTS idx_cv_skills_cv_id ON cv_skills(cv_id);
CREATE INDEX IF NOT EXISTS idx_cv_skills_skill_id ON cv_skills(skill_id);
CREATE INDEX IF NOT EXISTS idx_job_skills_job_id ON job_skills(job_id);
CREATE INDEX IF NOT EXISTS idx_job_skills_skill_id ON job_skills(skill_id);
CREATE INDEX IF NOT EXISTS idx_job_skills_importance ON job_skills(importance);

-- jobs
CREATE INDEX IF NOT EXISTS idx_jobs_title ON jobs(title);
CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company_name);
CREATE INDEX IF NOT EXISTS idx_jobs_location ON jobs(location);
CREATE INDEX IF NOT EXISTS idx_jobs_job_family_active ON jobs(job_family, is_active);
CREATE INDEX IF NOT EXISTS idx_jobs_work_mode ON jobs(work_mode);
CREATE INDEX IF NOT EXISTS idx_jobs_experience_range ON jobs(experience_min_years, experience_max_years);
CREATE INDEX IF NOT EXISTS idx_jobs_salary_range ON jobs(salary_min_vnd_month, salary_max_vnd_month);
CREATE INDEX IF NOT EXISTS idx_jobs_published_at ON jobs(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_etl_run_id ON jobs(etl_run_id);

-- rag tables
CREATE INDEX IF NOT EXISTS idx_rag_documents_source ON rag_documents(source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_document_id ON rag_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_created_at ON rag_chunks(created_at DESC);

-- JSONB
CREATE INDEX IF NOT EXISTS idx_jobs_metadata_gin ON jobs USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_rag_documents_metadata_gin ON rag_documents USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_metadata_gin ON rag_chunks USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_cv_gap_reports_role_ranking_gin ON cv_gap_reports USING GIN (role_ranking);
CREATE INDEX IF NOT EXISTS idx_recommendation_results_reasons_gin ON recommendation_results USING GIN (reasons);
CREATE INDEX IF NOT EXISTS idx_recommendation_results_matched_skills_gin ON recommendation_results USING GIN (matched_skills);
CREATE INDEX IF NOT EXISTS idx_recommendation_results_missing_skills_gin ON recommendation_results USING GIN (missing_skills);

-- vector indexes (cosine)
CREATE INDEX IF NOT EXISTS idx_rag_embeddings_vector_cosine
    ON rag_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_am WHERE amname = 'hnsw') THEN
        EXECUTE '
            CREATE INDEX IF NOT EXISTS idx_cv_embeddings_vector_cosine_hnsw
            ON cv_embeddings
            USING hnsw (embedding vector_cosine_ops)
        ';
    END IF;
END $$;

-- recommendation/chat
CREATE INDEX IF NOT EXISTS idx_cv_gap_reports_cv_id_created ON cv_gap_reports(cv_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_recommendation_runs_user_id ON recommendation_runs(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_recommendation_runs_cv_id ON recommendation_runs(cv_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_recommendation_results_run_rank ON recommendation_results(run_id, rank);
CREATE INDEX IF NOT EXISTS idx_recommendation_results_run_score ON recommendation_results(run_id, score DESC);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_cv_id ON chat_sessions(cv_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id, created_at);

-- CV scoring outputs and role benchmark cache

CREATE TABLE IF NOT EXISTS cv_scoring_results (
    score_id BIGSERIAL PRIMARY KEY,
    cv_id BIGINT NOT NULL REFERENCES cv_profiles(cv_id) ON DELETE CASCADE,
    total_score NUMERIC(5,2) NOT NULL,
    skill_score NUMERIC(5,2) NOT NULL,
    experience_score NUMERIC(5,2) NOT NULL,
    project_score NUMERIC(5,2) NOT NULL,
    education_score NUMERIC(5,2) NOT NULL,
    completeness_score NUMERIC(5,2) NOT NULL,
    grade TEXT NOT NULL,
    benchmark_role TEXT,
    strengths JSONB NOT NULL DEFAULT '[]'::jsonb,
    missing_skills JSONB NOT NULL DEFAULT '[]'::jsonb,
    priority_skills JSONB NOT NULL DEFAULT '[]'::jsonb,
    development_plan_30_60_90 JSONB NOT NULL DEFAULT '{}'::jsonb,
    subscores_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    model_version TEXT NOT NULL DEFAULT 'cv_scoring_v1',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (cv_id)
);

CREATE TABLE IF NOT EXISTS role_skill_benchmarks (
    role_name TEXT PRIMARY KEY,
    top_market_skills JSONB NOT NULL DEFAULT '[]'::jsonb,
    top_profile_skills JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    model_version TEXT NOT NULL DEFAULT 'cv_scoring_v1',
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cv_scoring_results_total_score ON cv_scoring_results(total_score DESC);
CREATE INDEX IF NOT EXISTS idx_cv_scoring_results_benchmark_role ON cv_scoring_results(benchmark_role);

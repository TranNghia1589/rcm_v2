-- Data contract tables for CV scoring pipeline (final schema)
-- This migration is additive and does not modify existing runtime tables.

CREATE TABLE IF NOT EXISTS cv_raw (
    cv_raw_id BIGSERIAL PRIMARY KEY,
    cv_id TEXT NOT NULL UNIQUE,
    file_name TEXT NOT NULL,
    source_path TEXT NOT NULL,
    file_hash TEXT,
    file_type TEXT NOT NULL CHECK (file_type IN ('pdf', 'docx', 'txt')),
    file_size_bytes BIGINT,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS cv_extracted (
    cv_extracted_id BIGSERIAL PRIMARY KEY,
    cv_id TEXT NOT NULL UNIQUE REFERENCES cv_raw(cv_id) ON DELETE CASCADE,
    schema_version TEXT NOT NULL DEFAULT 'cv_extracted.final',
    extract_status TEXT NOT NULL DEFAULT 'success' CHECK (extract_status IN ('success', 'partial', 'failed')),
    target_role TEXT,
    experience_years_text TEXT,
    skills JSONB NOT NULL DEFAULT '[]'::jsonb,
    projects JSONB NOT NULL DEFAULT '[]'::jsonb,
    parsed_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    parse_errors JSONB NOT NULL DEFAULT '[]'::jsonb,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS cv_labels (
    cv_label_id BIGSERIAL PRIMARY KEY,
    cv_id TEXT NOT NULL REFERENCES cv_raw(cv_id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    final_label_score NUMERIC(5,2) NOT NULL CHECK (final_label_score >= 0 AND final_label_score <= 100),
    grade TEXT NOT NULL CHECK (grade IN ('A', 'B', 'C', 'D', 'E')),
    subscore_skill NUMERIC(5,2) NOT NULL CHECK (subscore_skill >= 0 AND subscore_skill <= 30),
    subscore_experience NUMERIC(5,2) NOT NULL CHECK (subscore_experience >= 0 AND subscore_experience <= 25),
    subscore_achievement NUMERIC(5,2) NOT NULL CHECK (subscore_achievement >= 0 AND subscore_achievement <= 20),
    subscore_education NUMERIC(5,2) NOT NULL CHECK (subscore_education >= 0 AND subscore_education <= 10),
    subscore_formatting NUMERIC(5,2) NOT NULL CHECK (subscore_formatting >= 0 AND subscore_formatting <= 10),
    subscore_keywords NUMERIC(5,2) NOT NULL CHECK (subscore_keywords >= 0 AND subscore_keywords <= 5),
    label_source TEXT NOT NULL CHECK (label_source IN ('human_single', 'human_double', 'committee', 'rubric_bootstrap')),
    labeler_id TEXT NOT NULL,
    label_confidence NUMERIC(4,3) NOT NULL CHECK (label_confidence >= 0 AND label_confidence <= 1),
    labeled_at TIMESTAMPTZ NOT NULL,
    missing_skills JSONB NOT NULL DEFAULT '[]'::jsonb,
    strengths JSONB NOT NULL DEFAULT '[]'::jsonb,
    rationale_text TEXT,
    notes TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (
        ABS(
            COALESCE(subscore_skill, 0)
            + COALESCE(subscore_experience, 0)
            + COALESCE(subscore_achievement, 0)
            + COALESCE(subscore_education, 0)
            + COALESCE(subscore_formatting, 0)
            + COALESCE(subscore_keywords, 0)
            - COALESCE(final_label_score, 0)
        ) <= 1.0
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_cv_labels_cv_id_labeler_labeled_at
ON cv_labels (cv_id, labeler_id, labeled_at);

CREATE INDEX IF NOT EXISTS idx_cv_raw_ingested_at ON cv_raw (ingested_at DESC);
CREATE INDEX IF NOT EXISTS idx_cv_extracted_target_role ON cv_extracted (target_role);
CREATE INDEX IF NOT EXISTS idx_cv_extracted_extracted_at ON cv_extracted (extracted_at DESC);
CREATE INDEX IF NOT EXISTS idx_cv_labels_role ON cv_labels (role);
CREATE INDEX IF NOT EXISTS idx_cv_labels_final_score ON cv_labels (final_label_score DESC);

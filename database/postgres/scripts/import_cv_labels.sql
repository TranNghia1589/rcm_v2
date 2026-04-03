-- Import validated CV labels CSV into cv_labels.
-- Required psql variables:
--   labels_csv  : absolute path to CSV file
--   source_path : metadata source path for auto-created cv_raw rows

BEGIN;

CREATE OR REPLACE FUNCTION _safe_jsonb_list(raw_text TEXT)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    t TEXT;
    j JSONB;
BEGIN
    t := COALESCE(BTRIM(raw_text), '');
    IF t = '' THEN
        RETURN '[]'::jsonb;
    END IF;

    BEGIN
        j := t::jsonb;
        IF jsonb_typeof(j) = 'array' THEN
            RETURN j;
        END IF;
        RETURN jsonb_build_array(j);
    EXCEPTION WHEN others THEN
        RETURN jsonb_build_array(t);
    END;
END;
$$;

CREATE TEMP TABLE stage_cv_labels (
    cv_id TEXT,
    role TEXT,
    final_label_score TEXT,
    grade TEXT,
    subscore_skill TEXT,
    subscore_experience TEXT,
    subscore_achievement TEXT,
    subscore_education TEXT,
    subscore_formatting TEXT,
    subscore_keywords TEXT,
    label_source TEXT,
    labeler_id TEXT,
    label_confidence TEXT,
    labeled_at TEXT,
    missing_skills TEXT,
    strengths TEXT,
    rationale_text TEXT,
    notes TEXT
);

\copy stage_cv_labels FROM '__LABELS_CSV__' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')

-- Ensure cv_raw contains referenced cv_id for FK integrity.
INSERT INTO cv_raw (cv_id, file_name, source_path, file_type, metadata)
SELECT DISTINCT
    s.cv_id,
    s.cv_id || '.pdf',
    '__SOURCE_PATH__',
    'pdf',
    '{}'::jsonb
FROM stage_cv_labels s
WHERE COALESCE(TRIM(s.cv_id), '') <> ''
  AND NOT EXISTS (
      SELECT 1 FROM cv_raw r WHERE r.cv_id = s.cv_id
  );

INSERT INTO cv_labels (
    cv_id, role, final_label_score, grade,
    subscore_skill, subscore_experience, subscore_achievement,
    subscore_education, subscore_formatting, subscore_keywords,
    label_source, labeler_id, label_confidence, labeled_at,
    missing_skills, strengths, rationale_text, notes
)
SELECT
    TRIM(s.cv_id),
    TRIM(s.role),
    TRIM(s.final_label_score)::NUMERIC,
    UPPER(TRIM(s.grade)),
    TRIM(s.subscore_skill)::NUMERIC,
    TRIM(s.subscore_experience)::NUMERIC,
    TRIM(s.subscore_achievement)::NUMERIC,
    TRIM(s.subscore_education)::NUMERIC,
    TRIM(s.subscore_formatting)::NUMERIC,
    TRIM(s.subscore_keywords)::NUMERIC,
    TRIM(s.label_source),
    TRIM(s.labeler_id),
    TRIM(s.label_confidence)::NUMERIC,
    COALESCE(NULLIF(TRIM(s.labeled_at), ''), NOW()::TEXT)::TIMESTAMPTZ,
    _safe_jsonb_list(s.missing_skills),
    _safe_jsonb_list(s.strengths),
    NULLIF(TRIM(s.rationale_text), ''),
    NULLIF(TRIM(s.notes), '')
FROM stage_cv_labels s
WHERE COALESCE(TRIM(s.cv_id), '') <> ''
ON CONFLICT (cv_id, labeler_id, labeled_at)
DO UPDATE SET
    role = EXCLUDED.role,
    final_label_score = EXCLUDED.final_label_score,
    grade = EXCLUDED.grade,
    subscore_skill = EXCLUDED.subscore_skill,
    subscore_experience = EXCLUDED.subscore_experience,
    subscore_achievement = EXCLUDED.subscore_achievement,
    subscore_education = EXCLUDED.subscore_education,
    subscore_formatting = EXCLUDED.subscore_formatting,
    subscore_keywords = EXCLUDED.subscore_keywords,
    label_source = EXCLUDED.label_source,
    label_confidence = EXCLUDED.label_confidence,
    missing_skills = EXCLUDED.missing_skills,
    strengths = EXCLUDED.strengths,
    rationale_text = EXCLUDED.rationale_text,
    notes = EXCLUDED.notes;

DROP FUNCTION IF EXISTS _safe_jsonb_list(TEXT);

COMMIT;

// Params: $cv_id (int), $limit (int)
MATCH (cv:CV {cv_id: $cv_id})
OPTIONAL MATCH (cv)-[:HAS_SKILL]->(owned:Skill)
WITH cv, collect(DISTINCT toLower(owned.canonical_name)) AS owned_skill_names
OPTIONAL MATCH (cv)-[:HAS_CERTIFICATION]->(:Certification)-[:CERTIFIES_SKILL]->(certSkill:Skill)
WITH cv, owned_skill_names, collect(DISTINCT toLower(certSkill.canonical_name)) AS cert_skill_names
OPTIONAL MATCH (cv)-[:BEST_FIT_ROLE|TARGETS_ROLE]->(r:Role)
WITH cv, owned_skill_names, cert_skill_names, collect(DISTINCT toLower(r.name)) AS role_names
MATCH (j:Job)
OPTIONAL MATCH (j)-[req:REQUIRES_SKILL]->(need:Skill)
WITH j, role_names, cv, owned_skill_names, cert_skill_names, req, need,
     CASE WHEN toLower(need.canonical_name) IN owned_skill_names THEN 1 ELSE 0 END AS is_match,
     CASE WHEN toLower(need.canonical_name) IN cert_skill_names THEN 1 ELSE 0 END AS cert_match
WITH j, role_names, cv,
     sum(is_match) AS matched_skills,
     count(need) AS total_required,
     sum(
         CASE
             WHEN is_match = 1 THEN
                 CASE req.importance WHEN 'required' THEN 2.0 WHEN 'preferred' THEN 1.0 ELSE 0.5 END
             ELSE 0.0
         END
     ) AS weighted_match,
     sum(cert_match) AS cert_match_count
WITH j, role_names, cv, matched_skills, total_required, weighted_match, cert_match_count,
     CASE WHEN total_required = 0 THEN 0.0 ELSE toFloat(matched_skills)/toFloat(total_required) END AS coverage,
     CASE
         WHEN any(rn IN role_names WHERE rn <> '' AND (toLower(j.job_family) CONTAINS rn OR toLower(j.title) CONTAINS rn))
         THEN 0.2
         ELSE 0.0
     END AS role_bonus,
     CASE
         WHEN cv.experience_years IS NULL THEN 0.0
         WHEN j.experience_min_years IS NULL THEN 0.05
         WHEN toFloat(cv.experience_years) >= toFloat(j.experience_min_years) THEN 0.1
         ELSE 0.0
     END AS exp_bonus
WITH j, matched_skills, total_required, coverage, weighted_match, role_bonus, exp_bonus, cert_match_count,
     CASE WHEN matched_skills > 0 THEN 1 ELSE 0 END AS has_match
RETURN
  j.job_id AS job_id,
  j.title AS title,
  j.company_name AS company_name,
  j.location AS location,
  matched_skills,
  total_required,
  coverage,
  (weighted_match + coverage * 2.0 + role_bonus + exp_bonus + toFloat(cert_match_count) * 0.05) AS score
ORDER BY has_match DESC, score DESC, matched_skills DESC, coverage DESC, cert_match_count DESC
LIMIT $limit;

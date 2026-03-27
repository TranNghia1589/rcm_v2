// Params: $cv_id (int), $limit (int)
MATCH (cv:CV {cv_id: $cv_id})
OPTIONAL MATCH (cv)-[:HAS_SKILL]->(owned:Skill)
WITH cv, collect(DISTINCT toLower(owned.canonical_name)) AS owned_skill_names
OPTIONAL MATCH (cv)-[:BEST_FIT_ROLE|TARGETS_ROLE]->(r:Role)
WITH cv, owned_skill_names, collect(DISTINCT toLower(r.name)) AS role_names
MATCH (j:Job)
OPTIONAL MATCH (j)-[req:REQUIRES_SKILL]->(need:Skill)
WITH j, role_names, owned_skill_names, req, need,
     CASE WHEN toLower(need.canonical_name) IN owned_skill_names THEN 1 ELSE 0 END AS is_match
WITH j, role_names,
     sum(is_match) AS matched_skills,
     count(need) AS total_required,
     sum(
         CASE
             WHEN is_match = 1 THEN
                 CASE req.importance WHEN 'required' THEN 2.0 WHEN 'preferred' THEN 1.0 ELSE 0.5 END
             ELSE 0.0
         END
     ) AS weighted_match
WITH j, role_names, matched_skills, total_required, weighted_match,
     CASE WHEN total_required = 0 THEN 0.0 ELSE toFloat(matched_skills)/toFloat(total_required) END AS coverage,
     CASE
         WHEN any(rn IN role_names WHERE rn <> '' AND (toLower(j.job_family) CONTAINS rn OR toLower(j.title) CONTAINS rn))
         THEN 0.2
         ELSE 0.0
     END AS role_bonus
WITH j, matched_skills, total_required, coverage, weighted_match, role_bonus,
     CASE WHEN matched_skills > 0 THEN 1 ELSE 0 END AS has_match
RETURN
  j.job_id AS job_id,
  j.title AS title,
  j.company_name AS company_name,
  j.location AS location,
  matched_skills,
  total_required,
  coverage,
  (weighted_match + coverage * 2.0 + role_bonus) AS score
ORDER BY has_match DESC, score DESC, matched_skills DESC, coverage DESC
LIMIT $limit;

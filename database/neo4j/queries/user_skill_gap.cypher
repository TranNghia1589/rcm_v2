// Params: $cv_id (int), $limit (int)
MATCH (cv:CV {cv_id: $cv_id})-[:BEST_FIT_ROLE]->(r:Role)
OPTIONAL MATCH (cv)-[:HAS_SKILL]->(owned:Skill)
WITH cv, r, collect(DISTINCT toLower(owned.canonical_name)) AS owned_skills
MATCH (j:Job)-[:REQUIRES_SKILL]->(need:Skill)
WHERE j.job_family = r.name OR j.title CONTAINS r.name
WITH owned_skills, toLower(need.canonical_name) AS need_skill, count(*) AS freq
WHERE NOT need_skill IN owned_skills
RETURN need_skill AS missing_skill, freq
ORDER BY freq DESC
LIMIT $limit;

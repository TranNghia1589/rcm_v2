// Params: $cv_id (int), $limit (int)
MATCH (cv:CV {cv_id: $cv_id})-[:TARGETS_ROLE|BEST_FIT_ROLE]->(r:Role)
OPTIONAL MATCH (j:Job)-[:REQUIRES_SKILL]->(s:Skill)
WHERE j.job_family = r.name OR j.title CONTAINS r.name
RETURN
  r.name AS role_name,
  collect(DISTINCT s.canonical_name)[0..$limit] AS priority_skills,
  count(DISTINCT j) AS supporting_jobs
ORDER BY supporting_jobs DESC;

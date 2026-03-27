# Recommendation Labeling Rubric (Qrels)

Use this rubric to assign `relevance` in `qrels.csv` for each pair `(cv_id, job_id)`.

## Relevance scale

- `0` = Not relevant
  - Role/title mismatch, or clear skill/seniority mismatch.
- `1` = Weakly relevant
  - Some overlap exists, but multiple critical gaps.
- `2` = Relevant
  - Good role and skill fit, minor gaps acceptable.
- `3` = Highly relevant
  - Strong role fit + strong core-skill match + suitable seniority/location.

## Labeling criteria (in order)

1. Role/title fit
2. Core skill overlap (required skills first)
3. Seniority / years of experience fit
4. Location/work mode fit (if this is part of your use-case)

## Anti-bias rules

- Do not use model score while labeling.
- Label each pair based on CV text + job content only.
- Keep criteria fixed across all CVs.

## Quality control

- Double-check at least 20% rows on a second pass.
- For ambiguous cases, add notes in `notes` column.

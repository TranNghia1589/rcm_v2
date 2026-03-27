# Recommendation Evaluation Guide

This project keeps evaluation code separate from runtime pipeline code:

- Runtime recommendation logic: `src/recommendation/`
- Evaluation metrics/scripts: `src/evaluation/recommendation/`

That structure is common in production-like ML systems because:

- evaluation can evolve independently of serving logic
- avoids coupling experiment code with API runtime code
- easier to report and reproduce benchmark results

## 1) Input file formats

### Qrels (ground truth)

Required columns:

- `cv_id`
- `job_id`
- `relevance` (0, 1, 2, 3...)

Example (`csv`):

```csv
cv_id,job_id,relevance
2,162,3
2,163,2
2,176,2
3,44,1
```

### Predictions

Required columns:

- `cv_id`
- `job_id`
- and one of:
  - `rank` (ascending = better), or
  - `score` (descending = better)

Optional:

- `method` (e.g. `vector`, `graph`, `hybrid`) to compare multiple systems in one file.
- `job_family` (for coverage/diversity evaluation).

## 2) Ranking metrics

Before ranking metrics, build prediction file and qrels template:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/pipelines/run_recommendation_eval.ps1 -CvIds "1,2" -TopK 20
```

Then label `artifacts/evaluation/qrels_template.csv` and save it as `artifacts/evaluation/qrels.csv`.

Run:

```powershell
python -m src.evaluation.recommendation.eval_ranking `
  --predictions artifacts\evaluation\predictions.csv `
  --qrels artifacts\evaluation\qrels.csv `
  --k_list 5,10 `
  --output artifacts\evaluation\recommendation_ranking_summary.csv
```

Metrics output:

- `precision@K`
- `recall@K`
- `hitrate@K`
- `mrr@K`
- `ndcg@K`

## 3) Coverage/diversity metrics

Run:

```powershell
python -m src.evaluation.recommendation.eval_coverage `
  --predictions artifacts\evaluation\predictions.csv `
  --jobs_catalog artifacts\matching\jobs_matching_ready_v3.parquet `
  --k_list 5,10 `
  --output artifacts\evaluation\recommendation_coverage_summary.csv
```

Metrics output:

- `catalog_coverage@K`
- `intra_list_diversity@K` (proxy by `job_family`)

## 4) Recommended professional folder layout

```text
src/
  recommendation/        # serving/ranking logic used by API
  evaluation/
    recommendation/      # offline metrics + benchmark scripts
artifacts/
  evaluation/            # qrels, predictions, summary outputs
docs/runbooks/
  RECOMMENDATION_EVAL_GUIDE.md
```

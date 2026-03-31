param(
    [string]$CvIds = "1,2",
    [int]$TopK = 20,
    [string]$Methods = "vector,graph,hybrid"
)

$ErrorActionPreference = "Stop"
$py = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

Write-Host "=== Step 1: Build predictions (vector, graph, hybrid) ==="
& $py -m src.evaluation.recommendation.build_predictions `
  --cv_ids $CvIds `
  --methods $Methods `
  --top_k $TopK `
  --output "artifacts/evaluation/predictions.csv"
if ($LASTEXITCODE -ne 0) {
  throw "Step 1 failed. Check service dependencies (Neo4j for graph/hybrid, Postgres for vector/hybrid)."
}

Write-Host "=== Step 2: Build qrels template for labeling ==="
& $py -m src.evaluation.recommendation.build_qrels_template `
  --predictions "artifacts/evaluation/predictions.csv" `
  --output "artifacts/evaluation/qrels_template.csv"
if ($LASTEXITCODE -ne 0) {
  throw "Step 2 failed. predictions.csv was not generated or invalid."
}

Write-Host "Done. Next:"
Write-Host "1) Fill artifacts/evaluation/qrels_template.csv (column relevance in {0,1,2,3})"
Write-Host "2) Save labeled file as artifacts/evaluation/qrels.csv"
Write-Host "3) Run ranking eval:"
Write-Host "   python -m src.evaluation.recommendation.eval_ranking --predictions artifacts/evaluation/predictions.csv --qrels artifacts/evaluation/qrels.csv --k_list 5,10 --output artifacts/evaluation/recommendation_ranking_summary.csv"
Write-Host "4) Run coverage eval:"
Write-Host "   python -m src.evaluation.recommendation.eval_coverage --predictions artifacts/evaluation/predictions.csv --jobs_catalog artifacts/matching/jobs_matching_ready_v3.parquet --k_list 5,10 --output artifacts/evaluation/recommendation_coverage_summary.csv"

param(
  [switch]$SkipPreprocessEmbedding = $true
)

$py = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

$cmd = @("deploy\\scripts\\run_local_pipeline.py", "--stages", "preprocess_jobs", "extract_cv", "cv_gap", "load_core_tables", "cv_scoring", "rag_ingest", "graph_etl")
if ($SkipPreprocessEmbedding) {
  $cmd += "--skip-preprocess-embedding"
}

& $py @cmd
if ($LASTEXITCODE -ne 0) {
  throw "Pipeline execution failed."
}

Write-Host "[DONE] Full bootstrap pipeline completed."


param(
  [string]$DbHost = "localhost",
  [int]$Port = 5432,
  [string]$Database = "job_recommendation_system",
  [string]$User = "postgres",
  [string]$Password = "123456"
)

$ErrorActionPreference = "Stop"
$env:PGPASSWORD = $Password

Write-Host "[INFO] Init PostgreSQL + pgvector schema on ${DbHost}:$Port/$Database"

$newRoot = "apps/api/migrations"
$oldRoot = "database/postgres/migrations"

if (Test-Path $newRoot) {
  $root = $newRoot
} elseif (Test-Path $oldRoot) {
  $root = $oldRoot
} else {
  throw "Migration root not found. Expected '$newRoot' or '$oldRoot'."
}

$files = @(
  "$root/001_enable_pgvector.sql",
  "$root/002_create_rag_tables.sql",
  "$root/003_create_rag_indexes.sql",
  "$root/004_create_cv_scoring_tables.sql"
)

foreach ($f in $files) {
  if (!(Test-Path $f)) {
    throw "Migration file not found: $f"
  }
  Write-Host "[RUN] $f"
  psql -h $DbHost -p $Port -U $User -d $Database -v ON_ERROR_STOP=1 -f $f
  if ($LASTEXITCODE -ne 0) {
    throw "Migration failed ($LASTEXITCODE): $f"
  }
}

Write-Host "[DONE] PostgreSQL schema initialized."

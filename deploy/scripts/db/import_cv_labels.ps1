param(
  [string]$DbHost = "localhost",
  [int]$Port = 5432,
  [string]$Database = "job_recommendation_system",
  [string]$User = "postgres",
  [string]$Password = "123456",
  [string]$LabelsCsv = "data/reference/review/cv_labels_pilot_100.csv",
  [string]$SourcePath = "manual://cv_labels_import"
)

$ErrorActionPreference = "Stop"
$env:PGPASSWORD = $Password

if (!(Test-Path $LabelsCsv)) {
  throw "Labels CSV not found: $LabelsCsv"
}

$sqlFile = "database/postgres/scripts/import_cv_labels.sql"
if (!(Test-Path $sqlFile)) {
  throw "SQL script not found: $sqlFile"
}

$labelsAbs = (Resolve-Path $LabelsCsv).Path -replace "\\", "/"
$labelsAbsEsc = $labelsAbs -replace "'", "''"
$sourcePathEsc = $SourcePath -replace "'", "''"

Write-Host "[INFO] Importing CV labels from: $labelsAbs"
Write-Host "[INFO] Target DB: ${DbHost}:$Port/$Database"

$sqlContent = Get-Content $sqlFile -Raw
$sqlContent = $sqlContent.Replace("__LABELS_CSV__", $labelsAbsEsc)
$sqlContent = $sqlContent.Replace("__SOURCE_PATH__", $sourcePathEsc)

$sqlContent | psql `
  -h $DbHost `
  -p $Port `
  -U $User `
  -d $Database `
  -v ON_ERROR_STOP=1

if ($LASTEXITCODE -ne 0) {
  throw "Import failed (exit code: $LASTEXITCODE)"
}

Write-Host "[DONE] CV labels imported successfully."

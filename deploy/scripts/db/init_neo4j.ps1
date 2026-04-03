param(
  [string]$Neo4jUser = "neo4j",
  [string]$Neo4jPassword = "abcd@1234",
  [string]$Neo4jDatabase = "neo4j",
  [string]$CypherShellPath = "C:\Program Files\Neo4j Desktop 2\resources\offline\dbmss\neo4j-enterprise-2026.01.4\bin\cypher-shell.bat"
)

$ErrorActionPreference = "Stop"

Write-Host "[INFO] Init Neo4j schema on database '$Neo4jDatabase'"

if ([string]::IsNullOrWhiteSpace($CypherShellPath)) {
  $cmd = Get-Command cypher-shell -ErrorAction SilentlyContinue
  if ($null -ne $cmd) {
    $CypherShellPath = $cmd.Source
  }
}

if ([string]::IsNullOrWhiteSpace($CypherShellPath)) {
  throw "cypher-shell not found. Install Neo4j tools or pass -CypherShellPath 'C:\path\to\cypher-shell.bat'"
}

$root = "database/neo4j/schema"
if (!(Test-Path $root)) {
  throw "Neo4j schema root not found: '$root'"
}

$constraints = Join-Path $root "constraints.cypher"
$indexes = Join-Path $root "indexes.cypher"

if (!(Test-Path $constraints)) {
  throw "Missing file: $constraints"
}
if (!(Test-Path $indexes)) {
  throw "Missing file: $indexes"
}

& $CypherShellPath -u $Neo4jUser -p $Neo4jPassword -d $Neo4jDatabase -f $constraints
if ($LASTEXITCODE -ne 0) {
  throw "Failed to apply constraints.cypher (exit code: $LASTEXITCODE)"
}

& $CypherShellPath -u $Neo4jUser -p $Neo4jPassword -d $Neo4jDatabase -f $indexes
if ($LASTEXITCODE -ne 0) {
  throw "Failed to apply indexes.cypher (exit code: $LASTEXITCODE)"
}

Write-Host "[DONE] Neo4j schema initialized."


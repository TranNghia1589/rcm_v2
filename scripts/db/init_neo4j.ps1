param(
  [string]$Neo4jUser = "neo4j",
  [string]$Neo4jPassword = "your_password",
  [string]$Neo4jDatabase = "neo4j",
  [string]$CypherShellPath = ""
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

$newRoot = "services/graph/cypher"
$oldRoot = "database/neo4j/schema"

if (Test-Path $newRoot) {
  $root = $newRoot
} elseif (Test-Path $oldRoot) {
  $root = $oldRoot
} else {
  throw "Neo4j schema root not found. Expected '$newRoot' or '$oldRoot'."
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

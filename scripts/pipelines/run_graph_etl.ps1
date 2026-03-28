$py = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

& $py src\ingestion\jobs_to_neo4j.py `
  --postgres_config configs\db\postgres.yaml `
  --neo4j_config configs\db\neo4j.yaml `
  --reset_graph

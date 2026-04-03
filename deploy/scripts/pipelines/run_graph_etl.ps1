$py = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

& $py src\data_processing\ingestion\jobs_to_neo4j.py `
  --postgres_config config\db\postgres.yaml `
  --neo4j_config config\db\neo4j.yaml `
  --reset_graph

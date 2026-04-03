$py = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

& $py src\data_processing\ingestion\jobs_to_pgvector.py `
  --jobs_path experiments\artifacts\matching\jobs_chatbot_sections.parquet `
  --postgres_config config\db\postgres.yaml `
  --pgvector_config config\db\pgvector.yaml `
  --chunking_config config\rag\chunking.yaml `
  --embedding_config config\model\embedding.yaml


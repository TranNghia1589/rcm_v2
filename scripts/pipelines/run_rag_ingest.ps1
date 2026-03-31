$py = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

& $py src\ingestion\jobs_to_pgvector.py `
  --jobs_path artifacts\matching\jobs_chatbot_sections_v3.parquet `
  --postgres_config configs\db\postgres.yaml `
  --pgvector_config configs\db\pgvector.yaml `
  --chunking_config configs\rag\chunking.yaml `
  --embedding_config configs\model\embedding.yaml

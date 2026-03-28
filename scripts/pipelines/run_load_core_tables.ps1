$py = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

& $py src\ingestion\load_core_tables.py `
  --postgres_config configs\db\postgres.yaml `
  --jobs_parquet artifacts\matching\jobs_matching_ready_v3.parquet `
  --job_skill_map artifacts\matching\job_skill_map_v3.parquet `
  --cv_dataset data\processed\cv_extracted\cv_extracted_dataset.parquet `
  --gap_dir data\processed\cv_gap_reports

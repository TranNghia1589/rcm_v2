$py = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

& $py src\data_processing\ingestion\load_core_tables.py `
  --postgres_config config\db\postgres.yaml `
  --jobs_parquet experiments\artifacts\matching\jobs_matching_ready.parquet `
  --job_skill_map experiments\artifacts\matching\job_skill_map.parquet `
  --cv_dataset data\processed\cv_extracted\cv_extracted_dataset.parquet `
  --gap_dir data\processed\cv_gap_reports


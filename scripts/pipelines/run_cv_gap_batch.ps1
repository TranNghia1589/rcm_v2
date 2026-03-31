$py = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

& $py src\cv\run_gap_batch.py `
  --cv_dataset data\processed\cv_extracted\cv_extracted_dataset.parquet `
  --role_profiles data\reference\role_profiles.json `
  --output_dir data\processed\cv_gap_reports `
  --aggregate_jsonl data\processed\cv_gap_reports\cv_gap_dataset.jsonl `
  --aggregate_parquet data\processed\cv_gap_reports\cv_gap_dataset.parquet

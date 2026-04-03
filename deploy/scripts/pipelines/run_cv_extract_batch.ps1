$py = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

& $py src\cv\extract_cv_batch.py `
  --input_dir data\raw\cv_samples\INFORMATION-TECHNOLOGY `
  --output_dir data\processed\cv_extracted `
  --aggregate_jsonl data\processed\cv_extracted\cv_extracted_dataset.jsonl `
  --aggregate_parquet data\processed\cv_extracted\cv_extracted_dataset.parquet

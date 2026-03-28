$py = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

& $py src\scoring\run_cv_scoring_batch.py `
  --postgres_config configs\db\postgres.yaml `
  --role_profiles data\role_profiles\role_profiles.json `
  --model_version cv_scoring_v1

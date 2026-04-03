$py = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

& $py src\models\scoring\run_cv_scoring_batch.py `
  --postgres_config config\db\postgres.yaml `
  --role_profiles data\reference\final\role_profiles.json `
  --model_version cv_scoring_v1

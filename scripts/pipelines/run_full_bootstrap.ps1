$py = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

& $py src\pipelines\run_preprocess.py

& $py src\cv\extract_cv_batch.py `
  --input_dir data\raw\cv_samples\INFORMATION-TECHNOLOGY `
  --output_dir data\processed\cv_extracted `
  --aggregate_jsonl data\processed\cv_extracted\cv_extracted_dataset.jsonl `
  --aggregate_parquet data\processed\cv_extracted\cv_extracted_dataset.parquet

& $py src\cv\run_gap_batch.py `
  --cv_dataset data\processed\cv_extracted\cv_extracted_dataset.parquet `
  --role_profiles data\role_profiles\role_profiles.json `
  --output_dir data\processed\cv_gap_reports `
  --aggregate_jsonl data\processed\cv_gap_reports\cv_gap_dataset.jsonl `
  --aggregate_parquet data\processed\cv_gap_reports\cv_gap_dataset.parquet

& $py src\ingestion\load_core_tables.py `
  --postgres_config configs\db\postgres.yaml `
  --jobs_parquet artifacts\matching\jobs_matching_ready_v3.parquet `
  --job_skill_map artifacts\matching\job_skill_map_v3.parquet `
  --cv_dataset data\processed\cv_extracted\cv_extracted_dataset.parquet `
  --gap_dir data\processed\cv_gap_reports

& $py src\scoring\run_cv_scoring_batch.py `
  --postgres_config configs\db\postgres.yaml `
  --role_profiles data\role_profiles\role_profiles.json `
  --model_version cv_scoring_v1

& $py src\ingestion\jobs_to_pgvector.py `
  --jobs_path artifacts\matching\jobs_chatbot_sections_v3.parquet `
  --postgres_config configs\db\postgres.yaml `
  --pgvector_config configs\db\pgvector.yaml `
  --chunking_config configs\rag\chunking.yaml `
  --embedding_config configs\model\embedding.yaml

& $py src\ingestion\jobs_to_neo4j.py `
  --postgres_config configs\db\postgres.yaml `
  --neo4j_config configs\db\neo4j.yaml `
  --reset_graph

python src\ingestion\load_core_tables.py `
  --postgres_config configs\db\postgres.yaml `
  --jobs_parquet artifacts\matching\jobs_matching_ready_v3.parquet `
  --job_skill_map artifacts\matching\job_skill_map_v3.parquet `
  --resume_json data\processed\resume_extracted.json `
  --gap_json data\processed\gap_analysis_result.json

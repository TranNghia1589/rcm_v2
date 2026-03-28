# Project V3 - Job Recommendation + CV Chatbot

Production preprocessing source is now unified at:
- `src/pipelines/run_preprocess.py`

Notebook is archival/reference only:
- `notebooks/archive/legacy_notebooks/preprocessing_ver_phobert_copy_2.ipynb`

## Main Components

- `src/crawl/`: Job crawling and raw normalization.
- `src/pipelines/run_preprocess.py`: Main preprocessing + artifact build pipeline.
- `src/cv/`: CV extraction.
- `src/matching/`: Recommendation and gap analysis engines.
- `apps/api/`: Active API layer (`apps/api/app/server.py`).
- `legacy/deprecated_2026_03_27/`: Deprecated modules/scripts kept for reference only.

## Data & Artifact Layout

- `data/raw/jobs/`: Fresh crawl output (`topcv_all_fields_merged_*.csv|xlsx`).
- `data/raw/cv_samples/`: CV sample files.
- `data/processed/`: Processed intermediate outputs (`jobs_nlp_ready_*`, extracted CV, gap JSON).
- `data/reference/`: Static reference dictionaries/cases.
- `artifacts/matching/`: Final model-ready artifacts for recommender/chatbot.

## Standard Pipeline

1. Crawl:
- `python src/crawl/topcv_crawler.py`

2. Preprocess + build artifacts:
- `python src/pipelines/run_preprocess.py`

3. Extract CV:
- `python src/cv/extract_cv_info.py --cv_path <cv_file> --output_path data/processed/resume_extracted.json`

4. Extract batch CV:
- `python src/cv/extract_cv_batch.py --input_dir data/raw/cv_samples/INFORMATION-TECHNOLOGY --output_dir data/processed/cv_extracted --aggregate_jsonl data/processed/cv_extracted/cv_extracted_dataset.jsonl --aggregate_parquet data/processed/cv_extracted/cv_extracted_dataset.parquet`

5. Gap analysis batch:
- `python src/cv/run_gap_batch.py --cv_dataset data/processed/cv_extracted/cv_extracted_dataset.parquet --output_dir data/processed/cv_gap_reports --aggregate_jsonl data/processed/cv_gap_reports/cv_gap_dataset.jsonl --aggregate_parquet data/processed/cv_gap_reports/cv_gap_dataset.parquet`

6. Load PostgreSQL core tables:
- `python src/ingestion/load_core_tables.py --jobs_parquet artifacts/matching/jobs_matching_ready_v3.parquet --job_skill_map artifacts/matching/job_skill_map_v3.parquet --cv_dataset data/processed/cv_extracted/cv_extracted_dataset.parquet --gap_dir data/processed/cv_gap_reports`

7. CV scoring batch:
- `python src/scoring/run_cv_scoring_batch.py --postgres_config configs/db/postgres.yaml --role_profiles data/role_profiles/role_profiles.json --model_version cv_scoring_v1`

8. Sync RAG + Graph:
- `python src/ingestion/jobs_to_pgvector.py`
- `python src/ingestion/jobs_to_neo4j.py --reset_graph`

9. Chatbot API:
- `uvicorn apps.api.app.server:app --host 0.0.0.0 --port 8000 --reload`
- `POST /api/v1/chat/ask`
- `GET /api/v1/cv/score/{cv_id}`
- `POST /api/v1/recommend/hybrid`

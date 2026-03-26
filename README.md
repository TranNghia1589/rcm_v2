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
- `src/chatbot/`: Internal-only chatbot/retrieval logic.
- `scripts/dev/chatbot_1to1.py`: Interactive 1:1 chatbot CLI.
- `apps/api/`: API layer (incremental implementation).

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

4. Gap analysis:
- `python src/matching/gap_analysis.py --cv_json data/processed/resume_extracted.json --output_path data/processed/gap_analysis_result.json`

5. Chatbot (internal-only):
- `python src/chatbot/retrieval.py --question "CV cua toi phu hop job nao?" --gap_result data/processed/gap_analysis_result.json --cv_json data/processed/resume_extracted.json --top_k_jobs 10`

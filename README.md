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

## Update Today

- PostgreSQL + Neo4j local database flow is active for the project.
- FastAPI now supports CV upload into database tables:
  - `users`
  - `cv_profiles`
  - `cv_skills`
  - `cv_gap_reports`
- Chatbot now supports context-aware chat using:
  - `cv_id`
  - latest gap report from database
  - `session_id` history
- Chat history is persisted into:
  - `chat_sessions`
  - `chat_messages`
- Chatbot answer mode is tightened around processed CV context (`context_locked`) so answers stay closer to extracted CV, gap signals, and recent chat history.
- Current backend generation model for chatbot API is `llama3.1:8b` via local Ollama.
- Metric notebook for the current chatbot pipeline is available at:
  - `legacy/src/chatbot/metric_test/metric_test.ipynb`

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

## API Flow

1. Run API:
- `powershell -ExecutionPolicy Bypass -File scripts/dev/run_api.ps1`

2. Upload CV to database:
- `POST /api/v1/cv/upload`

3. Get CV detail from database:
- `GET /api/v1/cv/{cv_id}`

4. Chat with CV context:
- `POST /api/v1/chat/ask`
- minimal payload:
  - `question`
  - `cv_id`

5. Continue same conversation:
- resend returned `session_id`

## Example API Usage

Upload CV:

```powershell
curl.exe -X POST "http://127.0.0.1:8010/api/v1/cv/upload" `
  -F "file=@C:\path\to\your_cv.txt" `
  -F "full_name=Nguyen Van A"
```

Chat with uploaded CV:

```powershell
$body = @{
  question = "CV này hợp role nào nhất và còn thiếu gì để apply tốt hơn?"
  top_k = 3
  cv_id = 4
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8010/api/v1/chat/ask" -ContentType "application/json" -Body $body
```

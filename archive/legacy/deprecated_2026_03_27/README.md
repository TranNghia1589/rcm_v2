# Deprecated Modules (2026-03-27)

These files were moved out of the active pipeline to reduce overlap and confusion.

## Why deprecated
- The active job preprocessing pipeline is `src/data_processing/pipelines/run_preprocess.py`.
- The active chatbot runtime is API-based (`apps/api/app/server.py` + `apps/api/app/services/rag/*`).
- Older script-based/legacy modules are kept only for reference and rollback.

## Moved from active paths
- `src/preprocessing/clean_jobs.py`
- `src/preprocessing/build_training_tables.py`
- `src/chatbot/advisor.py`
- `src/chatbot/prompts.py`
- `src/chatbot/retrieval.py`
- `deploy/scripts/dev/chatbot_1to1.py`
- `deploy/scripts/dev/test_chatbot_interactive.py`
- `apps/api/app/main.py`


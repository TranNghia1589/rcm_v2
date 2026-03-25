# Project V3 - Job Recommendation + CV Chatbot (FastAPI-ready)

This folder is a reorganized, production-oriented structure for the thesis project.
It preserves data/code from `project_v2` while preparing clear modules for a future FastAPI website.

## Main Components

- `apps/api/`: FastAPI application skeleton (placeholder files created).
- `src/crawl/`: Crawling and raw job normalization.
- `src/preprocessing/`: Preprocessing and dataset-building logic.
- `src/matching/`: Job matching and gap analysis logic.
- `src/cv/`: CV extraction and CV scoring related logic.
- `src/chatbot/`: Chatbot advisor and retrieval logic.
- `src/pipelines/`: Runnable pipeline entry points.

## Data Layout

- `data/raw/`: Original crawled/source files.
- `data/interim/`: Temporary processing outputs.
- `data/processed/`: Final cleaned datasets for services.
- `data/reference/`: Dictionaries, catalogs, and evaluation cases.
- `artifacts/`: Embeddings, indices, manifests, and model-ready outputs.

## Notebook Policy

Only the final test notebook is kept under preprocessing experiments:
- `notebooks/experiments/preprocessing_ver_phobert copy 2.ipynb`

Legacy notebook outputs are preserved under:
- `notebooks/archive/outputs_preprocessing_v3/`

## Notes

- No files were deleted or moved from `project_v2`.
- `project_v3` is created as a separate copy/restructure workspace.
- FastAPI files are currently placeholders and can be implemented incrementally.

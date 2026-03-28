# Notebooks Policy

- Production preprocessing pipeline: `src/pipelines/run_preprocess.py`
- Legacy notebook kept for reference only: `notebooks/archive/legacy_notebooks/preprocessing_ver_phobert_copy_2.ipynb`
- Do not generate production artifacts under `notebooks/`.
- Official artifacts path: `artifacts/matching/`.

## Current Evaluation Notebook

- Chatbot metric notebook for the current API pipeline:
  - `legacy/src/chatbot/metric_test/metric_test.ipynb`
- Focus:
  - chatbot end-to-end metrics
  - backend model currently configured as `llama3.1:8b`
  - correctness, generation quality, factuality, efficiency, and human-eval style summary
- The notebook is for evaluation/reporting only, not for generating production artifacts.

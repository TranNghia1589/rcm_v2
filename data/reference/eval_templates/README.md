# Eval Label Templates

Thu muc nay chua cac file mau de ban dan nhan thu cong cho bo metric trong `src/evaluation/`.

## Quyet dinh quy mo mau (phu hop voi hien trang 122 CV, 367 job)

Muc tieu la du lon de metric co y nghia thong ke ban dau, nhung van kha thi de dan nhan:

- `cv_extraction_field_labels.csv`: 80 CV (khoang 66% tap CV)
- `cv_scoring_calibration_labels.csv`: 80 CV
- `cv_scoring_stability_runs.csv`: 40 CV x 3 lan cham = 120 dong
- `skill_gap_labels.csv`: 80 CV
- `graph_query_cases.csv`: 30 CV x 3 loai query = 90 case
- `rag_retrieval_cases.csv`: 120 case
- `rag_groundedness_answers.csv`: 120 case
- `rag_citation_answers.csv`: 120 case
- `recommendation_qrels_labels.csv`: 50 CV x 20 job/CV = 1000 cap (cv, job)
- `recommendation_predictions_template.csv`: 50 CV x top-10 = 500 dong
- `recommendation_explanation_labels.csv`: 200 dong
- `system_api_cases.csv`: 12 endpoint probe
- `system_request_logs_template.csv`: 200 dong log mau

Neu ban muon dat muc do tin cay cao hon (CI hep hon), co the tang quy mo theo cung ti le.

## Cach sinh lai bo template

Chay:

```powershell
.\.venv\Scripts\python deploy/scripts/evaluation/generate_eval_templates.py --num_cv 122 --num_jobs 367
```

Script se doc:

- `data/processed/cv_extracted/cv_extracted_dataset.parquet` (lay file name CV)
- `experiments/artifacts/matching/jobs_matching_ready_v3.parquet` (lay so luong job)

va ghi de cac file trong thu muc nay.

## Mapping template -> evaluator

- `cv_extraction_field_labels.csv`
  - `src/evaluation/cv_extraction/eval_cv_extraction_field_accuracy.py`
- `cv_scoring_calibration_labels.csv`
  - `src/evaluation/cv_scoring/eval_cv_scoring_calibration.py`
- `cv_scoring_stability_runs.csv`
  - `src/evaluation/cv_scoring/eval_cv_scoring_stability.py`
- `skill_gap_labels.csv`
  - `src/evaluation/cv_scoring/eval_skill_gap_agreement.py`
- `rag_retrieval_cases.csv`
  - `src/evaluation/rag/eval_retrieval.py`
- `rag_groundedness_answers.csv`
  - `src/evaluation/rag/eval_groundedness.py`
- `rag_citation_answers.csv`
  - `src/evaluation/rag/eval_citation_correctness.py`
- `recommendation_qrels_labels.csv`
  - `src/evaluation/recommendation/eval_ranking.py`
- `recommendation_predictions_template.csv`
  - `src/evaluation/recommendation/eval_coverage.py`
  - `src/evaluation/recommendation/eval_diversity_novelty.py`
- `recommendation_explanation_labels.csv`
  - `src/evaluation/recommendation/eval_explanation_grounding.py`
- `graph_query_cases.csv`
  - `src/evaluation/graph/eval_graph_query_correctness.py`
- `system_api_cases.csv`
  - `src/evaluation/system/eval_api_latency_reliability.py`
- `system_request_logs_template.csv`
  - `src/evaluation/system/eval_fallback_timeout_rate.py`

## Quy uoc dinh dang list

Cac cot list dung JSON string:

- `["Python","SQL","Power BI"]`
- `[101,203,317]`

## Luu y quan trong

- Cac file co `cv_id`, `job_id` can trung voi ID thuc te trong DB khi ban chay eval.
- Template duoc tao de dan nhan, nen cac cot nhan (relevance, human_score, expected_missing_skills...) de trong la binh thuong.
- Dan nhan xong moi chay metric.



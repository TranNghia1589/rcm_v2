# Evaluation Framework (System-Wide)

This folder is the canonical evaluation suite for the project (excluding the chatbot notebook-level report flow).

## Design Goals

1. Cover all core subsystems end-to-end:
   - CV extraction
   - CV scoring + skill-gap
   - RAG retrieval + grounding
   - Recommendation ranking + beyond-accuracy
   - Graph data + graph query correctness
   - System reliability/performance
2. Keep each file single-responsibility to avoid metric overlap.
3. Make every metric traceable to an external research/book source.

See metric references in:
- [METRIC_SOURCES.md](d:/TTTN/project_v3/src/evaluation/METRIC_SOURCES.md)

## Folder Structure

- `common.py`: shared utilities (I/O, normalization, summary export, basic stats).
- `cv_extraction/`
  - `eval_cv_extraction_field_accuracy.py`
  - `eval_cv_extraction_parse_success.py`
- `cv_scoring/`
  - `eval_cv_scoring_calibration.py`
  - `eval_cv_scoring_stability.py`
  - `eval_skill_gap_agreement.py`
- `rag/`
  - `eval_retrieval.py`
  - `eval_groundedness.py`
  - `eval_citation_correctness.py`
- `recommendation/`
  - `build_predictions.py`
  - `build_qrels_template.py`
  - `eval_ranking.py`
  - `eval_coverage.py`
  - `eval_diversity_novelty.py`
  - `eval_explanation_grounding.py`
- `graph/`
  - `eval_graph_completeness.py`
  - `eval_graph_query_correctness.py`
- `system/`
  - `eval_api_latency_reliability.py`
  - `eval_fallback_timeout_rate.py`

## Non-overlap Rules (Clean Architecture)

To keep the suite professional and non-duplicative:

1. `recommendation/eval_coverage.py`
   - Scope: **catalog coverage only**.
   - Does not compute diversity.
2. `recommendation/eval_diversity_novelty.py`
   - Scope: **diversity + novelty only**.
3. `rag/eval_groundedness.py`
   - Scope: answer-level faithfulness/hallucination.
4. `rag/eval_citation_correctness.py`
   - Scope: citation formatting and citation validity against provided sources.
5. `system/eval_api_latency_reliability.py`
   - Scope: live endpoint probing (success rate, p50/p95/p99).
6. `system/eval_fallback_timeout_rate.py`
   - Scope: aggregated request-log quality signals.

## Typical Execution Order

1. CV extraction (`cv_extraction/*`)
2. Scoring + gap (`cv_scoring/*`)
3. RAG retrieval/grounding (`rag/*`)
4. Recommendation ranking + beyond-accuracy (`recommendation/*`)
5. Graph completeness/query correctness (`graph/*`)
6. API/system reliability (`system/*`)

## Output Convention

Each evaluator writes:
1. Summary file (`--output`)
2. Details file (`<stem>_details.<ext>`)

This supports dashboarding + debugging from the same run.


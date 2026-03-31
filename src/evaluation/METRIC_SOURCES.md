# Metric Sources and Provenance

This file maps each evaluation metric to its academic/industry origin so the framework is auditable.

## Core References

1. Manning, Raghavan, Schütze. *Introduction to Information Retrieval* (Cambridge, 2008).  
   URL: https://nlp.stanford.edu/IR-book/  
   Chapter on evaluation (Precision/Recall/F-measure/ranked metrics): https://nlp.stanford.edu/IR-book/pdf/08eval.pdf

2. Järvelin, Kekäläinen (2002). *Cumulated Gain-based Evaluation of IR Techniques*.  
   DOI: https://doi.org/10.1145/582415.582418

3. Voorhees, Tice (2000). *The TREC-8 Question Answering Track*.  
   ACL Anthology: https://aclanthology.org/L00-1018

4. Shani, Gunawardana (2011). *Evaluating Recommendation Systems* (Recommender Systems Handbook, Ch.8).  
   DOI: https://doi.org/10.1007/978-0-387-85820-3_8

5. Vargas, Castells (2011). *Rank and Relevance in Novelty and Diversity Metrics for Recommender Systems*.  
   DOI: https://doi.org/10.1145/2043932.2043955

6. Papineni et al. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation*.  
   ACL Anthology: https://aclanthology.org/P02-1040

7. Lin (2004). *ROUGE: A Package for Automatic Evaluation of Summaries*.  
   ACL Anthology: https://aclanthology.org/W04-1013

8. Zhang et al. (2020). *BERTScore: Evaluating Text Generation with BERT*.  
   OpenReview (ICLR): https://openreview.net/forum?id=SkeHuCVFDr

9. Es et al. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation*.  
   DOI: https://doi.org/10.48550/arXiv.2309.15217

10. Ji et al. (2023). *Survey of Hallucination in Natural Language Generation*.  
    DOI: https://doi.org/10.1145/3571730

11. Bohnet et al. (2022/2023). *Attributed Question Answering: Evaluation and Modeling for Attributed LLMs*.  
    arXiv record: https://arxiv.org/abs/2212.08037

12. Wang, Strong (1996). *Beyond Accuracy: What Data Quality Means to Data Consumers*.  
    DOI: https://doi.org/10.1080/07421222.1996.11518099

13. Hyndman, Koehler (2006). *Another Look at Measures of Forecast Accuracy*.  
    DOI: https://doi.org/10.1016/j.ijforecast.2006.03.001

14. Google SRE Book/Workbook (SLI/SLO, latency percentiles, reliability).  
    SLO chapter: https://sre.google/sre-book/service-level-objectives/  
    Implementing SLOs: https://sre.google/workbook/implementing-slos/

15. Jaccard (1901/1908) set-overlap coefficient origins (Jaccard similarity family).

## File-to-Metric Mapping

### `cv_extraction/eval_cv_extraction_field_accuracy.py`
- Metrics: field-level Precision, Recall, F1, Exact Match.
- Basis: IR/classification evaluation conventions from (1).

### `cv_extraction/eval_cv_extraction_parse_success.py`
- Metrics: parse success rate, completeness/fill-rate, validation error rate.
- Basis: data quality dimensions (accuracy/completeness) from (12).

### `cv_scoring/eval_cv_scoring_calibration.py`
- Metrics: MAE, RMSE, Pearson r, Spearman rho, within-tolerance rate.
- Basis:
  - error metrics from (13),
  - correlation analysis from classical statistics.

### `cv_scoring/eval_cv_scoring_stability.py`
- Metrics: score std, coefficient of variation, max delta, grade instability rate.
- Basis: repeatability/reliability diagnostics in measurement practice.

### `cv_scoring/eval_skill_gap_agreement.py`
- Metrics: Precision/Recall/F1, Jaccard, Hit@K between predicted and labeled missing skills.
- Basis:
  - set/ranking overlap from (1),
  - Jaccard family from (15).

### `cv_scoring/eval_role_benchmark_quality.py`
- Metrics: role coverage rate, market/profile skill list quality, benchmark age.
- Basis: data completeness/freshness quality dimensions from (12).

### `rag/eval_retrieval.py`
- Metrics: Precision@K, Recall@K, MRR, HitRate@K for chunk retrieval.
- Basis: ranked retrieval evaluation from (1), MRR usage from (3).

### `rag/eval_groundedness.py`
- Metrics: faithfulness, hallucination rate, fallback usage.
- Basis: RAG faithfulness framing from (9), hallucination framing from (10).

### `rag/eval_citation_correctness.py`
- Metrics: citation presence rate, citation validity precision.
- Basis: attributed QA / source-grounded generation from (11).

### `recommendation/eval_ranking.py`
- Metrics: Precision@K, Recall@K, HitRate@K, MRR@K, NDCG@K.
- Basis: ranked retrieval/recsys evaluation from (1), (2), (3), (4).

### `recommendation/eval_coverage.py`
- Metrics: catalog coverage@K.
- Basis: recommender beyond-accuracy dimensions from (4).

### `recommendation/eval_diversity_novelty.py`
- Metrics: intra-list diversity, novelty.
- Basis: beyond-accuracy novelty/diversity from (4), (5).

### `recommendation/eval_explanation_grounding.py`
- Metrics: evidence coverage in explanation, matched/missing skill mention ratio.
- Basis: explainability/grounding design informed by (9), (11).

### `graph/eval_graph_completeness.py`
- Metrics: ETL completeness ratio (Postgres vs Neo4j counts).
- Basis: data completeness and consistency dimensions from (12).

### `graph/eval_graph_query_correctness.py`
- Metrics: containment pass rate, exclusion pass rate, overall case pass rate.
- Basis: functional query-test pass-rate style from software testing practice.

### `system/eval_api_latency_reliability.py`
- Metrics: success rate, timeout rate, latency mean/p50/p95/p99.
- Basis: SLI/SLO and percentile latency operations practice from (14).

### `system/eval_fallback_timeout_rate.py`
- Metrics: fallback rate, timeout rate, 5xx rate, stage-level breakdown.
- Basis: reliability/error-budget oriented operational monitoring from (14).


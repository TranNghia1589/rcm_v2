# Migration Runbook: RAG + KG Recommendation (Current Structure)

Tai lieu nay da duoc cap nhat theo cau truc hien tai cua repo.

## Phase 0 - Environment

1. Bootstrap moi truong Python
- Script: `deploy/scripts/bootstrap.ps1`

2. Khoi tao PostgreSQL + pgvector
- Migrations:
  - `database/postgres/migrations/001_enable_pgvector.sql`
  - `database/postgres/migrations/002_create_rag_tables.sql`
  - `database/postgres/migrations/003_create_rag_indexes.sql`
  - `database/postgres/migrations/004_create_cv_scoring_tables.sql`
- Config:
  - `config/db/postgres.yaml`
  - `config/db/pgvector.yaml`
- Script:
  - `deploy/scripts/db/init_postgres_pgvector.ps1`

3. Khoi tao Neo4j schema
- Files:
  - `database/neo4j/schema/constraints.cypher`
  - `database/neo4j/schema/indexes.cypher`
  - `config/db/neo4j.yaml`
- Script:
  - `deploy/scripts/db/init_neo4j.ps1`

## Phase 1 - Data Processing

4. Crawl va preprocess jobs
- Crawl module:
  - `src/data_processing/crawl/topcv_crawler.py`
- Preprocess pipeline:
  - `src/data_processing/pipelines/run_preprocess.py`
- Artifacts output:
  - `experiments/artifacts/matching/*.parquet`

5. Extract CV + Gap analysis
- CV extraction:
  - `src/models/cv/extract_cv_info.py`
  - `src/models/cv/extract_cv_batch.py`
- Gap batch:
  - `src/models/cv/run_gap_batch.py`
- Output:
  - `data/processed/cv_extracted/*`
  - `data/processed/cv_gap_reports/*`

6. Load core tables vao PostgreSQL
- Loader:
  - `src/data_processing/ingestion/load_core_tables.py`

## Phase 2 - RAG Serving

7. Chunking + Embedding + Index
- Files:
  - `src/models/rag/chunking.py`
  - `src/models/rag/embed.py`
  - `src/models/rag/index.py`
  - `src/data_processing/ingestion/jobs_to_pgvector.py`

8. Retrieval + Prompting + Generation
- Files:
  - `src/models/rag/retrieve.py`
  - `src/models/rag/prompting.py`
  - `src/models/rag/generate.py`
  - `apps/api/app/services/rag/retrieval_service.py`
  - `apps/api/app/services/rag/chat_service.py`

9. Chat API
- Endpoint:
  - `apps/api/app/api/v1/chatbot.py`
- Schema:
  - `apps/api/app/schemas/rag/chat.py`

## Phase 3 - Knowledge Graph + Hybrid Recommend

10. Graph ETL
- Files:
  - `src/models/graph/etl.py`
  - `src/data_processing/ingestion/jobs_to_neo4j.py`
  - `src/data_processing/ingestion/cv_to_graph.py`

11. Graph Query Service
- Files:
  - `src/models/graph/query_service.py`
  - `database/neo4j/queries/recommend_jobs.cypher`
  - `database/neo4j/queries/user_skill_gap.cypher`
  - `database/neo4j/queries/career_path.cypher`

12. Hybrid recommendation orchestration
- Files:
  - `src/models/recommendation/candidate_generation.py`
  - `src/models/recommendation/graph_ranking.py`
  - `src/models/recommendation/explanation.py`
  - `src/models/recommendation/orchestrator.py`
  - `apps/api/app/services/recommendation/hybrid_recommender.py`

## Phase 4 - CV Scoring + API

13. CV scoring batch
- Files:
  - `src/models/scoring/cv_scoring.py`
  - `src/models/scoring/run_cv_scoring_batch.py`

14. API entrypoint
- App:
  - `apps/api/app/server.py`
- Main router:
  - `apps/api/app/api/v1/router.py`

## Phase 5 - Evaluation and Quality Gate

15. Evaluation modules
- `src/evaluation/cv_extraction/*`
- `src/evaluation/cv_scoring/*`
- `src/evaluation/rag/*`
- `src/evaluation/recommendation/*`
- `src/evaluation/graph/*`
- `src/evaluation/system/*`

16. API tests
- `apps/api/tests/*`

## Recommended execution order

1. `bootstrap`
2. `db init`
3. `run_local_pipeline.py --dry-run`
4. `run_local_pipeline.py --stages preprocess_jobs extract_cv cv_gap load_core_tables cv_scoring rag_ingest graph_etl api_tests`
5. `uvicorn apps.api.app.server:app`

## Notes
- Cac module trong `archive/` chi de tham khao, khong phai runtime chinh.
- Nguon config chinh la thu muc `config/`, khong dung `configs/`.

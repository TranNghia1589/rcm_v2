# Migration Runbook: RAG + KG Recommendation

This runbook gives the exact implementation order from current project state to target architecture.

## Phase 0 - Environment and services

1. Setup PostgreSQL + pgvector
- Tool/app: PostgreSQL + pgvector extension
- Files:
  - `database/postgres/migrations/001_enable_pgvector.sql`
  - `database/postgres/migrations/002_create_rag_tables.sql`
  - `database/postgres/migrations/003_create_rag_indexes.sql`
  - `configs/db/postgres.yaml`
  - `configs/db/pgvector.yaml`
- Work:
  - Create DB
  - Enable `vector` extension
  - Create full schema for jobs/cv/recommendation/chatbot + documents/chunks/embeddings
  - Create ANN indexes and metadata indexes

2. Setup Neo4j
- Tool/app: Neo4j (Desktop or Docker)
- Files:
  - `database/neo4j/schema/constraints.cypher`
  - `database/neo4j/schema/indexes.cypher`
  - `configs/db/neo4j.yaml`
- Work:
  - Create constraints for unique node keys
  - Create indexes for frequent query properties

3. Create startup scripts
- Tool/app: PowerShell
- Files:
  - `scripts/db/init_postgres_pgvector.ps1`
  - `scripts/db/init_neo4j.ps1`
  - `scripts/db/load_graph_data.ps1`

## Phase 1 - RAG ingestion pipeline

4. Build chunking policy
- Tool/app: Python
- Files:
  - `src/rag/chunking.py`
  - `configs/rag/chunking.yaml`
- Work:
  - Define chunk size/overlap/section strategy
  - Keep chunk metadata (`job_id`, `section_type`, `source`)

5. Build embedding layer
- Tool/app: embedding model provider
- Files:
  - `src/rag/embed.py`
  - `src/infrastructure/embeddings/provider.py`
  - `configs/model/embedding.yaml`
- Work:
  - Standardize embedding interface
  - Return consistent vector dimension

6. Index chunks into pgvector
- Tool/app: PostgreSQL + pgvector
- Files:
  - `src/rag/index.py`
  - `src/infrastructure/db/postgres_client.py`
  - `src/infrastructure/db/pgvector_store.py`
  - `src/ingestion/jobs_to_pgvector.py`
  - `scripts/pipelines/run_rag_ingest.ps1`
- Work:
  - Upsert chunks and embeddings
  - Store version and timestamp
  - Add incremental re-index mode

## Phase 2 - RAG query serving

7. Implement retrieval flow
- Tool/app: SQL + pgvector cosine similarity
- Files:
  - `src/rag/retrieve.py`
  - `configs/rag/retrieval.yaml`
  - `apps/api/app/services/rag/retrieval_service.py`
- Work:
  - Embed user query
  - Retrieve top-k chunks
  - Optional metadata filtering

8. Implement prompt assembly + generation
- Tool/app: LLM API (Ollama/OpenAI)
- Files:
  - `src/rag/prompting.py`
  - `src/rag/generate.py`
  - `src/infrastructure/llm/ollama_client.py`
  - `src/infrastructure/llm/openai_client.py`
  - `configs/rag/prompting.yaml`
  - `apps/api/app/services/rag/chat_service.py`
- Work:
  - Build grounded prompt from retrieved chunks
  - Return answer with source citations

9. Expose chatbot API
- Tool/app: FastAPI
- Files:
  - `apps/api/app/main.py`
  - `apps/api/app/api/v1/endpoints/chatbot.py`
  - `apps/api/app/schemas/rag/chat.py`
  - `apps/api/app/api/v1/router.py`
- Work:
  - `POST /api/v1/chat/ask`
  - response includes `answer`, `sources`, `latency_ms`

## Phase 3 - Build knowledge graph for recommendation

10. Define graph data model
- Tool/app: Neo4j + Cypher
- Files:
  - `src/graph/models.py`
  - `configs/graph/schema.yaml`
  - `database/neo4j/schema/constraints.cypher`
- Work:
  - Nodes: `User`, `CV`, `Skill`, `Role`, `Job`, `Company`
  - Relations: `HAS_SKILL`, `TARGETS_ROLE`, `REQUIRES_SKILL`, `LACKS_SKILL`, `MATCHES`

11. ETL jobs/CV into graph
- Tool/app: Python + Neo4j driver
- Files:
  - `src/graph/etl.py`
  - `src/ingestion/jobs_to_neo4j.py`
  - `src/ingestion/cv_to_graph.py`
  - `src/infrastructure/db/neo4j_client.py`
  - `scripts/pipelines/run_graph_etl.ps1`
- Work:
  - Map parquet/json to graph entities
  - Upsert nodes and edges

12. Implement core Cypher queries
- Tool/app: Cypher
- Files:
  - `database/neo4j/queries/recommend_jobs.cypher`
  - `database/neo4j/queries/user_skill_gap.cypher`
  - `database/neo4j/queries/career_path.cypher`
  - `src/graph/cypher_queries.py`
  - `src/graph/query_service.py`
- Work:
  - Candidate jobs by role+skill fit
  - Skill gap extraction
  - Career path context graph

## Phase 4 - Hybrid recommendation orchestration

13. Candidate generation (vector side)
- Tool/app: pgvector retriever
- Files:
  - `src/recommendation/candidate_generation.py`
- Work:
  - Get semantic candidates from job chunks/text

14. Graph re-ranking
- Tool/app: Cypher + graph score
- Files:
  - `src/recommendation/graph_ranking.py`
  - `src/recommendation/orchestrator.py`
  - `configs/recommendation/hybrid.yaml`
- Work:
  - Re-rank candidates by graph signals:
    - matched skill coverage
    - missing critical skills
    - role consistency

15. LLM explanation layer
- Tool/app: LLM API
- Files:
  - `src/recommendation/explanation.py`
  - `apps/api/app/services/recommendation/hybrid_recommender.py`
  - `apps/api/app/schemas/recommendation/job.py`
- Work:
  - Explain recommendation from graph+retrieval evidence
  - Force grounded responses (no unsupported claims)

16. Expose recommendation API
- Tool/app: FastAPI
- Files:
  - `apps/api/app/api/v1/endpoints/recommend.py`
  - `apps/api/app/schemas/graph/subgraph.py`
- Work:
  - `POST /api/v1/recommend/jobs`
  - `POST /api/v1/recommend/explain`

## Phase 5 - Evaluation and hardening

17. Add RAG evaluation
- Files:
  - `src/evaluation/rag/eval_retrieval.py`
  - `src/evaluation/rag/eval_groundedness.py`
  - `tests/rag/test_retrieve.py`

18. Add recommendation evaluation
- Files:
  - `src/evaluation/recommendation/eval_ranking.py`
  - `src/evaluation/recommendation/eval_coverage.py`
  - `tests/recommendation/test_hybrid_recommender.py`
  - `tests/graph/test_cypher_queries.py`

19. Add integration tests
- Files:
  - `tests/integration/test_chatbot_api.py`
  - `tests/integration/test_recommendation_api.py`

20. Cut over from legacy modules
- Legacy baseline:
  - `src/chatbot/retrieval.py`
  - `src/matching/recommend_engine.py`
- Work:
  - Keep legacy for benchmark comparison first
  - Switch API to new services after parity checks

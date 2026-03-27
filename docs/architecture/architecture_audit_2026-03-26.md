# Architecture Audit (2026-03-26)

## Scope reviewed
- Structure and file contents under `src/`, `apps/`, `configs/`, `scripts/`, `tests/`, `docs/`.
- Runtime flow for CV extraction, matching, chatbot.

## Confirmed current state

### 1) Chatbot
- Main logic is in:
  - `src/chatbot/retrieval.py`
  - `src/chatbot/advisor.py`
  - `scripts/dev/chatbot_1to1.py`
- Current chatbot context is built mostly from:
  - `gap_result` JSON
  - `recommended_jobs` list
  - market gap report
- External LLM call is optional via Ollama/llama.cpp, but retrieval is not yet served from a runtime vector database query flow.

### 2) Recommendation
- Core recommendation is in:
  - `src/matching/recommend_engine.py`
  - `src/matching/gap_analysis.py`
  - `src/matching/gap_engine.py`
- Current ranking is heuristic/rule-based:
  - skill overlap
  - role bonus
  - missing-skill heuristics
- No implemented graph query layer (Cypher/Neo4j) in runtime recommendation.

### 3) Preprocess and embedding artifacts
- `src/pipelines/run_preprocess.py` builds:
  - job-level and section-level embeddings
  - chunk-like chatbot sections
  - parquet/npy artifacts for downstream usage
- This is a good base for RAG ingestion, but not yet connected to pgvector serving APIs.

### 4) API and configs
- Many API/config/test files exist but are empty placeholders:
  - `apps/api/app/**`
  - `configs/**`
  - `apps/api/tests/**`
- `apps/api/app/main.py` currently contains chatbot script-like logic, not a completed FastAPI app structure.

## Gap against target architecture

### Target A: Basic RAG (PostgreSQL + pgvector)
- Missing parts:
  - Online ingestion pipeline to pgvector tables
  - Runtime semantic retrieval service from pgvector
  - Router/service/schema wiring for chat API
  - Retrieval evaluation + groundedness checks

### Target B: KG + LLM recommendation (Neo4j + Cypher)
- Missing parts:
  - Graph schema constraints/index
  - ETL from CV/jobs/skills to graph entities/relations
  - Cypher query service for candidate + gap extraction
  - Hybrid orchestration (vector candidate + graph rerank + LLM explanation)

## Decision
- Keep current working logic as legacy baseline.
- Scaffold new directories/files for the target architecture so implementation can proceed incrementally without breaking current pipeline.

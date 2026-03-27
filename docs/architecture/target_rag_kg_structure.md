# Target Structure (RAG + KG)

This structure is prepared for:
- Chatbot: Basic RAG with PostgreSQL + pgvector
- Recommendation: Knowledge Graph + Cypher + LLM explanation

## Tree

```text
project_v3/
  database/
    postgres/
      migrations/
        001_enable_pgvector.sql
        002_create_rag_tables.sql
        003_create_rag_indexes.sql
    neo4j/
      schema/
        constraints.cypher
        indexes.cypher
      queries/
        recommend_jobs.cypher
        user_skill_gap.cypher
        career_path.cypher

  src/
    rag/
      chunking.py
      embed.py
      index.py
      retrieve.py
      prompting.py
      generate.py
      pipeline.py
    graph/
      models.py
      etl.py
      cypher_queries.py
      query_service.py
      reasoning.py
    recommendation/
      candidate_generation.py
      graph_ranking.py
      explanation.py
      orchestrator.py
    ingestion/
      jobs_to_pgvector.py
      jobs_to_neo4j.py
      cv_to_graph.py
    infrastructure/
      db/
        postgres_client.py
        pgvector_store.py
        neo4j_client.py
      llm/
        ollama_client.py
        openai_client.py
      embeddings/
        provider.py
    evaluation/
      rag/
        eval_retrieval.py
        eval_groundedness.py
      recommendation/
        eval_ranking.py
        eval_coverage.py

  apps/
    api/
      app/
        services/
          rag/
            chat_service.py
            retrieval_service.py
          graph/
            graph_query_service.py
          recommendation/
            hybrid_recommender.py
        schemas/
          rag/
            chat.py
          recommendation/
            job.py
          graph/
            subgraph.py

  configs/
    db/
      postgres.yaml
      pgvector.yaml
      neo4j.yaml
    rag/
      chunking.yaml
      retrieval.yaml
      prompting.yaml
    graph/
      schema.yaml
      query.yaml
    recommendation/
      hybrid.yaml

  scripts/
    db/
      init_postgres_pgvector.ps1
      init_neo4j.ps1
      load_graph_data.ps1
    pipelines/
      run_rag_ingest.ps1
      run_graph_etl.ps1
      run_hybrid_recommend.ps1

  tests/
    rag/
      test_retrieve.py
    graph/
      test_cypher_queries.py
    recommendation/
      test_hybrid_recommender.py
    integration/
      test_chatbot_api.py
      test_recommendation_api.py
```

## Notes
- Missing files were scaffolded as empty placeholders so you can implement gradually.
- Existing modules under `src/chatbot` and `src/matching` can be treated as legacy baseline while migrating.

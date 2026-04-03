# Target Structure (RAG + KG) - Current

Cau truc duoi day phan anh trang thai hien tai cua repo (khong con la scaffold).

```text
job_recommendation_system/
  apps/
    api/
      app/
        api/v1/
        schemas/
        services/
    web/

  config/
    db/
    rag/
    model/
    recommendation/
    graph/
    pipeline/

  data/
    raw/
    processed/
    reference/

  database/
    postgres/
      migrations/
    neo4j/
      schema/
      queries/

  deploy/
    scripts/
      db/
      pipelines/
      dev/
    docker/
    k8s/

  experiments/
    artifacts/

  src/
    data_processing/
      crawl/
      preprocessing/
      ingestion/
      pipelines/
    models/
      cv/
      matching/
      scoring/
      rag/
      graph/
      recommendation/
    evaluation/
    utils/
      infrastructure/
      data_contracts/

  services/
    embedding/
    ingestion/
    graph/
    llm/
    nlp/

  tests/
  docs/
  archive/
```

## Core runtime modules
- API app: `apps/api/app/server.py`
- Pipeline runner: `deploy/scripts/run_local_pipeline.py`
- RAG chat service: `apps/api/app/services/rag/chat_service.py`
- Hybrid recommendation service: `apps/api/app/services/recommendation/hybrid_recommender.py`
- Graph query service: `src/models/graph/query_service.py`

## Notes
- Runtime config duoc doc tu `config/`.
- `archive/` chi de doi chieu, khong phai execution path chinh.
- `services/*` hien tai la wrapper/layer theo huong microservice, mot so module con rat nhe.

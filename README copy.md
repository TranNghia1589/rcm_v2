project_v3/                          (sau khi sắp xếp dự kiến)

├─ config/                           # đổi từ configs/
│  ├─ app/
│  ├─ db/
│  ├─ graph/
│  ├─ model/
│  ├─ pipeline/
│  ├─ rag/
│  └─ recommendation/
│
├─ data/
│  ├─ raw/
│  ├─ interim/                       # sẽ tạo mới (hiện chưa có)
│  ├─ processed/
│  └─ reference/
│
├─ notebooks/
├─ src/
│  ├─ crawl/
│  ├─ cv/
│  ├─ data_contracts/
│  ├─ evaluation/
│  ├─ graph/
│  ├─ ingestion/
│  ├─ infrastructure/
│  │  ├─ db/
│  │  ├─ embeddings/
│  │  └─ llm/
│  ├─ preprocessing/
│  ├─ rag/
│  ├─ recommendation/
│  └─ scoring/
│
├─ tests/
│  ├─ graph/
│  │  └─ test_cypher_queries.py
│  ├─ integration/
│  │  ├─ test_chatbot_api.py         # từ apps/api/tests/test_chatbot.py
│  │  ├─ test_recommendation_api.py  # từ apps/api/tests/test_recommend.py
│  │  ├─ test_cv_score_api.py        # từ apps/api/tests/test_cv_score.py
│  │  ├─ test_jobs_api.py            # từ apps/api/tests/test_jobs.py
│  │  └─ test_health_api.py          # từ apps/api/tests/test_health.py
│  ├─ rag/
│  │  └─ test_retrieve.py
│  └─ recommendation/
│     └─ test_hybrid_recommender.py
│
├─ deploy/                           # đổi từ infra/ + gom file deploy root
│  ├─ docker/
│  │  ├─ api.Dockerfile
│  │  ├─ web.Dockerfile
│  │  └─ worker.Dockerfile
│  ├─ k8s/
│  │  ├─ api-deployment.yaml
│  │  ├─ web-deployment.yaml
│  │  ├─ worker-deployment.yaml
│  │  └─ ingress.yaml
│  ├─ terraform/
│  │  └─ README.md
│  └─ docker-compose.yml             # từ root docker-compose.yml
│
├─ database/
│  ├─ neo4j/
│  │  ├─ queries/
│  │  └─ schema/
│  └─ postgres/
│     └─ migrations/                 # canonical migration duy nhất
│
├─ docs/
│  ├─ api/
│  ├─ architecture/
│  ├─ ml/
│  ├─ product/
│  ├─ runbooks/
│  └─ reference/
│     └─ project-structure-guide.md  # từ README copy.md
│
├─ experiments/                      # đổi từ artifacts/
│  ├─ evaluation/
│  ├─ matching/
│  └─ hybrid_response.json
│
├─ apps/
│  ├─ api/
│  │  └─ app/
│  └─ web/
│
├─ scripts/
├─ requirements/
├─ .gitignore
├─ pyproject.toml
└─ README.md

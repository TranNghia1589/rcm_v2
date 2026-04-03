# Pipeline Commands (Windows PowerShell)

Tai lieu nay da duoc dong bo theo cau truc hien tai cua repo.

## 0) Di chuyen vao root du an
```powershell
cd D:\TTTN\job_recommendation_system
```

## 1) Bootstrap nhanh (tao venv + cai dependencies)
```powershell
powershell -ExecutionPolicy Bypass -File deploy\scripts\bootstrap.ps1
```

## 2) Khoi tao schema DB
```powershell
powershell -ExecutionPolicy Bypass -File deploy\scripts\db\init_postgres_pgvector.ps1
powershell -ExecutionPolicy Bypass -File deploy\scripts\db\init_neo4j.ps1 -CypherShellPath "<path-to-cypher-shell.bat>" -Neo4jUser neo4j -Neo4jPassword "<password>" -Neo4jDatabase neo4j
```

## 3) Chay pipeline local (khuyen nghi)
Dry-run:
```powershell
python deploy\scripts\run_local_pipeline.py --dry-run
```

Full stages:
```powershell
python deploy\scripts\run_local_pipeline.py --stages preprocess_jobs extract_cv cv_gap load_core_tables cv_scoring rag_ingest graph_etl api_tests
```

Neu muon skip PhoBERT embedding trong preprocess:
```powershell
python deploy\scripts\run_local_pipeline.py --stages preprocess_jobs extract_cv cv_gap load_core_tables cv_scoring rag_ingest graph_etl api_tests --skip-preprocess-embedding
```

## 4) Chay tung stage
```powershell
python deploy\scripts\run_local_pipeline.py --stages preprocess_jobs
python deploy\scripts\run_local_pipeline.py --stages extract_cv cv_gap
python deploy\scripts\run_local_pipeline.py --stages load_core_tables cv_scoring
python deploy\scripts\run_local_pipeline.py --stages rag_ingest
python deploy\scripts\run_local_pipeline.py --stages graph_etl
python deploy\scripts\run_local_pipeline.py --stages api_tests
```

## 5) Chay API local
```powershell
python -m uvicorn apps.api.app.server:app --host 0.0.0.0 --port 8000 --reload
```

## 6) API smoke checks
```powershell
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/docs
curl http://127.0.0.1:8000/api/v1/cv/score/1
```

## 7) Chay Docker Compose (tuy chon)
```powershell
docker compose up -d postgres neo4j
docker compose up -d api
docker compose down
```

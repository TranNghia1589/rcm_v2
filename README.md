# Career AI Platform (Project V3)

Backend cho hệ thống gợi ý việc làm + phân tích CV + chatbot tư vấn CV.

## Runtime Chính
- API: `apps/api/app/server.py`
- Pipeline local runner: `scripts/run_local_pipeline.py`
- Preprocess jobs: `src/pipelines/run_preprocess.py`

Notebook cũ chỉ để tham khảo, không phải nguồn chạy production.

## Cấu trúc dữ liệu
- `data/raw/jobs/`: dữ liệu crawl job đầu vào (`topcv_all_fields_merged_*.csv|xlsx`)
- `data/raw/cv_samples/INFORMATION-TECHNOLOGY/`: CV PDF
- `data/processed/cv_extracted/`: CV JSON + dataset parquet/jsonl
- `data/processed/cv_gap_reports/`: kết quả gap theo CV
- `artifacts/matching/`: artifacts cho matching/rag

## Setup môi trường
```powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1
```

## Khởi tạo DB schema
```powershell
powershell -ExecutionPolicy Bypass -File scripts\db\init_postgres_pgvector.ps1
powershell -ExecutionPolicy Bypass -File scripts\db\init_neo4j.ps1 -CypherShellPath "<path-to-cypher-shell.bat>" -Neo4jUser neo4j -Neo4jPassword "<password>" -Neo4jDatabase neo4j
```

## Chạy pipeline (khuyến nghị)
Dry-run (xem lệnh):
```powershell
python scripts/run_local_pipeline.py --dry-run
```

Chạy full stages:
```powershell
python scripts/run_local_pipeline.py --stages preprocess_jobs extract_cv cv_gap load_core_tables cv_scoring rag_ingest graph_etl api_tests
```

Nếu môi trường PhoBERT chưa sẵn sàng (model/torch), có thể tạm skip embedding ở preprocess:
```powershell
python scripts/run_local_pipeline.py --stages preprocess_jobs extract_cv cv_gap load_core_tables cv_scoring rag_ingest graph_etl api_tests --skip-preprocess-embedding
```

## Chạy từng stage
```powershell
python scripts/run_local_pipeline.py --stages preprocess_jobs
python scripts/run_local_pipeline.py --stages extract_cv cv_gap
python scripts/run_local_pipeline.py --stages load_core_tables cv_scoring
python scripts/run_local_pipeline.py --stages rag_ingest
python scripts/run_local_pipeline.py --stages graph_etl
python scripts/run_local_pipeline.py --stages api_tests
```

## Chạy API local
```powershell
python -m uvicorn apps.api.app.server:app --host 0.0.0.0 --port 8000 --reload
```

## Chạy bằng Docker Compose
```powershell
docker compose up -d postgres neo4j
docker compose up -d api
```

Tắt dịch vụ:
```powershell
docker compose down
```

Endpoints chính:
- `GET /healthz`
- `POST /api/v1/chat/ask`
- `GET /api/v1/cv/score/{cv_id}`
- `POST /api/v1/recommend/hybrid`
- `POST /api/v1/recommend/graph/jobs`

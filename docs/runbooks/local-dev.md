# Local Dev

Tài liệu này mô tả luồng chạy local theo cấu trúc project hiện tại.

## 1) Bootstrap môi trường
```powershell
powershell -ExecutionPolicy Bypass -File deploy\scripts\bootstrap.ps1
```

## 2) Khởi tạo schema DB
PostgreSQL + pgvector:
```powershell
powershell -ExecutionPolicy Bypass -File deploy\scripts\db\init_postgres_pgvector.ps1
```

Neo4j:
```powershell
powershell -ExecutionPolicy Bypass -File deploy\scripts\db\init_neo4j.ps1 -CypherShellPath "<path-to-cypher-shell.bat>" -Neo4jUser neo4j -Neo4jPassword "<password>" -Neo4jDatabase neo4j
```

## 3) Chạy pipeline local bằng stage runner
Xem trước lệnh:
```powershell
python deploy/scripts/run_local_pipeline.py --dry-run
```

Chạy đầy đủ:
```powershell
python deploy/scripts/run_local_pipeline.py --stages preprocess_jobs extract_cv cv_gap load_core_tables cv_scoring rag_ingest graph_etl api_tests
```

Nếu PhoBERT preprocess chưa sẵn sàng trong môi trường hiện tại, chạy tạm:
```powershell
python deploy/scripts/run_local_pipeline.py --stages preprocess_jobs extract_cv cv_gap load_core_tables cv_scoring rag_ingest graph_etl api_tests --skip-preprocess-embedding
```

## 4) Chạy API local
```powershell
python -m uvicorn apps.api.app.server:app --host 0.0.0.0 --port 8000 --reload
```

## 4.1) Chạy Docker Compose (tuỳ chọn)
```powershell
docker compose up -d postgres neo4j
docker compose up -d api
```

Stop:
```powershell
docker compose down
```

Kiểm tra nhanh:
- `GET /healthz`
- `POST /api/v1/chat/ask`
- `GET /api/v1/cv/score/{cv_id}`
- `POST /api/v1/recommend/hybrid`

## 5) Test API
```powershell
python deploy/scripts/run_local_pipeline.py --stages api_tests
```

## Ghi chú
- Tài liệu này chỉ đồng bộ lệnh chạy thực tế, không thay đổi logic xử lý.
- Warning `SwigPy*` trong pytest là warning dependency, không làm fail pipeline.



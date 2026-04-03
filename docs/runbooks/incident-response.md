# Incident Response (Local/Staging)

## Muc tieu
Xu ly nhanh cac su co runtime API/RAG/Recommendation trong local hoac staging.

## 1) Xac dinh muc do
- P0: API down hoan toan (`/healthz` fail)
- P1: endpoint chinh loi hang loat (`/chat/ask`, `/recommend/*`)
- P2: quality giam (fallback rate cao, ket qua khong grounded)

## 2) Checklist trong 10 phut dau
```powershell
docker compose ps
docker compose logs api --tail=200
docker compose logs postgres --tail=200
docker compose logs neo4j --tail=200
curl http://127.0.0.1:8000/healthz
```

## 3) Nhom su co pho bien va cach xu ly

### A. API loi ket noi DB
- Trieu chung: 500 o endpoint phu thuoc Postgres/Neo4j
- Xu ly:
  1. Kiem tra bien moi truong trong `.env`
  2. Kiem tra `config/db/*.yaml` (co de password rong dung fallback)
  3. Restart service:
```powershell
docker compose restart postgres neo4j api
```

### B. Chatbot fallback qua nhieu
- Trieu chung: `used_fallback=true` tang bat thuong
- Xu ly:
  1. Kiem tra bang/chunks RAG da ingest chua
  2. Chay lai stage ingest:
```powershell
python deploy/scripts/run_local_pipeline.py --stages rag_ingest
```

### C. Recommend ket qua trong/khong hop ly
- Xu ly:
  1. Kiem tra graph ETL:
```powershell
python deploy/scripts/run_local_pipeline.py --stages graph_etl
```
  2. Kiem tra du lieu matching artifacts trong `experiments/artifacts/matching`

## 4) Xac nhan sau fix
```powershell
python -m pytest -q apps/api/tests tests/integration tests/rag tests/recommendation tests/graph
```

## 5) Hau kiem
- Ghi lai nguyen nhan goc va cach khac phuc vao changelog/runbook.
- Neu la bug code: bo sung test hoi quy truc tiep tai `tests/*`.

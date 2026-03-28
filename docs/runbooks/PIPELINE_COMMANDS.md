# Pipeline Commands (Windows PowerShell)

## 0) Vao thu muc du an
```powershell
cd D:\TTTN\project_v3
```

## 1) Tao va kich hoat venv
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 2) Cai dependencies
```powershell
pip install -r requirements\dev.txt
pip install -r requirements\api.txt
```

## 3) Crawl jobs (tuy chon)
```powershell
python src\crawl\topcv_crawler.py
```

## 4) Preprocess jobs + build artifacts
```powershell
python src\pipelines\run_preprocess.py
```

## 4.1) Hoac chay full bootstrap theo thu tu
```powershell
powershell -ExecutionPolicy Bypass -File scripts\pipelines\run_full_bootstrap.ps1
```

## 4.2) CV scoring batch (Giai doan 3)
```powershell
powershell -ExecutionPolicy Bypass -File scripts\pipelines\run_cv_scoring_batch.ps1
```

## 5) Trich xuat 1 CV
```powershell
python src\cv\extract_cv_info.py --cv_path data\raw\cv_samples\cv_data_manual.txt --output_path data\processed\resume_extracted.json
```

## 6) Trich xuat batch CV (folder)
```powershell
powershell -ExecutionPolicy Bypass -File scripts\pipelines\run_cv_extract_batch.ps1
```

## 7) Gap analysis
```powershell
python src\matching\gap_analysis.py --cv_json data\processed\resume_extracted.json --output_path data\processed\gap_analysis_result.json
```

## 8) Run API (active entrypoint)
```powershell
python -m uvicorn apps.api.app.server:app --host 0.0.0.0 --port 8000 --reload
```

## 9) API quick checks
```powershell
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/docs
curl http://127.0.0.1:8000/api/v1/cv/score/1
```

## 9.1) Hybrid recommend alias endpoint
```powershell
powershell -ExecutionPolicy Bypass -File scripts\pipelines\run_hybrid_recommend.ps1 -CvId 1 -Question "Goi y cong viec phu hop"
```

## 10) Evaluate baseline cases
```powershell
python src\pipelines\run_eval.py
```

## DB setup cho database moi
```powershell
powershell -ExecutionPolicy Bypass -File scripts\db\init_postgres_pgvector.ps1
powershell -ExecutionPolicy Bypass -File scripts\db\init_neo4j.ps1
```

# Pipeline Commands (Windows PowerShell)

## 0) Vao thu muc du an
```powershell
cd <duong_dan_den_project_v2>
```

## 1) Tao va kich hoat venv
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 2) Cai dependencies
```powershell
pip install -r requirements_chatbot.txt
pip install beautifulsoup4 openpyxl pyarrow sentencepiece
pip install torch tqdm transformers underthesea
```

## 3) Kiem tra nhanh moi truong
```powershell
python -c "import torch,tqdm,transformers,underthesea,pandas,numpy; print('OK')"
```

## 4) Crawl du lieu jobs (tuy chon, neu can data moi)
```powershell
python scripts\crawl_topcv.py
```

## 5) Chay preprocessing jobs
```powershell
python src\pipelines\preprocessing.py
```

## 6) Trich xuat thong tin CV
```powershell
python src\cv_processing\extract_cv_info.py --cv_path data\raw\cv_samples\cv_data_manual.txt --output_path data\processed\resume_extracted.json
```

## 7) Gap analysis
```powershell
python src\matching\gap_analysis.py --cv_json data\processed\resume_extracted.json --output_path data\processed\gap_analysis_result.json
```

## 8) Khoi dong Ollama
```powershell
ollama serve
```

Mo terminal thu 2:
```powershell
ollama pull llama3:latest
ollama list
```

## 9) Test chatbot theo file gap analysis
```powershell
python src\chatbot\chat_router.py --question "CV cua toi phu hop vi tri nao?" --gap_result data\processed\gap_analysis_result.json
```

## 10) Chay interactive test (tuy chon)
```powershell
python scripts\test_chatbot_interactive.py
```

## 11) Neu bi loi proxy khi pull model
```powershell
Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:ALL_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:GIT_HTTP_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:GIT_HTTPS_PROXY -ErrorAction SilentlyContinue
ollama pull llama3:latest
```

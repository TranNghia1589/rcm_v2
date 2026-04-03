# Architecture Review

## Current Status
He thong hien tai da chuyen sang cau truc thong nhat va co runtime chinh:
- API: `apps/api/app/server.py`
- Pipeline runner: `deploy/scripts/run_local_pipeline.py`
- Data processing: `src/data_processing/*`
- Core models: `src/models/*`
- Evaluation: `src/evaluation/*`

## Canonical Structure Pointers
- Overall structure: `docs/architecture/target_rag_kg_structure.md`
- Local setup/run: `docs/runbooks/local-dev.md`
- Pipeline commands: `docs/runbooks/PIPELINE_COMMANDS.md`

## Practical Notes
- Thu muc `config/` la nguon cau hinh hien hanh.
- Thu muc `archive/` chi de doi chieu tham khao, khong phai runtime.
- Neu co xung dot tai lieu, uu tien `README.md` va cac runbook trong `docs/runbooks/`.

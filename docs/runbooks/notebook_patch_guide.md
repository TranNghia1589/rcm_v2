# Notebook Patch Guide (Current)

## Muc tieu
Tai lieu nay huong dan khi nao nen sua notebook va khi nao phai dua logic ve script runtime.

## Nguyen tac quan trong
- Notebook chi dung cho EDA/phan tich thu nghiem.
- Logic production phai nam o:
  - `src/data_processing/pipelines/*`
  - `src/models/*`
  - `deploy/scripts/run_local_pipeline.py`
- Khong coi notebook la "source of truth" cho pipeline chinh.

## Khi can patch notebook
- Ban can tai lap nhanh mot phan tich trong bao cao.
- Ban can tao chart/phan tich bo tro cho evaluation.
- Ban KHONG can patch notebook neu thay doi do thuoc runtime API/pipeline.

## Quy trinh patch de khong lech runtime
1. Xac dinh notebook muc tieu trong `notebooks/`.
2. Ghi ro pham vi patch (chi markdown/EDA hay co logic xu ly).
3. Neu co logic xu ly:
   - Refactor sang module Python trong `src/*` truoc.
   - Notebook goi lai ham tu module do.
4. Chay smoke test sau patch:
```powershell
python -m pytest -q apps/api/tests tests/rag tests/recommendation tests/graph
```
5. Cap nhat runbook lien quan neu patch anh huong command van hanh.

## Checklist review truoc khi merge
- Notebook con chay duoc tu dau den cuoi.
- Khong co hardcode path may ca nhan.
- Khong co secret/token/password trong output cell.
- Neu co thay doi logic: da co test trong `tests/*`.

## Canonical references
- `README.md`
- `docs/runbooks/local-dev.md`
- `docs/runbooks/PIPELINE_COMMANDS.md`
- `docs/runbooks/MIGRATION_RAG_KG_STEPS.md`

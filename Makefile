.PHONY: help bootstrap pipeline pipeline-dry test-api run-api db-init-pg db-init-neo4j

help:
	@echo "make bootstrap        - install dev/api dependencies"
	@echo "make pipeline         - run full local data pipeline"
	@echo "make pipeline-dry     - print pipeline commands only"
	@echo "make test-api         - run API tests"
	@echo "make run-api          - start FastAPI app"
	@echo "make db-init-pg       - initialize PostgreSQL schema"
	@echo "make db-init-neo4j    - initialize Neo4j constraints/indexes"

bootstrap:
	@powershell -ExecutionPolicy Bypass -File deploy\scripts\bootstrap.ps1

pipeline:
	@powershell -ExecutionPolicy Bypass -Command "$py = if (Test-Path '.\.venv\Scripts\python.exe') { '.\.venv\Scripts\python.exe' } else { 'python' }; & $$py deploy\scripts\run_local_pipeline.py"

pipeline-dry:
	@powershell -ExecutionPolicy Bypass -Command "$py = if (Test-Path '.\.venv\Scripts\python.exe') { '.\.venv\Scripts\python.exe' } else { 'python' }; & $$py deploy\scripts\run_local_pipeline.py --dry-run"

test-api:
	@powershell -ExecutionPolicy Bypass -Command "$py = if (Test-Path '.\.venv\Scripts\python.exe') { '.\.venv\Scripts\python.exe' } else { 'python' }; & $$py -m pytest apps/api/tests -q -p no:cacheprovider"

run-api:
	@powershell -ExecutionPolicy Bypass -Command "$py = if (Test-Path '.\.venv\Scripts\python.exe') { '.\.venv\Scripts\python.exe' } else { 'python' }; & $$py -m uvicorn apps.api.app.server:app --host 0.0.0.0 --port 8000 --reload"

db-init-pg:
	@powershell -ExecutionPolicy Bypass -File deploy\scripts\db\init_postgres_pgvector.ps1

db-init-neo4j:
	@powershell -ExecutionPolicy Bypass -File deploy\scripts\db\init_neo4j.ps1

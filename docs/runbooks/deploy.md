# Deploy Runbook (Current)

## Scope
Tai lieu nay mo ta deploy local demo/staging theo cau truc repo hien tai.

## 1) Prerequisites
- Docker Desktop
- File `.env` da duoc tao tu `.env.example` va co gia tri that cho:
  - `POSTGRES_PASSWORD`
  - `NEO4J_PASSWORD`

## 2) Build images
```powershell
docker compose build embedding api
```

## 3) Start services
```powershell
docker compose up -d postgres neo4j
docker compose up -d embedding api
```

## 4) Health checks
```powershell
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/docs
```

## 5) Optional: run web app locally
```powershell
npm --prefix apps/web install
npm --prefix apps/web run dev
```

## 6) Stop services
```powershell
docker compose down
```

## 7) Canonical files
- Compose: `docker-compose.yml`
- API image: `deploy/docker/api.Dockerfile`
- Embedding image: `deploy/docker/embedding.Dockerfile`
- K8s manifests: `deploy/k8s/*`

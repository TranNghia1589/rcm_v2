from apps.api.app.api.v1 import jobs as jobs_endpoint
from apps.api.app.server import app
from fastapi.testclient import TestClient


class _DummyJobsService:
    def list_jobs(self, *, limit: int = 20, offset: int = 0):
        return [
            {
                "job_id": 1,
                "title": "Backend Engineer",
                "company_name": "Demo Co",
                "location": "HCM",
                "job_family": "Software Engineering",
                "work_mode": "hybrid",
                "is_active": True,
                "updated_at": "2026-03-28T00:00:00",
            }
        ]


def test_jobs_list_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(jobs_endpoint, "get_jobs_service", lambda: _DummyJobsService())
    client = TestClient(app)
    res = client.get("/api/v1/jobs?limit=10&offset=0")

    assert res.status_code == 200
    body = res.json()
    assert len(body["items"]) == 1
    assert body["items"][0]["job_id"] == 1

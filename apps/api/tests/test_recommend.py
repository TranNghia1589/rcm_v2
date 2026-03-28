from apps.api.app.api.v1.endpoints import recommend as recommend_endpoint
from apps.api.app.server import app
from fastapi.testclient import TestClient


class _DummyHybridService:
    def recommend(self, *, question: str, cv_id: int) -> dict:
        return {
            "cv_id": cv_id,
            "question": question,
            "items": [
                {
                    "job_id": 1,
                    "title": "Data Analyst",
                    "company_name": "Demo Co",
                    "location": "HCM",
                    "matched_skills": 3,
                    "total_required": 5,
                    "coverage": 0.6,
                    "vector_score": 0.7,
                    "graph_score": 0.65,
                    "hybrid_score": 0.68,
                    "supporting_chunk_ids": [1, 2],
                }
            ],
            "explanation": "demo",
        }


def test_recommend_hybrid_alias_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(recommend_endpoint, "get_hybrid_service", lambda: _DummyHybridService())
    client = TestClient(app)
    payload = {"cv_id": 1, "question": "goi y cong viec"}

    r1 = client.post("/api/v1/recommend/hybrid/jobs", json=payload)
    r2 = client.post("/api/v1/recommend/hybrid", json=payload)

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["cv_id"] == 1
    assert r2.json()["items"][0]["job_id"] == 1

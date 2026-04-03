from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.app.api.v1 import recommendations as recommend_endpoint
from apps.api.app.server import app


class _DummyHybridService:
    def recommend(self, *, question: str, cv_id: int) -> dict:
        return {
            "cv_id": cv_id,
            "question": question,
            "items": [
                {
                    "job_id": 10,
                    "title": "Data Scientist",
                    "company_name": "Demo AI",
                    "location": "HCM",
                    "matched_skills": 4,
                    "total_required": 6,
                    "coverage": 0.67,
                    "vector_score": 0.8,
                    "graph_score": 0.7,
                    "hybrid_score": 0.75,
                    "supporting_chunk_ids": [7, 8],
                }
            ],
            "explanation": "Top fit from vector + graph evidence.",
        }


class _DummyGraphService:
    def __init__(self, neo4j_config_path):
        self.neo4j_config_path = neo4j_config_path

    def recommend_jobs(self, cv_id: int, limit: int = 10) -> list[dict]:
        return [
            {
                "job_id": 3,
                "title": "Data Analyst",
                "company_name": "Graph Co",
                "location": "HN",
                "matched_skills": 3,
                "total_required": 5,
                "coverage": 0.6,
                "score": 0.62,
            }
        ]

    def skill_gap(self, cv_id: int, limit: int = 15) -> list[dict]:
        return [{"missing_skill": "tableau", "freq": 8}]

    def career_path(self, cv_id: int, limit: int = 5) -> list[dict]:
        return [{"role": "Senior Data Analyst", "transition_score": 0.7}]


def test_recommendation_api_hybrid_contract(monkeypatch) -> None:
    monkeypatch.setattr(recommend_endpoint, "get_hybrid_service", lambda: _DummyHybridService())
    client = TestClient(app)

    payload = {"cv_id": 8, "question": "goi y viec lam data phu hop"}
    res = client.post("/api/v1/recommend/hybrid/jobs", json=payload)

    assert res.status_code == 200
    body = res.json()
    assert body["cv_id"] == 8
    assert body["items"][0]["job_id"] == 10
    assert body["items"][0]["supporting_chunk_ids"] == [7, 8]
    assert body["explanation"] != ""


def test_recommendation_api_graph_skill_gap_contract(monkeypatch) -> None:
    monkeypatch.setattr(recommend_endpoint, "GraphQueryAppService", _DummyGraphService)
    client = TestClient(app)

    res = client.post("/api/v1/recommend/graph/skill-gap", json={"cv_id": 5, "limit": 10})
    assert res.status_code == 200
    body = res.json()
    assert body["cv_id"] == 5
    assert len(body["items"]) == 1
    assert body["items"][0]["missing_skill"] == "tableau"


def test_recommendation_api_validates_cv_id() -> None:
    client = TestClient(app)
    res = client.post("/api/v1/recommend/hybrid/jobs", json={"cv_id": 0, "question": "abc"})
    assert res.status_code == 422

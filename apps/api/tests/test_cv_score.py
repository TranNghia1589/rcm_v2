from apps.api.app.api.v1 import cv as cv_endpoint
from apps.api.app.server import app
from fastapi.testclient import TestClient


class _DummyCVScoreService:
    def get_cv_score(self, *, cv_id: int):
        return {
            "cv_id": cv_id,
            "total_score": 77.5,
            "grade": "B",
            "benchmark_role": "Data Analyst",
            "skill_score": 25.0,
            "experience_score": 18.0,
            "project_score": 15.0,
            "education_score": 9.0,
            "completeness_score": 10.5,
            "strengths": ["sql", "python"],
            "missing_skills": ["tableau"],
            "priority_skills": ["tableau"],
            "development_plan_30_60_90": {"day_30": [], "day_60": [], "day_90": []},
            "subscores_json": {"weights": {}},
            "metadata": {},
            "model_version": "cv_scoring_v1",
            "updated_at": "2026-03-28T00:00:00",
        }


def test_cv_score_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(cv_endpoint, "get_cv_score_service", lambda: _DummyCVScoreService())
    client = TestClient(app)
    res = client.get("/api/v1/cv/score/1")
    assert res.status_code == 200
    assert res.json()["cv_id"] == 1
    assert "total_score" in res.json()

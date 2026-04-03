from __future__ import annotations

from apps.api.app.api.v1 import career as career_endpoint
from apps.api.app.server import app
from fastapi.testclient import TestClient


class _DummyCareerService:
    def analyze(
        self,
        *,
        intent: str,
        query: str,
        preferred_role: str | None,
        cv_text: str | None,
        cv_file_path: str | None,
        cv_filename: str | None,
        debug: bool = False,
    ) -> dict:
        return {
            "cv_snapshot": {
                "internal_cv_id": "abc123",
                "source_type": "raw_text",
                "extracted_target_role": "Data Analyst",
                "experience_years": "2",
                "skills": ["SQL", "Python"],
                "projects_count": 1,
            },
            "role_inference": {
                "selected_role": "Data Analyst",
                "confidence": 0.82,
                "requires_confirmation": False,
                "candidates": [
                    {"role": "Data Analyst", "confidence": 0.82, "reasons": ["matched_skills_ratio=0.67"]},
                    {"role": "Data Scientist", "confidence": 0.41, "reasons": ["low_signal_match"]},
                ],
            },
            "orchestration": {
                "intent": "score",
                "next_actions": ["parse_cv_completed", "run_score_pipeline"],
                "notes": [],
            },
            "debug_payload": {"query": query} if debug else None,
        }


def test_career_analyze_new_user_flow(monkeypatch) -> None:
    monkeypatch.setattr(career_endpoint, "get_career_service", lambda: _DummyCareerService())
    client = TestClient(app)

    payload = {
        "intent": "auto",
        "query": "toi muon cham diem cv",
        "cv_text": "Data Analyst with SQL and Python experience",
    }
    res = client.post("/api/v1/career/analyze", json=payload)

    assert res.status_code == 200
    body = res.json()
    assert body["cv_snapshot"]["internal_cv_id"] == "abc123"
    assert body["role_inference"]["selected_role"] == "Data Analyst"
    assert body["orchestration"]["intent"] == "score"


def test_career_analyze_requires_cv_source() -> None:
    client = TestClient(app)
    res = client.post("/api/v1/career/analyze", json={"intent": "score", "query": "hello"})
    assert res.status_code == 422

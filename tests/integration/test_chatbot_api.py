from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.app.api.v1 import chatbot as chatbot_endpoint
from apps.api.app.server import app


class _DummyChatService:
    def ask(self, *, question: str, top_k: int = 5, cv_id: int | None = None) -> dict:
        return {
            "answer": f"Q={question}; cv_id={cv_id}; top_k={top_k}",
            "sources": [
                {"chunk_id": 1, "document_id": 100, "title": "JD Data Analyst", "distance": 0.12},
                {"chunk_id": 2, "document_id": 101, "title": "JD Data Engineer", "distance": 0.2},
            ],
            "retrieval_count": 2,
            "used_fallback": False,
            "fallback_reason": "",
            "fallback_stage": "",
        }


def test_chat_api_contract_contains_sources_and_metadata(monkeypatch) -> None:
    monkeypatch.setattr(chatbot_endpoint, "get_chat_service", lambda: _DummyChatService())
    client = TestClient(app)

    payload = {"question": "toi nen cai thien CV nhu the nao", "top_k": 2, "cv_id": 3}
    res = client.post("/api/v1/chat/ask", json=payload)

    assert res.status_code == 200
    body = res.json()
    assert body["retrieval_count"] == 2
    assert body["used_fallback"] is False
    assert len(body["sources"]) == 2
    assert body["sources"][0]["chunk_id"] == 1


def test_chat_api_validates_minimum_question_length() -> None:
    client = TestClient(app)
    payload = {"question": "ab", "top_k": 2}
    res = client.post("/api/v1/chat/ask", json=payload)
    assert res.status_code == 422

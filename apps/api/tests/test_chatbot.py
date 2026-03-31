from apps.api.app.api.v1 import chatbot as chatbot_endpoint
from apps.api.app.server import app
from fastapi.testclient import TestClient


class _DummyChatService:
    def ask(self, *, question: str, top_k: int = 5, cv_id: int | None = None) -> dict:
        return {
            "answer": f"cv_id={cv_id}; q={question}; top_k={top_k}",
            "sources": [],
            "retrieval_count": 0,
            "used_fallback": False,
            "fallback_reason": "",
            "fallback_stage": "",
        }


def test_chat_endpoint_accepts_cv_id(monkeypatch) -> None:
    monkeypatch.setattr(chatbot_endpoint, "get_chat_service", lambda: _DummyChatService())
    client = TestClient(app)
    payload = {"question": "Tu van CV cho toi", "top_k": 3, "cv_id": 7}
    res = client.post("/api/v1/chat/ask", json=payload)

    assert res.status_code == 200
    body = res.json()
    assert "cv_id=7" in body["answer"]


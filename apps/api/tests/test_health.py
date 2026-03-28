from fastapi.testclient import TestClient

from apps.api.app.server import app


client = TestClient(app)


def test_healthz_ok() -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json().get("status") == "ok"

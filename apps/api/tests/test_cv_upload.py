from __future__ import annotations

from apps.api.app.api.v1 import cv as cv_endpoint
from apps.api.app.server import app
from fastapi.testclient import TestClient


def test_cv_upload_txt_success(monkeypatch) -> None:
    monkeypatch.setattr(
        cv_endpoint,
        "extract_cv_info",
        lambda _path: {"cv_id": "mock-cv", "target_role": "Data Analyst", "skills": ["python", "sql"]},
    )
    client = TestClient(app)
    res = client.post(
        "/api/v1/cv/upload?file_name=sample_cv.txt",
        content=b"Data Analyst with Python SQL",
        headers={"content-type": "text/plain"},
    )

    assert res.status_code == 200
    body = res.json()
    assert body["file_name"] == "sample_cv.txt"
    assert body["extracted"]["cv_id"] == "mock-cv"


def test_cv_upload_reject_unsupported_file() -> None:
    client = TestClient(app)
    res = client.post(
        "/api/v1/cv/upload?file_name=sample_cv.exe",
        content=b"binary",
        headers={"content-type": "application/octet-stream"},
    )

    assert res.status_code == 422

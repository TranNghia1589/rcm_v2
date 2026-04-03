from __future__ import annotations

from pathlib import Path

from src.models.graph.query_service import GraphQueryService


def test_recommend_jobs_uses_expected_query_file_and_params(monkeypatch) -> None:
    service = GraphQueryService(neo4j_cfg_path="config/db/neo4j.yaml")
    captured: dict[str, object] = {}

    def _fake_run_query_file(self, file_path: Path, params: dict):
        captured["file_path"] = file_path
        captured["params"] = params
        return [{"job_id": 1}]

    monkeypatch.setattr(GraphQueryService, "_run_query_file", _fake_run_query_file)

    rows = service.recommend_jobs(cv_id="7", limit="4")

    assert rows == [{"job_id": 1}]
    assert str(captured["file_path"]).endswith("database\\neo4j\\queries\\recommend_jobs.cypher")
    assert captured["params"] == {"cv_id": 7, "limit": 4}


def test_skill_gap_and_career_path_params_are_int(monkeypatch) -> None:
    service = GraphQueryService(neo4j_cfg_path="config/db/neo4j.yaml")
    calls: list[dict] = []

    def _fake_run_query_file(self, file_path: Path, params: dict):
        calls.append({"path": str(file_path), "params": params})
        return []

    monkeypatch.setattr(GraphQueryService, "_run_query_file", _fake_run_query_file)

    service.user_skill_gap(cv_id="11", limit="6")
    service.career_path(cv_id="11", limit="2")

    assert calls[0]["params"] == {"cv_id": 11, "limit": 6}
    assert calls[1]["params"] == {"cv_id": 11, "limit": 2}
    assert calls[0]["path"].endswith("database\\neo4j\\queries\\user_skill_gap.cypher")
    assert calls[1]["path"].endswith("database\\neo4j\\queries\\career_path.cypher")

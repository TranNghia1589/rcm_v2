from __future__ import annotations

from src.models.recommendation import orchestrator


def test_hybrid_orchestrator_merges_and_ranks_vector_graph(monkeypatch) -> None:
    cfg = orchestrator.HybridConfig(
        vector_top_k=20,
        graph_top_k=20,
        final_top_k=3,
        weight_vector=0.5,
        weight_graph=0.5,
        enable_llm_explanation=False,
    )
    monkeypatch.setattr(orchestrator.HybridConfig, "from_yaml", staticmethod(lambda _path: cfg))
    monkeypatch.setattr(
        orchestrator,
        "vector_candidates",
        lambda **_kwargs: [
            {
                "job_id": 1,
                "title": "Data Analyst",
                "company_name": "A",
                "location": "HCM",
                "vector_score": 0.9,
                "supporting_chunk_ids": [10],
            },
            {
                "job_id": 2,
                "title": "ML Engineer",
                "company_name": "B",
                "location": "HN",
                "vector_score": 0.1,
                "supporting_chunk_ids": [11],
            },
        ],
    )
    monkeypatch.setattr(
        orchestrator,
        "graph_candidates",
        lambda **_kwargs: [
            {
                "job_id": 2,
                "title": "ML Engineer",
                "company_name": "B",
                "location": "HN",
                "matched_skills": 4,
                "total_required": 5,
                "coverage": 0.8,
                "graph_score": 0.8,
            },
            {
                "job_id": 3,
                "title": "Data Engineer",
                "company_name": "C",
                "location": "DN",
                "matched_skills": 2,
                "total_required": 5,
                "coverage": 0.4,
                "graph_score": 0.4,
            },
        ],
    )

    out = orchestrator.run_hybrid_recommendation(
        question="goi y job data",
        cv_id=1,
        postgres_config_path="unused",
        neo4j_config_path="unused",
        retrieval_config_path="unused",
        embedding_config_path="unused",
        hybrid_config_path="unused",
    )

    ranked_job_ids = [x["job_id"] for x in out["items"]]
    assert ranked_job_ids == [2, 1, 3]
    assert out["warnings"] == []
    assert out["explanation"] == ""


def test_hybrid_orchestrator_keeps_graph_results_when_vector_fails(monkeypatch) -> None:
    cfg = orchestrator.HybridConfig(enable_llm_explanation=False, final_top_k=5)
    monkeypatch.setattr(orchestrator.HybridConfig, "from_yaml", staticmethod(lambda _path: cfg))

    def _raise_vector(**_kwargs):
        raise RuntimeError("vector down")

    monkeypatch.setattr(orchestrator, "vector_candidates", _raise_vector)
    monkeypatch.setattr(
        orchestrator,
        "graph_candidates",
        lambda **_kwargs: [
            {
                "job_id": 9,
                "title": "Analytics Engineer",
                "company_name": "Demo",
                "location": "HCM",
                "matched_skills": 3,
                "total_required": 5,
                "coverage": 0.6,
                "graph_score": 0.7,
            }
        ],
    )

    out = orchestrator.run_hybrid_recommendation(
        question="goi y viec",
        cv_id=77,
        postgres_config_path="unused",
        neo4j_config_path="unused",
        retrieval_config_path="unused",
        embedding_config_path="unused",
        hybrid_config_path="unused",
    )

    assert len(out["items"]) == 1
    assert out["items"][0]["job_id"] == 9
    assert any("vector_candidates_failed" in w for w in out["warnings"])

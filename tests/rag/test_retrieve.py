from __future__ import annotations

from dataclasses import dataclass

from src.models.rag import retrieve as rag_retrieve


def test_preview_truncates_with_ellipsis() -> None:
    text = "a" * 20
    assert rag_retrieve._preview(text, 10) == "aaaaaaa..."


def test_retrieve_chunks_uses_query_embedding_and_override_top_k(monkeypatch) -> None:
    @dataclass
    class _DummyPgCfg:
        pass

    class _DummyClient:
        def __init__(self, _cfg) -> None:
            self.cfg = _cfg

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    captured: dict[str, object] = {}

    class _DummyStore:
        def __init__(self, _client) -> None:
            self.client = _client

        def search_chunks(self, *, query_embedding, top_k, source_type):
            captured["query_embedding"] = query_embedding
            captured["top_k"] = top_k
            captured["source_type"] = source_type
            return [{"chunk_id": 1, "document_id": 10, "title": "T", "distance": 0.1, "content": "x"}]

    class _ExplodingEmbedder:
        def embed_text(self, _question: str):
            raise AssertionError("embed_text must not be called when query_embedding is provided")

    monkeypatch.setattr(rag_retrieve.PostgresConfig, "from_yaml", staticmethod(lambda _path: _DummyPgCfg()))
    monkeypatch.setattr(
        rag_retrieve.RetrievalConfig,
        "from_yaml",
        staticmethod(lambda _path: rag_retrieve.RetrievalConfig(top_k=9, source_type="job")),
    )
    monkeypatch.setattr(rag_retrieve, "PostgresClient", _DummyClient)
    monkeypatch.setattr(rag_retrieve, "PgVectorStore", _DummyStore)

    out = rag_retrieve.retrieve_chunks(
        question="Data engineer role",
        postgres_config_path="unused.yaml",
        retrieval_config_path="unused.yaml",
        embedding_config_path="unused.yaml",
        top_k_override=3,
        embedder=_ExplodingEmbedder(),
        query_embedding=[0.11, 0.22, 0.33],
    )

    assert len(out) == 1
    assert captured["query_embedding"] == [0.11, 0.22, 0.33]
    assert captured["top_k"] == 3
    assert captured["source_type"] == "job"

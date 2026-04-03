from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.models.rag.embed import create_embedder_from_yaml


BASE_DIR = Path(__file__).resolve().parents[3]
EMBEDDING_CFG = BASE_DIR / "config" / "model" / "embedding.yaml"


class EmbedRequest(BaseModel):
    texts: list[str] = Field(default_factory=list)


class EmbedResponse(BaseModel):
    vectors: list[list[float]]
    dim: int
    model: str


@dataclass
class Runtime:
    embedder: object | None = None
    model_name: str = "unknown"


runtime = Runtime()
app = FastAPI(title="Embedding Service", version="0.1.0")


@app.on_event("startup")
def _startup() -> None:
    runtime.embedder = create_embedder_from_yaml(EMBEDDING_CFG)
    runtime.model_name = str(getattr(runtime.embedder, "model_name", "unknown"))


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/embed", response_model=EmbedResponse)
def embed(payload: EmbedRequest) -> EmbedResponse:
    texts = [str(x or "").strip() for x in payload.texts]
    if not texts:
        return EmbedResponse(vectors=[], dim=0, model=runtime.model_name)
    assert runtime.embedder is not None
    vectors = runtime.embedder.embed_texts(texts)
    dim = len(vectors[0]) if vectors else 0
    return EmbedResponse(vectors=vectors, dim=dim, model=runtime.model_name)

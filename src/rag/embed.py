from __future__ import annotations

import math
import re
import hashlib
import shutil
import os
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path

import yaml

try:
    import torch
    from transformers import AutoModel, AutoTokenizer

    _HAS_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    AutoModel = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    _HAS_TRANSFORMERS = False


@dataclass
class HashEmbeddingConfig:
    dimension: int = 768
    normalize: bool = True


class HashEmbedder:
    """Deterministic local embedder with no external dependency.

    This is a bootstrap embedder so ingestion/retrieval can run immediately.
    You can later replace it with a stronger embedding model provider.
    """

    def __init__(self, config: HashEmbeddingConfig | None = None) -> None:
        self.config = config or HashEmbeddingConfig()
        if self.config.dimension <= 0:
            raise ValueError("Embedding dimension must be > 0.")

    def _tokenize(self, text: str) -> list[str]:
        low = (text or "").lower()
        cleaned = re.sub(r"[^a-z0-9\u00C0-\u024F\s]+", " ", low)
        return [tok for tok in cleaned.split() if tok]

    def _hash_index_sign(self, token: str) -> tuple[int, float]:
        h = hashlib.sha256(token.encode("utf-8")).hexdigest()
        idx = int(h[:8], 16) % self.config.dimension
        sign = 1.0 if int(h[8:10], 16) % 2 == 0 else -1.0
        return idx, sign

    def embed_text(self, text: str) -> list[float]:
        vec = [0.0] * self.config.dimension
        tokens = self._tokenize(text)
        if not tokens:
            return vec

        for tok in tokens:
            idx, sign = self._hash_index_sign(tok)
            vec[idx] += sign

        if self.config.normalize:
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
        return vec

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


@dataclass
class PhoBERTEmbeddingConfig:
    model_name: str = "vinai/phobert-base"
    cache_dir: str = "hf_cache"
    local_files_only: bool = True
    normalize: bool = True
    max_length: int = 256
    batch_size: int = 16
    device: str = "auto"  # auto/cpu/cuda
    use_safetensors: bool = True
    hf_token: str | None = None


class PhoBERTEmbedder:
    def __init__(self, config: PhoBERTEmbeddingConfig | None = None) -> None:
        if not _HAS_TRANSFORMERS:
            raise RuntimeError("transformers/torch are required for PhoBERT embedding.")
        self.config = config or PhoBERTEmbeddingConfig()
        self.model_name = self.config.model_name
        self._device = self._resolve_device(self.config.device)

        kwargs = {
            "cache_dir": self.config.cache_dir,
            "local_files_only": self.config.local_files_only,
            "token": self.config.hf_token,
            "use_safetensors": self.config.use_safetensors,
        }

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, **kwargs)
            self.model = AutoModel.from_pretrained(self.config.model_name, **kwargs)
        except Exception:
            merged_local_dir = self._build_merged_local_snapshot_dir()
            if not merged_local_dir:
                raise
            self.tokenizer = AutoTokenizer.from_pretrained(
                merged_local_dir,
                local_files_only=True,
                use_safetensors=self.config.use_safetensors,
            )
            self.model = AutoModel.from_pretrained(
                merged_local_dir,
                local_files_only=True,
                use_safetensors=self.config.use_safetensors,
            )
        self.model.eval()
        self.model.to(self._device)

    def _build_merged_local_snapshot_dir(self) -> str | None:
        base = Path(self.config.cache_dir) / "models--vinai--phobert-base" / "snapshots"
        if not base.exists():
            return None

        snapshots = [p for p in base.iterdir() if p.is_dir()]
        if not snapshots:
            return None

        cfg_dir = None
        weight_dir = None
        for p in snapshots:
            has_cfg = (p / "config.json").exists() and (
                (p / "tokenizer.json").exists() or (p / "vocab.txt").exists()
            )
            has_weight = (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()
            if has_cfg and cfg_dir is None:
                cfg_dir = p
            if has_weight and weight_dir is None:
                weight_dir = p

        if cfg_dir is None and weight_dir is None:
            return None
        if cfg_dir is None:
            cfg_dir = weight_dir
        if weight_dir is None:
            weight_dir = cfg_dir
        if cfg_dir is None or weight_dir is None:
            return None

        merged = Path(self.config.cache_dir) / "models--vinai--phobert-base" / "merged_local"
        merged.mkdir(parents=True, exist_ok=True)

        for fname in ["config.json", "tokenizer.json", "vocab.txt", "bpe.codes", "special_tokens_map.json"]:
            src = cfg_dir / fname
            dst = merged / fname
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

        for fname in ["model.safetensors", "pytorch_model.bin"]:
            src = weight_dir / fname
            dst = merged / fname
            if not src.exists():
                continue
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src, dst)
            except FileExistsError:
                # Another process may have created it in parallel.
                continue
            except Exception:
                if not dst.exists():
                    shutil.copy2(src, dst)

        if not (merged / "config.json").exists():
            return None
        if not ((merged / "model.safetensors").exists() or (merged / "pytorch_model.bin").exists()):
            return None
        return str(merged)

    def _resolve_device(self, device: str) -> str:
        if device == "cpu":
            return "cpu"
        if device == "cuda":
            return "cuda"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _normalize(self, vec: list[float]) -> list[float]:
        if not self.config.normalize:
            return vec
        norm = math.sqrt(sum(v * v for v in vec))
        if norm <= 0:
            return vec
        return [v / norm for v in vec]

    def _mean_pool(self, last_hidden_state: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        bs = max(1, int(self.config.batch_size))

        with torch.no_grad():
            for i in range(0, len(texts), bs):
                batch = texts[i : i + bs]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self._device) for k, v in encoded.items()}
                model_out = self.model(**encoded)
                pooled = self._mean_pool(model_out.last_hidden_state, encoded["attention_mask"])
                vecs = pooled.detach().cpu().tolist()
                out.extend([self._normalize([float(x) for x in v]) for v in vecs])
        return out


def _create_embedder_from_yaml_uncached(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    provider = str(raw.get("provider", "phobert")).lower().strip()

    if provider == "hash":
        cfg = HashEmbeddingConfig(
            dimension=int(raw.get("dimension", 768)),
            normalize=bool(raw.get("normalize", True)),
        )
        emb = HashEmbedder(cfg)
        emb.model_name = "hash-embedder-768"
        return emb

    hf_token_env_name = str(raw.get("hf_token_env", "HF_TOKEN")).strip()
    hf_token = None
    if hf_token_env_name:
        try:
            import os

            value = os.getenv(hf_token_env_name, "").strip()
            hf_token = value or None
        except Exception:
            hf_token = None

    cfg = PhoBERTEmbeddingConfig(
        model_name=str(raw.get("model_name", "vinai/phobert-base")),
        cache_dir=str(raw.get("cache_dir", "hf_cache")),
        local_files_only=bool(raw.get("local_files_only", True)),
        normalize=bool(raw.get("normalize", True)),
        max_length=int(raw.get("max_length", 256)),
        batch_size=int(raw.get("batch_size", 16)),
        device=str(raw.get("device", "auto")),
        use_safetensors=bool(raw.get("use_safetensors", True)),
        hf_token=hf_token,
    )
    return PhoBERTEmbedder(cfg)


@lru_cache(maxsize=8)
def _create_embedder_cached(path: str) -> object:
    return _create_embedder_from_yaml_uncached(path)


def create_embedder_from_yaml(path: str | Path, *, use_cache: bool = True):
    normalized_path = str(Path(path).resolve())
    if use_cache:
        return _create_embedder_cached(normalized_path)
    return _create_embedder_from_yaml_uncached(normalized_path)


def clear_embedder_cache() -> None:
    _create_embedder_cached.cache_clear()

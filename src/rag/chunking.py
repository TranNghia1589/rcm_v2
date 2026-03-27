from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ChunkingConfig:
    max_chars: int = 900
    overlap_chars: int = 150
    min_chunk_chars: int = 120

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ChunkingConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(
            max_chars=int(raw.get("max_chars", 900)),
            overlap_chars=int(raw.get("overlap_chars", 150)),
            min_chunk_chars=int(raw.get("min_chunk_chars", 120)),
        )


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def estimate_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text.split()))


def split_text(text: str, config: ChunkingConfig) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []

    if len(text) <= config.max_chars:
        return [text]

    chunks: list[str] = []
    step = max(1, config.max_chars - config.overlap_chars)
    start = 0
    n = len(text)

    while start < n:
        end = min(n, start + config.max_chars)
        chunk = text[start:end]

        if end < n:
            last_space = chunk.rfind(" ")
            if last_space > int(config.max_chars * 0.6):
                end = start + last_space
                chunk = text[start:end]

        chunk = chunk.strip()
        if len(chunk) >= config.min_chunk_chars or not chunks:
            chunks.append(chunk)
        start += step

    return chunks

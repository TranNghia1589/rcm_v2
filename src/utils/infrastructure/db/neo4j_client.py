from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover
    GraphDatabase = None  # type: ignore

_DOTENV_LOADED = False


def _load_dotenv_once() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    if load_dotenv is not None:
        load_dotenv()
    _DOTENV_LOADED = True


def _env_str(*keys: str) -> str | None:
    for key in keys:
        val = os.getenv(key)
        if val is None:
            continue
        txt = val.strip()
        if txt:
            return txt
    return None


def _parse_bool(value: str) -> bool | None:
    txt = value.strip().lower()
    if txt in {"1", "true", "yes", "on"}:
        return True
    if txt in {"0", "false", "no", "off"}:
        return False
    return None


@dataclass
class Neo4jConfig:
    uri: str
    user: str
    password: str
    database: str = "neo4j"
    encrypted: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Neo4jConfig":
        _load_dotenv_once()
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        uri = _env_str("NEO4J_URI") or str(raw.get("uri", "bolt://localhost:7687"))
        user = _env_str("NEO4J_USER") or str(raw.get("user", "neo4j"))
        password = _env_str("NEO4J_PASSWORD") or str(raw.get("password", ""))
        database = _env_str("NEO4J_DATABASE") or str(raw.get("database", "neo4j"))
        encrypted_raw = _env_str("NEO4J_ENCRYPTED")
        if encrypted_raw is None:
            encrypted = bool(raw.get("encrypted", False))
        else:
            parsed = _parse_bool(encrypted_raw)
            encrypted = bool(parsed) if parsed is not None else bool(raw.get("encrypted", False))
        return cls(
            uri=uri,
            user=user,
            password=password,
            database=database,
            encrypted=encrypted,
        )


class Neo4jClient:
    def __init__(self, config: Neo4jConfig) -> None:
        self.config = config
        self._driver = None

    def connect(self) -> None:
        if GraphDatabase is None:
            raise RuntimeError("neo4j driver missing. Run: pip install neo4j>=5.23")
        if self._driver is not None:
            return
        self._driver = GraphDatabase.driver(
            self.config.uri,
            auth=(self.config.user, self.config.password),
            encrypted=self.config.encrypted,
        )

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> "Neo4jClient":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def run(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if self._driver is None:
            raise RuntimeError("Neo4j driver is not connected.")
        with self._driver.session(database=self.config.database) as session:
            result = session.run(query, params or {})
            return [dict(record) for record in result]

    def execute(self, query: str, params: dict[str, Any] | None = None) -> None:
        _ = self.run(query, params)

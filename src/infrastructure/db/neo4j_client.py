from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover
    GraphDatabase = None  # type: ignore


@dataclass
class Neo4jConfig:
    uri: str
    user: str
    password: str
    database: str = "neo4j"
    encrypted: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Neo4jConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(
            uri=str(raw.get("uri", "bolt://localhost:7687")),
            user=str(raw.get("user", "neo4j")),
            password=str(raw.get("password", "")),
            database=str(raw.get("database", "neo4j")),
            encrypted=bool(raw.get("encrypted", False)),
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

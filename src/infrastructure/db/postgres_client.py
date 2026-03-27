from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

try:
    import psycopg

    _DRIVER = "psycopg3"
except ImportError:  # pragma: no cover
    psycopg = None  # type: ignore
    _DRIVER = "missing"


@dataclass
class PostgresConfig:
    host: str
    port: int
    database: str
    user: str
    password: str
    sslmode: str = "prefer"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PostgresConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(
            host=str(raw.get("host", "localhost")),
            port=int(raw.get("port", 5432)),
            database=str(raw.get("database", "")),
            user=str(raw.get("user", "postgres")),
            password=str(raw.get("password", "")),
            sslmode=str(raw.get("sslmode", "prefer")),
        )


class PostgresClient:
    def __init__(self, config: PostgresConfig) -> None:
        self.config = config
        self._conn = None

    def connect(self) -> None:
        if _DRIVER == "missing":
            raise RuntimeError(
                "psycopg is not installed. Run: pip install psycopg[binary]>=3.1"
            )
        if self._conn is not None:
            return
        self._conn = psycopg.connect(  # type: ignore[union-attr]
            host=self.config.host,
            port=self.config.port,
            dbname=self.config.database,
            user=self.config.user,
            password=self.config.password,
            sslmode=self.config.sslmode,
            autocommit=False,
        )

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "PostgresClient":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._conn is None:
            return
        if exc:
            self._conn.rollback()
        else:
            self._conn.commit()
        self.close()

    def commit(self) -> None:
        if self._conn is not None:
            self._conn.commit()

    def rollback(self) -> None:
        if self._conn is not None:
            self._conn.rollback()

    def execute(self, query: str, params: tuple[Any, ...] | None = None) -> None:
        if self._conn is None:
            raise RuntimeError("Connection is not open.")
        with self._conn.cursor() as cur:
            cur.execute(query, params)

    def fetch_one(self, query: str, params: tuple[Any, ...] | None = None) -> Any:
        if self._conn is None:
            raise RuntimeError("Connection is not open.")
        with self._conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchone()

    def fetch_all(self, query: str, params: tuple[Any, ...] | None = None) -> list[Any]:
        if self._conn is None:
            raise RuntimeError("Connection is not open.")
        with self._conn.cursor() as cur:
            cur.execute(query, params)
            return list(cur.fetchall())

    def executemany(self, query: str, seq: Iterable[tuple[Any, ...]]) -> None:
        if self._conn is None:
            raise RuntimeError("Connection is not open.")
        with self._conn.cursor() as cur:
            cur.executemany(query, seq)

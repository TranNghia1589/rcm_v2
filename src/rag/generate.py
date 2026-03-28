from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import PurePosixPath
from time import sleep
from urllib.parse import urlparse, urlunparse

import requests


@dataclass
class OllamaConfig:
    url: str = "http://localhost:11434/api/generate"
    model: str = "llama3:latest"
    timeout_sec: int = 120
    temperature: float = 0.2
    retries: int = 1
    retry_backoff_sec: float = 0.35
    keep_alive: str = "10m"


class OllamaGenerator:
    def __init__(self, config: OllamaConfig | None = None) -> None:
        self.config = config or OllamaConfig()
        self.config.url = self._normalize_url(self.config.url)
        self._session = _shared_session()

    def _normalize_url(self, url: str) -> str:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        if hostname.lower() != "localhost":
            return url
        netloc = parsed.netloc.replace("localhost", "127.0.0.1", 1)
        return urlunparse(parsed._replace(netloc=netloc))

    def _tags_url(self) -> str:
        marker = "/api/generate"
        if self.config.url.endswith(marker):
            return self.config.url[: -len(marker)] + "/api/tags"
        return str(PurePosixPath(self.config.url).parent / "tags")

    def _available_models(self, timeout: int) -> list[str]:
        try:
            res = self._session.get(self._tags_url(), timeout=min(timeout, 15))
            res.raise_for_status()
            data = res.json() or {}
        except Exception:
            return []
        models = data.get("models", []) if isinstance(data, dict) else []
        names: list[str] = []
        for item in models:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("model") or "").strip()
            if name:
                names.append(name)
        return names

    def _resolve_model(self, preferred_model: str, timeout: int) -> str:
        available = self._available_models(timeout)
        if not available:
            return preferred_model
        if preferred_model in available:
            return preferred_model

        preferred_candidates = [
            "llama3.1:8b",
            "llama3.2:1b",
            "gemma3:1b",
            "llama3:latest",
        ]
        for candidate in preferred_candidates:
            if candidate in available:
                return candidate
        return available[0]

    def generate(
        self,
        prompt: str,
        *,
        timeout_sec: int | None = None,
        temperature: float | None = None,
        retries: int | None = None,
    ) -> str:
        attempts = max(1, int(retries if retries is not None else self.config.retries) + 1)
        timeout = int(timeout_sec if timeout_sec is not None else self.config.timeout_sec)
        last_error: Exception | None = None
        active_model = self._resolve_model(self.config.model, timeout)

        for attempt in range(attempts):
            try:
                payload = {
                    "model": active_model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": self.config.keep_alive,
                    "options": {
                        "temperature": float(
                            temperature if temperature is not None else self.config.temperature
                        )
                    },
                }
                res = self._session.post(self.config.url, json=payload, timeout=timeout)
                if res.status_code == 404:
                    fallback_model = self._resolve_model(active_model, timeout)
                    if fallback_model != active_model:
                        active_model = fallback_model
                        self.config.model = fallback_model
                        continue
                res.raise_for_status()
                data = res.json()
                self.config.model = active_model
                return str(data.get("response", "") or "").strip()
            except Exception as exc:  # pragma: no cover
                last_error = exc
                if attempt + 1 < attempts:
                    sleep(self.config.retry_backoff_sec * (2**attempt))
        if last_error is not None:
            raise last_error
        raise RuntimeError("Ollama generation failed without explicit error.")


@lru_cache(maxsize=1)
def _shared_session() -> requests.Session:
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

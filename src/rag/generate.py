from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from time import sleep

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
        self._session = _shared_session()

    def generate(
        self,
        prompt: str,
        *,
        timeout_sec: int | None = None,
        temperature: float | None = None,
        retries: int | None = None,
    ) -> str:
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.config.keep_alive,
            "options": {"temperature": float(temperature if temperature is not None else self.config.temperature)},
        }
        attempts = max(1, int(retries if retries is not None else self.config.retries) + 1)
        timeout = int(timeout_sec if timeout_sec is not None else self.config.timeout_sec)
        last_error: Exception | None = None

        for attempt in range(attempts):
            try:
                res = self._session.post(self.config.url, json=payload, timeout=timeout)
                res.raise_for_status()
                data = res.json()
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

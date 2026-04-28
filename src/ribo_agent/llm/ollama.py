"""Ollama HTTP client for local development.

Ollama exposes an OpenAI-ish API at http://localhost:11434 by default.
We use the native /api/generate endpoint (not the /v1/chat/completions
compat layer) because it returns token counts directly — cheaper to
keep accurate cost metrics.

Assumes `ollama serve` is running and the named model has been pulled:
    brew install ollama
    ollama serve &
    ollama pull qwen2.5:7b-instruct
"""
from __future__ import annotations

import time

import httpx

from .base import LLMResponse


class OllamaClient:
    """Minimal Ollama client. Thread-safe for read-only use."""

    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        base_url: str = "http://localhost:11434",
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout_s)

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if stop:
            payload["options"]["stop"] = stop

        t0 = time.perf_counter()
        resp = self._client.post("/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
        latency_ms = (time.perf_counter() - t0) * 1000

        return LLMResponse(
            text=data.get("response", ""),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            latency_ms=latency_ms,
            model=self.model,
            backend="ollama",
        )

    def health(self) -> bool:
        try:
            resp = self._client.get("/api/tags", timeout=3.0)
            resp.raise_for_status()
            tags = {m["name"] for m in resp.json().get("models", [])}
            # match either "qwen2.5:7b-instruct" or "qwen2.5:7b-instruct-q4_0"
            return any(t.startswith(self.model.split(":")[0]) for t in tags)
        except Exception:
            return False

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

"""Client factory — translates config dicts into `LLMClient` instances.

This is the single place where backend selection happens. Every agent
takes an `LLMClient`, never instantiates one directly, so the agent
layer stays backend-agnostic.
"""
from __future__ import annotations

from typing import Any

from .base import LLMClient


def make_client(cfg: dict[str, Any]) -> LLMClient:
    """Build an `LLMClient` from a config dict.

    Expected schema (matches `configs/*.yaml` under the `llm:` key):

        backend: ollama|azureml
        model: str
        base_url: str               # ollama only
        endpoint_url: str           # azureml only
        timeout_s: float            # optional
    """
    backend = cfg.get("backend", "ollama").lower()

    if backend == "ollama":
        from .ollama import OllamaClient

        return OllamaClient(
            model=cfg.get("model", "qwen2.5:7b-instruct"),
            base_url=cfg.get("base_url", "http://localhost:11434"),
            timeout_s=cfg.get("timeout_s", 120.0),
        )

    if backend == "azureml":
        from .azureml import AzureMLClient

        return AzureMLClient(
            endpoint_url=cfg.get("endpoint_url"),
            api_key=cfg.get("api_key"),
            deployment=cfg.get("deployment"),
            model=cfg.get("model", "qwen2.5-7b-instruct"),
            timeout_s=cfg.get("timeout_s", 120.0),
        )

    raise ValueError(f"unknown llm.backend: {backend!r}")

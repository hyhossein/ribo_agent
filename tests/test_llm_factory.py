"""Tests for the LLM client factory and the Ollama client shape.

These tests do NOT require an Ollama server to be running. Network calls
are avoided by checking health() returns False gracefully, and by
patching the factory.
"""
from __future__ import annotations

import pytest

from ribo_agent.llm import LLMClient, OllamaClient, make_client


def test_factory_builds_ollama_by_default() -> None:
    c = make_client({"backend": "ollama"})
    assert isinstance(c, OllamaClient)
    assert c.model == "qwen2.5:7b-instruct"
    assert c.base_url == "http://localhost:11434"


def test_factory_honours_config_overrides() -> None:
    c = make_client({
        "backend": "ollama",
        "model": "llama3.1:8b",
        "base_url": "http://example.test:9999",
    })
    assert isinstance(c, OllamaClient)
    assert c.model == "llama3.1:8b"
    assert c.base_url == "http://example.test:9999"


def test_factory_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unknown llm.backend"):
        make_client({"backend": "totally-made-up"})


def test_factory_azureml_requires_credentials(monkeypatch) -> None:
    # strip any local env that would accidentally let the client construct
    monkeypatch.delenv("AZUREML_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AZUREML_ENDPOINT_KEY", raising=False)
    with pytest.raises(RuntimeError, match="AZUREML_ENDPOINT"):
        make_client({"backend": "azureml"})


def test_ollama_health_returns_bool_when_server_unreachable() -> None:
    # point at a port nothing is listening on; should return False,
    # never raise — health() must be safe to call from CI.
    c = OllamaClient(base_url="http://127.0.0.1:1")
    assert c.health() is False


def test_ollama_client_satisfies_llmclient_protocol() -> None:
    # Protocol is runtime_checkable-compatible via duck typing — just
    # assert the methods are there with the right names.
    c = OllamaClient()
    assert callable(getattr(c, "complete", None))
    assert callable(getattr(c, "health", None))

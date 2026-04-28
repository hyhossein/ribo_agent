"""LLM client protocol.

Every backend (Ollama for local dev, Azure ML Managed Endpoint for
production) must expose the same `complete()` signature so the agents
themselves never import a specific backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResponse:
    """What a client returns for a single completion.

    Keeping `prompt_tokens` and `completion_tokens` first-class means cost
    and latency dashboards can be built without the agent layer knowing
    which backend ran.
    """

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None
    model: str | None = None
    backend: str | None = None


class LLMClient(Protocol):
    """Minimal surface area every backend implements.

    We deliberately keep this tiny — two methods. Anything richer
    (streaming, tool use, structured output) belongs in a subclass so the
    core contract stays testable.
    """

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        """Return a single completion for `prompt`."""
        ...

    def health(self) -> bool:
        """Return True if the backend is reachable and the model is loaded."""
        ...

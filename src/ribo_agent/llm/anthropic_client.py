"""Anthropic Claude LLM client."""
from __future__ import annotations
import os, time, anthropic
from .base import LLMResponse

def _resolve_env(val):
    if val and val.startswith("${") and val.endswith("}"):
        return os.environ.get(val[2:-1], "")
    return val

class AnthropicClient:
    def __init__(self, cfg):
        self.model = cfg.get("model", "claude-sonnet-4-20250514")
        api_key = _resolve_env(cfg.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", ""))
        if not api_key:
            raise ValueError("Set ANTHROPIC_API_KEY or llm.api_key in config.")
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, prompt, *, temperature=0.0, max_tokens=256, stop=None):
        t0 = time.perf_counter()
        resp = self._client.messages.create(
            model=self.model, max_tokens=max_tokens, temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        text = resp.content[0].text if resp.content else ""
        return LLMResponse(
            text=text, prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens, latency_ms=elapsed_ms,
            model=self.model, backend="anthropic",
        )

    def health(self):
        return True

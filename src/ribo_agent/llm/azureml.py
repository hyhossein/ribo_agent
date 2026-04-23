"""Azure ML Managed Online Endpoint client.

When we promote the chosen model to Azure ML, the deployment exposes an
HTTPS endpoint that speaks the same request shape as Ollama for our
purposes (prompt in, text + tokens out). This stub keeps the interface
fixed so only `configs/*.yaml` and this file change at promotion time —
no agent-level refactor.

Deployment recipe (to be executed in a later release):
    1. Register the model to the Azure ML workspace
    2. Deploy to a Managed Online Endpoint (e.g. Qwen2.5-7B via vLLM)
    3. Set the two env vars below to what `az ml online-endpoint show`
       returns

Env vars:
    AZUREML_ENDPOINT_URL   — https://...inference.ml.azure.com/score
    AZUREML_ENDPOINT_KEY   — primary or secondary key from the endpoint
"""
from __future__ import annotations

import os
import time

import httpx

from .base import LLMResponse


class AzureMLClient:
    """Client for an Azure ML Managed Online Endpoint.

    STATUS: stub. Unit-tested against the request-shape contract only.
    Replace the payload shape once the deployment template is finalised.
    """

    def __init__(
        self,
        endpoint_url: str | None = None,
        api_key: str | None = None,
        deployment: str | None = None,
        model: str = "qwen2.5-7b-instruct",
        timeout_s: float = 120.0,
    ) -> None:
        self.endpoint_url = endpoint_url or os.environ.get("AZUREML_ENDPOINT_URL", "")
        self.api_key = api_key or os.environ.get("AZUREML_ENDPOINT_KEY", "")
        self.deployment = deployment
        self.model = model

        if not self.endpoint_url or not self.api_key:
            raise RuntimeError(
                "AzureMLClient requires AZUREML_ENDPOINT_URL and "
                "AZUREML_ENDPOINT_KEY (either as constructor args or env vars)."
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.deployment:
            headers["azureml-model-deployment"] = self.deployment

        self._client = httpx.Client(
            base_url=self.endpoint_url,
            headers=headers,
            timeout=timeout_s,
        )

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        # Shape is vLLM-style; final deployment may use a different schema.
        # Keep this isolated so changes stay local to this file.
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            payload["stop"] = stop

        t0 = time.perf_counter()
        resp = self._client.post("", json=payload)
        resp.raise_for_status()
        data = resp.json()
        latency_ms = (time.perf_counter() - t0) * 1000

        # vLLM returns choices[0].text; tolerate both common shapes
        text = (
            data.get("text")
            or (data.get("choices") or [{}])[0].get("text")
            or ""
        )
        usage = data.get("usage", {}) or {}
        return LLMResponse(
            text=text,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_ms=latency_ms,
            model=self.model,
            backend="azureml",
        )

    def health(self) -> bool:
        # Azure ML endpoints do not expose a standard health sub-path; a
        # minimal completion is the cheapest deterministic probe.
        try:
            self.complete("ping", max_tokens=1)
            return True
        except Exception:
            return False

    def close(self) -> None:
        self._client.close()

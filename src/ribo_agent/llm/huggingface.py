"""HuggingFace Transformers client for fine-tuned models.

Loads a model from a local directory containing safetensors weights
(e.g. the output of `transformers.Trainer.save_model()`).

Usage:
    client = HuggingFaceClient(model_path="path/to/finetuned-qwen")
    resp = client.complete("What is ...")
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import LLMResponse


class HuggingFaceClient:
    """Load a fine-tuned model from safetensors and run local inference."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str | None = None,
        torch_dtype: str = "auto",
        timeout_s: float = 300.0,
    ) -> None:
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.timeout_s = timeout_s

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        # resolve dtype
        _dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        resolved_dtype = _dtype_map.get(torch_dtype, "auto")

        print(f"[hf] loading model from {self.model_path} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=resolved_dtype,
            device_map=self.device if self.device != "mps" else None,
            trust_remote_code=True,
        )
        if self.device == "mps":
            self.model = self.model.to("mps")

        self.model.eval()
        print(f"[hf] model loaded — {sum(p.numel() for p in self.model.parameters()) / 1e6:.0f}M params")

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[-1]

        gen_kwargs: dict = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9

        # stop sequences → eos token ids (best effort)
        extra_eos: list[int] = []
        if stop:
            for s in stop:
                ids = self.tokenizer.encode(s, add_special_tokens=False)
                if ids:
                    extra_eos.append(ids[0])
        if extra_eos:
            gen_kwargs["eos_token_id"] = [
                self.tokenizer.eos_token_id, *extra_eos
            ]

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000

        # decode only the NEW tokens
        new_ids = output_ids[0][prompt_len:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)

        # trim at stop sequences
        if stop:
            for s in stop:
                idx = text.find(s)
                if idx != -1:
                    text = text[:idx]

        return LLMResponse(
            text=text.strip(),
            prompt_tokens=prompt_len,
            completion_tokens=len(new_ids),
            latency_ms=latency_ms,
            model=self.model_path,
            backend="huggingface",
        )

    def health(self) -> bool:
        return self.model is not None

    def close(self) -> None:
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

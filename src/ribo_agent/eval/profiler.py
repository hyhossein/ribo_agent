"""Cost analysis and time profiling for LLM inference.

Tracks per-question and aggregate:
  - Token usage (prompt + completion)
  - Latency (wall clock, p50/p90/p99)
  - Estimated cost (configurable rates)
  - Retrieval time vs generation time breakdown
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class QuestionProfile:
    """Timing and cost for a single question."""
    qid: str
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    retrieval_method: str = ""
    model: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunProfile:
    """Aggregate profiling for an entire eval run."""
    questions: list[QuestionProfile] = field(default_factory=list)
    total_questions: int = 0
    total_time_s: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_retrieval_ms: float = 0.0
    avg_generation_ms: float = 0.0
    avg_total_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p90_ms: float = 0.0
    latency_p99_ms: float = 0.0
    tokens_per_second: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        # Don't dump individual questions in the summary
        d.pop("questions", None)
        return d


# Default cost rates (USD per 1K tokens) — adjust for your model
DEFAULT_RATES = {
    "ollama": {"prompt": 0.0, "completion": 0.0},      # local = free
    "huggingface": {"prompt": 0.0, "completion": 0.0},  # local = free
    "azureml": {"prompt": 0.0003, "completion": 0.0006},
    "openai_gpt4": {"prompt": 0.03, "completion": 0.06},
}


class CostTracker:
    """Accumulates cost and timing data across questions."""

    def __init__(
        self,
        backend: str = "ollama",
        model: str = "unknown",
        cost_rates: dict | None = None,
    ) -> None:
        self.backend = backend
        self.model = model
        self.rates = (cost_rates or DEFAULT_RATES).get(
            backend, {"prompt": 0.0, "completion": 0.0}
        )
        self._profiles: list[QuestionProfile] = []
        self._run_start: float | None = None

    def start_run(self) -> None:
        self._run_start = time.perf_counter()
        self._profiles = []

    def record(
        self,
        qid: str,
        *,
        retrieval_ms: float = 0.0,
        generation_ms: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        retrieval_method: str = "",
    ) -> QuestionProfile:
        total_tokens = prompt_tokens + completion_tokens
        cost = (
            (prompt_tokens / 1000) * self.rates["prompt"]
            + (completion_tokens / 1000) * self.rates["completion"]
        )
        profile = QuestionProfile(
            qid=qid,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            total_ms=retrieval_ms + generation_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=round(cost, 6),
            retrieval_method=retrieval_method,
            model=self.model,
        )
        self._profiles.append(profile)
        return profile

    def _percentile(self, values: list[float], p: float) -> float:
        if not values:
            return 0.0
        values = sorted(values)
        k = max(0, min(len(values) - 1, int(round((len(values) - 1) * p))))
        return values[k]

    def summarize(self) -> RunProfile:
        n = len(self._profiles)
        if n == 0:
            return RunProfile()

        total_time = (
            time.perf_counter() - self._run_start
            if self._run_start else 0.0
        )
        latencies = [p.total_ms for p in self._profiles]
        total_tokens = sum(p.total_tokens for p in self._profiles)

        return RunProfile(
            questions=self._profiles,
            total_questions=n,
            total_time_s=round(total_time, 2),
            total_prompt_tokens=sum(p.prompt_tokens for p in self._profiles),
            total_completion_tokens=sum(p.completion_tokens for p in self._profiles),
            total_tokens=total_tokens,
            total_cost_usd=round(sum(p.estimated_cost_usd for p in self._profiles), 6),
            avg_retrieval_ms=round(sum(p.retrieval_ms for p in self._profiles) / n, 1),
            avg_generation_ms=round(sum(p.generation_ms for p in self._profiles) / n, 1),
            avg_total_ms=round(sum(p.total_ms for p in self._profiles) / n, 1),
            latency_p50_ms=round(self._percentile(latencies, 0.5), 1),
            latency_p90_ms=round(self._percentile(latencies, 0.9), 1),
            latency_p99_ms=round(self._percentile(latencies, 0.99), 1),
            tokens_per_second=round(total_tokens / max(total_time, 0.001), 1),
        )

    def format_summary(self) -> str:
        s = self.summarize()
        lines = [
            "## Cost & Performance Profile",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Total questions | {s.total_questions} |",
            f"| Total time | {s.total_time_s:.1f}s |",
            f"| Avg retrieval | {s.avg_retrieval_ms:.0f} ms |",
            f"| Avg generation | {s.avg_generation_ms:.0f} ms |",
            f"| Avg total | {s.avg_total_ms:.0f} ms |",
            f"| Latency p50 | {s.latency_p50_ms:.0f} ms |",
            f"| Latency p90 | {s.latency_p90_ms:.0f} ms |",
            f"| Latency p99 | {s.latency_p99_ms:.0f} ms |",
            f"| Total tokens | {s.total_tokens:,} |",
            f"| Tokens/sec | {s.tokens_per_second:.1f} |",
            f"| Estimated cost | ${s.total_cost_usd:.4f} |",
            f"| Model | {self.model} |",
            f"| Backend | {self.backend} |",
        ]
        return "\n".join(lines)

"""Agent protocol: given an MCQ, predict one of A/B/C/D."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..parsers.schema import MCQ


@dataclass
class Prediction:
    qid: str
    predicted: str | None         # A/B/C/D or None if the agent refused/failed
    correct: str                  # ground-truth letter
    is_correct: bool
    raw_response: str = ""
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    citations: list[dict] = field(default_factory=list)   # [{source, citation, score, snippet}]
    extras: dict = field(default_factory=dict)


class Agent(Protocol):
    def answer(self, mcq: MCQ) -> Prediction: ...

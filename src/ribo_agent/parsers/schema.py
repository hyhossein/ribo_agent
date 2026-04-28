"""Canonical record shape for every parsed multiple-choice question."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class MCQ:
    """One multiple-choice question in canonical form.

    Fields are intentionally flat so the JSONL on disk is trivial to read,
    diff, grep, and stream.
    """

    qid: str                           # stable id, unique within the project
    source: str                        # which PDF it came from
    stem: str                          # the question text
    options: dict[str, str]            # {'A': '...', 'B': '...', 'C': '...', 'D': '...'}
    correct: str                       # 'A' | 'B' | 'C' | 'D'

    # optional metadata present on some sources (rich on sample_level1)
    content_domain: str | None = None
    competency: str | None = None
    cognitive_level: str | None = None
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

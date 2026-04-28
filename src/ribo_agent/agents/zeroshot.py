"""Zero-shot agent. No retrieval, no exemplars.

Purpose: establish the floor. Whatever number this produces is how well
the raw LLM knows Canadian insurance law. RAG (v0.5.0) will be measured
as lift above this baseline.

Answer extraction is defensive. Models return the answer in a handful of
shapes; in order of preference we accept:

  1. `<answer>X</answer>` — explicit tag we asked for in the prompt
  2. Final line of the form `Answer: X` / `The answer is X`
  3. Any sole uppercase A/B/C/D in the last 100 chars
  4. First A/B/C/D anywhere in the response

If none of those match, the prediction is None and the question counts
as wrong. We NEVER guess — surfacing the failure is more useful than
inventing an answer.
"""
from __future__ import annotations

import re

from ..llm.base import LLMClient
from ..parsers.schema import MCQ
from .base import Prediction


SYSTEM = (
    "You are preparing to take the Ontario RIBO Level 1 insurance broker "
    "licensing exam. You answer multiple-choice questions by selecting "
    "the single best option."
)

USER_TEMPLATE = """Answer the following multiple-choice question from the Ontario RIBO Level 1 broker licensing exam.

Question: {stem}

Options:
A. {a}
B. {b}
C. {c}
D. {d}

Think step by step briefly, then give your final answer on the last line in the exact format:
<answer>LETTER</answer>

where LETTER is one of A, B, C, or D."""


_TAG_RE = re.compile(r"<answer>\s*([A-D])\s*</answer>", re.IGNORECASE)
_FINAL_LINE_RE = re.compile(
    r"(?:answer\s*(?:is|:)?\s*)([A-D])\b", re.IGNORECASE
)


def extract_answer(text: str) -> str | None:
    """Return the predicted letter or None if the response is uninterpretable."""
    if not text:
        return None

    # 1. explicit tag
    m = _TAG_RE.search(text)
    if m:
        return m.group(1).upper()

    # 2. "Answer: X" or "The answer is X"
    tail = text[-400:]
    m = _FINAL_LINE_RE.search(tail)
    if m:
        return m.group(1).upper()

    # 3. any standalone A/B/C/D in the last 100 chars
    m = re.search(r"\b([A-D])\b", text[-100:])
    if m:
        return m.group(1).upper()

    # 4. first A/B/C/D anywhere
    m = re.search(r"\b([A-D])\b", text)
    return m.group(1).upper() if m else None


def build_prompt(mcq: MCQ) -> str:
    return (
        SYSTEM
        + "\n\n"
        + USER_TEMPLATE.format(
            stem=mcq.stem,
            a=mcq.options["A"],
            b=mcq.options["B"],
            c=mcq.options["C"],
            d=mcq.options["D"],
        )
    )


class ZeroShotAgent:
    def __init__(
        self,
        llm: LLMClient,
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens

    def answer(self, mcq: MCQ) -> Prediction:
        prompt = build_prompt(mcq)
        resp = self.llm.complete(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=["</answer>"],
        )
        # If the model stopped on </answer>, close the tag so _TAG_RE matches.
        raw = resp.text
        if _TAG_RE.search(raw) is None and re.search(
            r"<answer>\s*[A-D]\s*$", raw, re.IGNORECASE
        ):
            raw = raw + "</answer>"

        predicted = extract_answer(raw)
        return Prediction(
            qid=mcq.qid,
            predicted=predicted,
            correct=mcq.correct,
            is_correct=predicted == mcq.correct,
            raw_response=resp.text,
            latency_ms=resp.latency_ms,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            extras={"model": resp.model, "backend": resp.backend},
        )

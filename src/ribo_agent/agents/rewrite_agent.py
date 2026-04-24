"""Question-rewrite agent.

Two-stage pipeline:
1. Rewrite: an LLM clarifies the question — expands abbreviations,
   identifies the relevant regulation/section, removes ambiguity.
2. Answer: the rewritten question + wiki context feeds into the
   answer prompt.

This addresses a common failure mode: the model knows the rule but
can't connect the original question phrasing to the right section.
"""
from __future__ import annotations

import re
import time

from ..llm.base import LLMClient
from ..parsers.schema import MCQ
from .base import Prediction
from .wiki_agent import WikiAgent
from .zeroshot import extract_answer


REWRITE_PROMPT = """You are an expert Ontario insurance law tutor. A student is about to
answer a RIBO Level 1 licensing exam question. Before they see it,
rewrite the question to make it clearer and more precise.

Rules for rewriting:
- Expand any abbreviations (OAP = Ontario Automobile Policy, RIB Act = Registered Insurance Brokers Act)
- Identify which specific regulation, by-law, or act section is being tested
- Clarify any ambiguous pronouns or references
- Keep all four options exactly as they are — only rewrite the stem
- Add a one-line hint: "This question tests: [topic/section]"

Original question: {stem}

Options:
A. {a}
B. {b}
C. {c}
D. {d}

Rewrite the question stem only (keep options unchanged), then add the hint line."""


class RewriteAgent:
    def __init__(
        self,
        llm: LLMClient,
        wiki_agent: WikiAgent | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        rewrite_max_tokens: int = 256,
    ) -> None:
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.rewrite_max_tokens = rewrite_max_tokens

        # Reuse or create a wiki agent for the answer step
        if wiki_agent is None:
            self._wiki_agent = WikiAgent(llm, temperature=temperature, max_tokens=max_tokens)
        else:
            self._wiki_agent = wiki_agent

    def _rewrite(self, mcq: MCQ) -> str:
        prompt = REWRITE_PROMPT.format(
            stem=mcq.stem,
            a=mcq.options["A"],
            b=mcq.options["B"],
            c=mcq.options["C"],
            d=mcq.options["D"],
        )
        resp = self.llm.complete(
            prompt,
            temperature=0.0,
            max_tokens=self.rewrite_max_tokens,
        )
        return resp.text

    def answer(self, mcq: MCQ) -> Prediction:
        t0 = time.perf_counter()

        # Stage 1: rewrite
        rewritten = self._rewrite(mcq)

        # Stage 2: create a modified MCQ with the rewritten stem
        rewritten_mcq = MCQ(
            qid=mcq.qid,
            source=mcq.source,
            stem=rewritten,
            options=mcq.options,
            correct=mcq.correct,
            content_domain=mcq.content_domain,
            competency=mcq.competency,
            cognitive_level=mcq.cognitive_level,
            extras=mcq.extras,
        )

        # Answer using the wiki agent
        pred = self._wiki_agent.answer(rewritten_mcq)

        total_latency = (time.perf_counter() - t0) * 1000

        return Prediction(
            qid=mcq.qid,
            predicted=pred.predicted,
            correct=mcq.correct,
            is_correct=pred.predicted == mcq.correct,
            raw_response=f"[REWRITTEN STEM]\n{rewritten}\n\n[ANSWER]\n{pred.raw_response}",
            latency_ms=total_latency,
            prompt_tokens=pred.prompt_tokens,
            completion_tokens=pred.completion_tokens,
            extras={
                "model": pred.extras.get("model", ""),
                "backend": pred.extras.get("backend", ""),
                "agent": "rewrite+wiki",
                "rewritten_stem": rewritten,
            },
        )

"""RAG agent — retrieval-augmented generation with document citations.

Retrieves the top-k most relevant KB chunks for each question, injects
them into the prompt with source citations, and asks the LLM to answer.
Every prediction carries a `citations` list so answers are traceable
back to the original documents.
"""
from __future__ import annotations

import re

from ..kb.retriever import Retriever, RetrievalHit
from ..llm.base import LLMClient
from ..parsers.schema import MCQ
from .base import Prediction
from .zeroshot import extract_answer


SYSTEM = (
    "You are preparing to take the Ontario RIBO Level 1 insurance broker "
    "licensing exam. You answer multiple-choice questions by selecting "
    "the single best option.\n\n"
    "IMPORTANT: Base your reasoning on the reference documents provided. "
    "When explaining your answer, cite the specific document and section "
    "that supports your choice (e.g. 'According to RIB Act s. 14, ...')."
)

USER_TEMPLATE = """Use the following reference documents to answer the question. Cite the relevant source(s) in your reasoning.

{context}

---

Question: {stem}

Options:
A. {a}
B. {b}
C. {c}
D. {d}

Instructions:
1. Think step by step, citing the relevant document section(s) that support your reasoning.
2. Give your final answer on the last line in the exact format:
<answer>LETTER</answer>

where LETTER is one of A, B, C, or D."""


_TAG_RE = re.compile(r"<answer>\s*([A-D])\s*</answer>", re.IGNORECASE)


def _format_context(hits: list[RetrievalHit]) -> str:
    """Format retrieved chunks as numbered reference blocks."""
    blocks: list[str] = []
    for i, hit in enumerate(hits, 1):
        header = f"[{i}] {hit.citation} (source: {hit.source})"
        blocks.append(f"{header}\n{hit.text}")
    return "\n\n".join(blocks)


def _hit_to_citation(hit: RetrievalHit, rank: int) -> dict:
    """Convert a RetrievalHit to a serialisable citation dict."""
    return {
        "rank": rank,
        "source": hit.source,
        "citation": hit.citation,
        "section": hit.chunk.section,
        "score": round(hit.score, 4),
        "snippet": hit.text[:300],
    }


class RAGAgent:
    """Retrieval-augmented agent with document citations."""

    def __init__(
        self,
        llm: LLMClient,
        retriever: Retriever,
        *,
        top_k: int = 5,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.top_k = top_k
        self.temperature = temperature
        self.max_tokens = max_tokens

    def answer(self, mcq: MCQ) -> Prediction:
        # 1. Retrieve relevant chunks
        query = mcq.stem
        hits = self.retriever.search(query, k=self.top_k)

        # 2. Build prompt with context
        context = _format_context(hits)
        prompt = (
            SYSTEM
            + "\n\n"
            + USER_TEMPLATE.format(
                context=context,
                stem=mcq.stem,
                a=mcq.options["A"],
                b=mcq.options["B"],
                c=mcq.options["C"],
                d=mcq.options["D"],
            )
        )

        # 3. Call LLM
        resp = self.llm.complete(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=["</answer>"],
        )

        # 4. Extract answer
        raw = resp.text
        if _TAG_RE.search(raw) is None and re.search(
            r"<answer>\s*[A-D]\s*$", raw, re.IGNORECASE
        ):
            raw = raw + "</answer>"
        predicted = extract_answer(raw)

        # 5. Build citations list
        citations = [
            _hit_to_citation(hit, rank=i)
            for i, hit in enumerate(hits, 1)
        ]

        return Prediction(
            qid=mcq.qid,
            predicted=predicted,
            correct=mcq.correct,
            is_correct=predicted == mcq.correct,
            raw_response=resp.text,
            latency_ms=resp.latency_ms,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            citations=citations,
            extras={
                "model": resp.model,
                "backend": resp.backend,
                "top_k": self.top_k,
            },
        )

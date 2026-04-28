"""RAG agent — retrieval-augmented generation with document citations.

Supports both simple (FAISS-only) and hybrid (FAISS + BM25 + KG) retrieval.
Every prediction carries a `citations` list traceable to document → section → text.
Includes timing breakdown for cost/performance profiling.
"""
from __future__ import annotations

import re
import time

from ..kb.retriever import Retriever, RetrievalHit
from ..llm.base import LLMClient
from ..parsers.schema import MCQ
from .base import Prediction
from .zeroshot import extract_answer


SYSTEM = (
    "You are preparing to take the Ontario RIBO Level 1 insurance broker "
    "licensing exam. You answer multiple-choice questions by selecting "
    "the single best option. Use the reference documents when relevant, "
    "but trust your knowledge if the documents are not helpful."
)

USER_TEMPLATE = """Reference documents:
{context}

---

Question: {stem}

A. {a}
B. {b}
C. {c}
D. {d}

First identify which reference (if any) is relevant, quote the key phrase, then pick the best answer.
<answer>LETTER</answer>"""


_TAG_RE = re.compile(r"<answer>\s*([A-D])\s*</answer>", re.IGNORECASE)


def _format_context_simple(hits: list[RetrievalHit]) -> str:
    blocks: list[str] = []
    for i, hit in enumerate(hits, 1):
        header = f"[{i}] {hit.citation} (source: {hit.source})"
        blocks.append(f"{header}\n{hit.text[:500].rstrip()}")
    return "\n\n".join(blocks)


def _format_context_hybrid(hits) -> str:
    """Format HybridHit objects."""
    blocks: list[str] = []
    for i, hit in enumerate(hits, 1):
        header = f"[{i}] {hit.citation} (source: {hit.source})"
        page_info = ""
        if hit.chunk.page_number:
            page_info = f" [page {hit.chunk.page_number}]"
        blocks.append(f"{header}{page_info}\n{hit.text[:500].rstrip()}")
    return "\n\n".join(blocks)


def _hit_to_citation_simple(hit: RetrievalHit, rank: int) -> dict:
    return {
        "rank": rank,
        "source": hit.source,
        "citation": hit.citation,
        "section": hit.chunk.section,
        "page_number": hit.chunk.page_number,
        "score": round(hit.score, 4),
        "snippet": hit.text[:300],
        "retrieval_signals": ["dense"],
    }


class RAGAgent:
    """Retrieval-augmented agent with document citations."""

    def __init__(
        self,
        llm: LLMClient,
        retriever,  # Retriever or HybridRetriever
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
        # detect retriever type
        self._is_hybrid = hasattr(retriever, 'kg')

    def answer(self, mcq: MCQ) -> Prediction:
        # 1. Retrieve
        t_ret = time.perf_counter()
        hits = self.retriever.search(mcq.stem, k=self.top_k)
        retrieval_ms = (time.perf_counter() - t_ret) * 1000

        # 2. Build prompt
        if self._is_hybrid:
            context = _format_context_hybrid(hits)
        else:
            context = _format_context_simple(hits)

        prompt = (
            SYSTEM + "\n\n"
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
        t_gen = time.perf_counter()
        resp = self.llm.complete(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=["</answer>"],
        )
        generation_ms = (time.perf_counter() - t_gen) * 1000

        # 4. Extract answer
        raw = resp.text
        if _TAG_RE.search(raw) is None and re.search(
            r"<answer>\s*[A-D]\s*$", raw, re.IGNORECASE
        ):
            raw = raw + "</answer>"
        predicted = extract_answer(raw)

        # 5. Build citations
        if self._is_hybrid:
            citations = [hit.to_citation_dict(rank=i) for i, hit in enumerate(hits, 1)]
        else:
            citations = [_hit_to_citation_simple(hit, rank=i) for i, hit in enumerate(hits, 1)]

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
                "retrieval_ms": round(retrieval_ms, 1),
                "generation_ms": round(generation_ms, 1),
                "retrieval_type": "hybrid" if self._is_hybrid else "dense",
            },
        )

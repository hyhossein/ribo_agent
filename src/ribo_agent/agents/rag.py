"""RAG agent — retrieval-augmented generation with document citations.

Supports both simple (FAISS-only) and hybrid (FAISS + BM25 + KG) retrieval.
Every prediction carries a `citations` list traceable to document → section → text.
Includes timing breakdown, confidence scoring, and clean citation paths.
"""
from __future__ import annotations

import re
import time

from ..kb.retriever import Retriever, RetrievalHit
from ..llm.base import LLMClient
from ..parsers.schema import MCQ
from .base import Prediction
from .zeroshot import extract_answer


# ---------------------------------------------------------------------------
# Prompt — reasoning-first (not lookup)
# ---------------------------------------------------------------------------
SYSTEM = (
    "You are an expert Ontario insurance professional with deep knowledge "
    "of the Registered Insurance Brokers Act, Ontario Regulation 991, RIBO "
    "bylaws, OAP-1, and general insurance principles. You are taking the "
    "RIBO Level 1 licensing exam."
)

USER_TEMPLATE = """Reference documents (use if relevant, but reason from principles):
{context}

---

Question: {stem}

A. {a}
B. {b}
C. {c}
D. {d}

Instructions:
1. Identify the insurance concept being tested.
2. Recall the relevant legal rule or principle — from the references OR your training.
3. Eliminate wrong options with brief reasoning.
4. State your answer.

You MUST pick exactly one of A, B, C, or D. Never refuse.

<answer>LETTER</answer>"""


_TAG_RE = re.compile(r"<answer>\s*([A-D])\s*</answer>", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Citation formatting — clean, human-readable
# ---------------------------------------------------------------------------
_SOURCE_LABELS = {
    "RIB_Act_1990": "Registered Insurance Brokers Act",
    "Ont_Reg_991": "Ontario Regulation 991",
    "RIBO_Bylaws": "RIBO Bylaws",
    "OAP_1": "Ontario Auto Policy (OAP-1)",
}


def _clean_source(raw: str) -> str:
    """Convert raw source id to readable name."""
    return _SOURCE_LABELS.get(raw, raw.replace("_", " "))


def _clean_citation_path(source: str, section: str | None,
                         page: int | None) -> str:
    """Build 'Document → Section X → Page Y' string."""
    parts = [_clean_source(source)]
    if section:
        parts.append(f"§ {section}")
    if page:
        parts.append(f"p. {page}")
    return " → ".join(parts)


def _format_context(hits, is_hybrid: bool) -> str:
    blocks: list[str] = []
    for i, hit in enumerate(hits, 1):
        page = hit.chunk.page_number
        path = _clean_citation_path(hit.source, hit.chunk.section, page)
        blocks.append(f"[{i}] {path}\n{hit.text[:500].rstrip()}")
    return "\n\n".join(blocks)


def _hit_to_citation(hit, rank: int, is_hybrid: bool) -> dict:
    page = hit.chunk.page_number
    signals = (hit.source_signals if is_hybrid
               else ["dense"])
    return {
        "rank": rank,
        "source": _clean_source(hit.source),
        "citation": hit.citation,
        "path": _clean_citation_path(hit.source, hit.chunk.section, page),
        "section": hit.chunk.section,
        "page_number": page,
        "score": round(hit.score, 4),
        "snippet": hit.text[:300],
        "retrieval_signals": signals,
    }


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------
def _compute_confidence(
    predicted: str | None,
    raw: str,
    hits,
    retrieval_ms: float,
) -> float:
    """0-1 confidence score based on multiple signals."""
    if predicted is None:
        return 0.0

    score = 0.0

    # Signal 1: explicit <answer> tag present (model was confident enough to tag)
    if _TAG_RE.search(raw):
        score += 0.35
    else:
        score += 0.10  # weaker extraction

    # Signal 2: top retrieval score (high relevance = more grounded)
    if hits:
        top_score = hits[0].score if hasattr(hits[0], 'score') else 0
        score += min(0.30, top_score * 0.35)  # cap at 0.30

    # Signal 3: answer mentioned multiple times (consistency)
    mentions = len(re.findall(rf"\b{predicted}\b", raw[-300:]))
    if mentions >= 2:
        score += 0.15
    elif mentions == 1:
        score += 0.08

    # Signal 4: reasoning length (very short = likely guessing)
    if len(raw) > 200:
        score += 0.15
    elif len(raw) > 50:
        score += 0.05

    return round(min(1.0, score), 3)


class RAGAgent:
    """Retrieval-augmented agent with document citations and confidence."""

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
        self._is_hybrid = hasattr(retriever, 'kg')

    def answer(self, mcq: MCQ) -> Prediction:
        # 1. Retrieve
        t_ret = time.perf_counter()
        hits = self.retriever.search(mcq.stem, k=self.top_k)
        retrieval_ms = (time.perf_counter() - t_ret) * 1000

        # 2. Build prompt
        context = _format_context(hits, self._is_hybrid)
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

        # 4. Extract answer (robust — never refuse)
        raw = resp.text
        if _TAG_RE.search(raw) is None and re.search(
            r"<answer>\s*[A-D]\s*$", raw, re.IGNORECASE
        ):
            raw = raw + "</answer>"
        predicted = extract_answer(raw)

        # Fallback: if still None, pick the first letter mentioned after "answer"
        if predicted is None:
            m = re.search(r"[Aa]nswer.*?([A-D])", raw)
            if m:
                predicted = m.group(1).upper()
            else:
                # absolute last resort: pick most-mentioned option
                counts = {L: len(re.findall(rf"\b{L}\b", raw))
                          for L in "ABCD"}
                best = max(counts, key=counts.get)
                if counts[best] > 0:
                    predicted = best

        # 5. Confidence
        confidence = _compute_confidence(predicted, raw, hits, retrieval_ms)

        # 6. Clean citations
        citations = [
            _hit_to_citation(hit, rank=i, is_hybrid=self._is_hybrid)
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
                "retrieval_ms": round(retrieval_ms, 1),
                "generation_ms": round(generation_ms, 1),
                "retrieval_type": "hybrid" if self._is_hybrid else "dense",
                "confidence": confidence,
            },
        )

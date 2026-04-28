"""Faithfulness validator — checks if LLM citations are grounded in retrieved chunks.

Given a model's response and the retrieved chunks, this module checks
whether the text the model claims to cite actually appears in the
retrieved context. This catches hallucinated citations.

Three checks:
  1. **Overlap check** — do quoted phrases from the response appear in
     the retrieved chunks? (token-level Jaccard)
  2. **Citation mention check** — does the response reference the same
     source/section identifiers as the retrieved chunks?
  3. **Grounding score** — what fraction of the response's reasoning
     sentences can be traced to a retrieved chunk?

Each prediction gets a faithfulness verdict:
  GROUNDED     — response cites retrieved text, citations match
  PARTIAL      — some citations match, some don't
  UNGROUNDED   — response contradicts or ignores retrieved context
  NO_CONTEXT   — no retrieval was performed (zero-shot)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class FaithfulnessVerdict(str, Enum):
    GROUNDED = "GROUNDED"
    PARTIAL = "PARTIAL"
    UNGROUNDED = "UNGROUNDED"
    NO_CONTEXT = "NO_CONTEXT"


@dataclass
class FaithfulnessResult:
    verdict: FaithfulnessVerdict
    grounding_score: float        # 0.0 – 1.0
    matched_citations: list[str]  # citations found in both response and chunks
    unmatched_claims: list[str]   # claims in response not found in chunks
    details: str                  # human-readable explanation


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens, stripped of punctuation."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _extract_quoted_phrases(text: str) -> list[str]:
    """Extract text inside quotation marks from the response."""
    return re.findall(r'"([^"]{10,})"', text) + re.findall(r"'([^']{10,})'", text)


def _extract_citation_refs(text: str) -> list[str]:
    """Extract section/source references like 's. 14', 'Section 7', 'RIB Act'."""
    patterns = [
        r"s\.\s*\d+(?:\(\d+\))?",           # s. 14, s. 14(2)
        r"Section\s+\d+",                     # Section 7
        r"Article\s+\d+",                     # Article 3
        r"Reg(?:ulation)?\.?\s*\d{3}",        # Reg. 991, Regulation 991
        r"RIB\s+Act",                         # RIB Act
        r"OAP\s+\d+",                         # OAP 1
        r"By-?Law\s+(?:No\.?\s*)?\d+",        # By-Law No. 1
    ]
    refs: list[str] = []
    for pat in patterns:
        refs.extend(re.findall(pat, text, re.IGNORECASE))
    return refs


def _sentence_split(text: str) -> list[str]:
    """Split text into rough sentences."""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sents if len(s) > 15]


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def check_faithfulness(
    response: str,
    citations: list[dict],
    *,
    threshold_grounded: float = 0.4,
    threshold_partial: float = 0.15,
) -> FaithfulnessResult:
    """Evaluate how well the response is grounded in the retrieved chunks.

    Args:
        response: The raw LLM response text.
        citations: List of citation dicts from the Prediction (each has
            'source', 'citation', 'snippet', 'score').
        threshold_grounded: Minimum grounding score for GROUNDED verdict.
        threshold_partial: Minimum grounding score for PARTIAL verdict.

    Returns:
        FaithfulnessResult with verdict, score, and details.
    """
    if not citations:
        return FaithfulnessResult(
            verdict=FaithfulnessVerdict.NO_CONTEXT,
            grounding_score=0.0,
            matched_citations=[],
            unmatched_claims=[],
            details="No retrieval context — zero-shot prediction.",
        )

    # Combine all chunk text into one reference corpus
    chunk_texts = [c.get("snippet", "") for c in citations]
    corpus = " ".join(chunk_texts)
    corpus_tokens = _tokenize(corpus)

    # 1. Citation reference matching
    response_refs = _extract_citation_refs(response)
    chunk_refs = set()
    for c in citations:
        chunk_refs.update(_extract_citation_refs(c.get("citation", "")))
        chunk_refs.update(_extract_citation_refs(c.get("snippet", "")))

    matched_citations = [
        r for r in response_refs
        if any(_jaccard(_tokenize(r), _tokenize(cr)) > 0.5 for cr in chunk_refs)
    ]

    # 2. Quoted phrase overlap
    quoted = _extract_quoted_phrases(response)
    quotes_found = sum(
        1 for q in quoted
        if any(q.lower() in ct.lower() for ct in chunk_texts)
    )

    # 3. Sentence-level grounding
    sentences = _sentence_split(response)
    grounded_sents = 0
    unmatched: list[str] = []
    for sent in sentences:
        sent_tokens = _tokenize(sent)
        overlap = _jaccard(sent_tokens, corpus_tokens)
        if overlap > 0.15:
            grounded_sents += 1
        else:
            # Skip very short sentences or meta-sentences
            if len(sent_tokens) > 5 and not sent.lower().startswith(("the answer", "therefore")):
                unmatched.append(sent[:120])

    # Composite score
    n_sents = max(len(sentences), 1)
    sent_score = grounded_sents / n_sents
    quote_score = (quotes_found / max(len(quoted), 1)) if quoted else 0.5
    ref_score = (len(matched_citations) / max(len(response_refs), 1)) if response_refs else 0.5

    grounding_score = 0.5 * sent_score + 0.3 * ref_score + 0.2 * quote_score

    # Verdict
    if grounding_score >= threshold_grounded:
        verdict = FaithfulnessVerdict.GROUNDED
    elif grounding_score >= threshold_partial:
        verdict = FaithfulnessVerdict.PARTIAL
    else:
        verdict = FaithfulnessVerdict.UNGROUNDED

    details_parts = [
        f"Sentence grounding: {grounded_sents}/{n_sents} ({sent_score:.0%})",
        f"Citation refs matched: {len(matched_citations)}/{len(response_refs)}",
    ]
    if quoted:
        details_parts.append(f"Quoted phrases verified: {quotes_found}/{len(quoted)}")
    if unmatched:
        details_parts.append(f"Ungrounded claims: {len(unmatched)}")

    return FaithfulnessResult(
        verdict=verdict,
        grounding_score=round(grounding_score, 4),
        matched_citations=matched_citations,
        unmatched_claims=unmatched[:5],  # cap at 5
        details=" | ".join(details_parts),
    )

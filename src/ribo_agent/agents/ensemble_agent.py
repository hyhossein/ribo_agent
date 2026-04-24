"""Ensemble agent v3: targeted fixes for each failure mode.

Pipeline:
1. Rewrite the question (expand abbreviations, identify regulation)
2. Answer using the compiled wiki
3. Check confidence via hedging detection
4. If low confidence → RAG fallback (BM25 over raw chunks)
5. If calculation question → self-consistency voting (5x, temp=0.7)
6. Log full trace: rewritten stem, wiki sections, chunks, votes, citations

Designed to push accuracy from 88.76% to 90%+ by targeting the three
failure patterns identified in docs/ERROR_ANALYSIS.md.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

from ..llm.base import LLMClient
from ..parsers.schema import MCQ
from .base import Prediction
from .zeroshot import extract_answer


# Hedging patterns that signal low confidence
_HEDGE_PATTERNS = [
    r"cannot verify",
    r"not covered in the.+wiki",
    r"based on general.+principles",
    r"typically\b",
    r"most likely",
    r"would likely",
    r"I'm not (?:entirely )?sure",
    r"cannot determine",
    r"without.+information",
    r"appears to be",
]
_HEDGE_RE = re.compile("|".join(_HEDGE_PATTERNS), re.IGNORECASE)

# Patterns suggesting a calculation question
_CALC_PATTERNS = [
    r"\$\d+",
    r"\d+%",
    r"co-?insurance",
    r"deductible",
    r"replacement (?:cost|benefit)",
    r"income replacement",
    r"OPCF\s*\d+",
    r"calculate",
    r"how much",
    r"what amount",
]
_CALC_RE = re.compile("|".join(_CALC_PATTERNS), re.IGNORECASE)

WIKI_ANSWER_PROMPT = """You are taking the Ontario RIBO Level 1 insurance broker licensing exam.
You have access to a comprehensive study wiki compiled from the official
study materials. Use it to answer the following question.

IMPORTANT: Cite the specific section, regulation, or by-law that supports
your answer. If you are unsure, say "LOW CONFIDENCE" before your answer.

STUDY WIKI:
{wiki}

---

Question: {stem}

Options:
A. {a}
B. {b}
C. {c}
D. {d}

Think step by step. Cite the specific regulation or section. Then give your
final answer on the last line in the exact format:
<answer>LETTER</answer>"""

RAG_FALLBACK_PROMPT = """You are taking the Ontario RIBO Level 1 insurance broker licensing exam.
Your first attempt was uncertain. Here are the most relevant raw study
material passages retrieved specifically for this question.

RETRIEVED PASSAGES:
{passages}

STUDY WIKI (for additional context):
{wiki_short}

---

Question: {stem}

Options:
A. {a}
B. {b}
C. {c}
D. {d}

Read the retrieved passages carefully. The answer is in there. Think step
by step, cite the exact passage, then give your final answer:
<answer>LETTER</answer>"""

REWRITE_PROMPT = """You are an expert Ontario insurance law tutor. Rewrite this exam question
to be clearer: expand abbreviations (OAP, RIB Act, OPCF), identify which
regulation/section is being tested, clarify any ambiguous references.
Keep the four options unchanged — only rewrite the question stem.

Original: {stem}

Options: A. {a} | B. {b} | C. {c} | D. {d}

Rewritten question stem:"""


def _simple_bm25(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Dead-simple keyword match scorer. No external deps."""
    query_terms = set(re.findall(r"\w+", query.lower()))
    scored = []
    for c in chunks:
        text_terms = set(re.findall(r"\w+", c.get("text", "").lower()))
        citation_terms = set(re.findall(r"\w+", c.get("citation", "").lower()))
        overlap = len(query_terms & (text_terms | citation_terms))
        scored.append((overlap, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


class EnsembleAgent:
    def __init__(
        self,
        llm: LLMClient,
        kb_path: Path | None = None,
        wiki_cache_path: Path | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        wiki_max_tokens: int = 4096,
        sc_samples: int = 5,
        sc_temperature: float = 0.7,
    ) -> None:
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.wiki_max_tokens = wiki_max_tokens
        self.sc_samples = sc_samples
        self.sc_temperature = sc_temperature

        root = Path(__file__).resolve().parents[3]
        self.kb_path = kb_path or root / "data" / "kb" / "chunks.jsonl"
        self.wiki_cache = wiki_cache_path or root / "data" / "kb" / "wiki_compiled.md"

        self._wiki: str | None = None
        self._chunks: list[dict] | None = None

    def _get_wiki(self) -> str:
        if self._wiki is None:
            if self.wiki_cache.exists():
                self._wiki = self.wiki_cache.read_text()
            else:
                # Build wiki using the wiki_agent's compilation
                from .wiki_agent import WikiAgent
                wa = WikiAgent(self.llm, self.kb_path, self.wiki_cache,
                               wiki_max_tokens=self.wiki_max_tokens)
                self._wiki = wa._build_wiki()
        return self._wiki

    def _get_chunks(self) -> list[dict]:
        if self._chunks is None:
            self._chunks = [json.loads(l) for l in self.kb_path.open()]
        return self._chunks

    def _rewrite(self, mcq: MCQ) -> str:
        prompt = REWRITE_PROMPT.format(
            stem=mcq.stem,
            a=mcq.options["A"], b=mcq.options["B"],
            c=mcq.options["C"], d=mcq.options["D"],
        )
        resp = self.llm.complete(prompt, temperature=0.0, max_tokens=256)
        return resp.text.strip()

    def _answer_with_wiki(self, stem: str, mcq: MCQ) -> tuple[str, str]:
        wiki = self._get_wiki()
        max_wiki = 30000
        wiki_text = wiki[:max_wiki] if len(wiki) > max_wiki else wiki

        prompt = WIKI_ANSWER_PROMPT.format(
            wiki=wiki_text, stem=stem,
            a=mcq.options["A"], b=mcq.options["B"],
            c=mcq.options["C"], d=mcq.options["D"],
        )
        resp = self.llm.complete(prompt, temperature=self.temperature,
                                 max_tokens=self.max_tokens)
        raw = resp.text
        if re.search(r"<answer>\s*[A-D]\s*$", raw, re.IGNORECASE):
            raw += "</answer>"
        return raw, extract_answer(raw) or ""

    def _is_low_confidence(self, response: str) -> bool:
        return bool(_HEDGE_RE.search(response))

    def _is_calculation(self, stem: str, options: dict) -> bool:
        text = stem + " ".join(options.values())
        return bool(_CALC_RE.search(text))

    def _rag_fallback(self, stem: str, mcq: MCQ) -> tuple[str, str]:
        chunks = self._get_chunks()
        relevant = _simple_bm25(stem, chunks, top_k=5)
        passages = "\n\n---\n\n".join(
            f"[{c.get('citation', '?')}]\n{c.get('text', '')}"
            for c in relevant
        )
        wiki = self._get_wiki()
        wiki_short = wiki[:10000] if len(wiki) > 10000 else wiki

        prompt = RAG_FALLBACK_PROMPT.format(
            passages=passages, wiki_short=wiki_short, stem=stem,
            a=mcq.options["A"], b=mcq.options["B"],
            c=mcq.options["C"], d=mcq.options["D"],
        )
        resp = self.llm.complete(prompt, temperature=0.0,
                                 max_tokens=self.max_tokens)
        raw = resp.text
        if re.search(r"<answer>\s*[A-D]\s*$", raw, re.IGNORECASE):
            raw += "</answer>"
        return raw, extract_answer(raw) or ""

    def _self_consistency(self, stem: str, mcq: MCQ) -> tuple[str, dict, str]:
        wiki = self._get_wiki()
        max_wiki = 30000
        wiki_text = wiki[:max_wiki] if len(wiki) > max_wiki else wiki

        prompt = WIKI_ANSWER_PROMPT.format(
            wiki=wiki_text, stem=stem,
            a=mcq.options["A"], b=mcq.options["B"],
            c=mcq.options["C"], d=mcq.options["D"],
        )

        votes: Counter = Counter()
        responses: list[str] = []
        for _ in range(self.sc_samples):
            resp = self.llm.complete(prompt, temperature=self.sc_temperature,
                                     max_tokens=self.max_tokens)
            raw = resp.text
            if re.search(r"<answer>\s*[A-D]\s*$", raw, re.IGNORECASE):
                raw += "</answer>"
            letter = extract_answer(raw)
            if letter:
                votes[letter] += 1
            responses.append(raw)

        vote_dict = dict(votes)
        winner = votes.most_common(1)[0][0] if votes else ""
        all_responses = "\n---VOTE---\n".join(responses)
        return all_responses, vote_dict, winner

    def answer(self, mcq: MCQ) -> Prediction:
        t0 = time.perf_counter()
        trace: dict = {"agent": "ensemble_v3"}

        # Stage 1: Rewrite
        rewritten = self._rewrite(mcq)
        trace["rewritten_stem"] = rewritten

        # Stage 2: Wiki answer
        wiki_response, wiki_answer = self._answer_with_wiki(rewritten, mcq)
        trace["wiki_response"] = wiki_response
        trace["wiki_answer"] = wiki_answer

        is_low_conf = self._is_low_confidence(wiki_response)
        is_calc = self._is_calculation(mcq.stem, mcq.options)
        trace["low_confidence"] = is_low_conf
        trace["is_calculation"] = is_calc

        final_answer = wiki_answer
        final_response = wiki_response

        # Stage 3: RAG fallback for low-confidence answers
        if is_low_conf:
            rag_response, rag_answer = self._rag_fallback(rewritten, mcq)
            trace["rag_fallback"] = True
            trace["rag_response"] = rag_response
            trace["rag_answer"] = rag_answer
            if rag_answer:
                final_answer = rag_answer
                final_response = rag_response

        # Stage 4: Self-consistency for calculation questions
        if is_calc:
            sc_responses, sc_votes, sc_winner = self._self_consistency(rewritten, mcq)
            trace["self_consistency"] = True
            trace["sc_votes"] = sc_votes
            trace["sc_winner"] = sc_winner
            if sc_winner:
                final_answer = sc_winner
                final_response = sc_responses

        trace["final_answer"] = final_answer
        latency = (time.perf_counter() - t0) * 1000

        return Prediction(
            qid=mcq.qid,
            predicted=final_answer if final_answer in "ABCD" else None,
            correct=mcq.correct,
            is_correct=(final_answer == mcq.correct),
            raw_response=final_response[:5000],
            latency_ms=latency,
            prompt_tokens=None,
            completion_tokens=None,
            extras=trace,
        )

"""Wiki-backed agent (Karpathy LLM Wiki pattern).

Instead of retrieving raw chunks per question, we pre-compile the entire
study corpus into a structured knowledge wiki at startup. Questions are
then answered against the compiled wiki, not raw documents.

The wiki is built once per eval run and cached. This amortizes the
compilation cost across all 169 questions.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..llm.base import LLMClient
from ..parsers.schema import MCQ
from .base import Prediction
from .zeroshot import extract_answer


WIKI_SYSTEM = """You are a senior Ontario insurance law expert preparing a study guide.
Given the following raw study material chunks, compile a comprehensive
knowledge wiki organized by topic. Include all specific section numbers,
definitions, rules, exceptions, and cross-references. Be exhaustive —
every detail matters for the licensing exam.

Output a well-structured reference document that another expert could
use to answer any RIBO Level 1 exam question."""

ANSWER_SYSTEM = """You are taking the Ontario RIBO Level 1 insurance broker licensing exam.
You have access to a comprehensive study wiki compiled from the official
study materials. Use it to answer the following question.

STUDY WIKI:
{wiki}

---

Answer the following multiple-choice question. Think step by step,
citing specific sections from the wiki. Then give your final answer.

Question: {stem}

Options:
A. {a}
B. {b}
C. {c}
D. {d}

Think step by step briefly, then give your final answer on the last line in the exact format:
<answer>LETTER</answer>

where LETTER is one of A, B, C, or D."""


class WikiAgent:
    def __init__(
        self,
        llm: LLMClient,
        kb_path: Path | None = None,
        wiki_cache_path: Path | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        wiki_max_tokens: int = 4096,
    ) -> None:
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.wiki_max_tokens = wiki_max_tokens

        # Default paths
        root = Path(__file__).resolve().parents[3]
        self.kb_path = kb_path or root / "data" / "kb" / "chunks.jsonl"
        self.wiki_cache = wiki_cache_path or root / "data" / "kb" / "wiki_compiled.md"

        self._wiki: str | None = None

    def _load_chunks(self) -> list[dict]:
        chunks = []
        with self.kb_path.open() as f:
            for line in f:
                chunks.append(json.loads(line))
        return chunks

    def _build_wiki(self) -> str:
        """Compile chunks into a structured wiki using the LLM."""
        if self.wiki_cache.exists():
            return self.wiki_cache.read_text()

        chunks = self._load_chunks()

        # Group by source for structured compilation
        by_source: dict[str, list[str]] = {}
        for c in chunks:
            src = c.get("source", "unknown")
            text = c.get("text", "")
            citation = c.get("citation", "")
            by_source.setdefault(src, []).append(f"[{citation}]\n{text}")

        # Build wiki in sections to fit context
        wiki_parts = []
        for source, texts in by_source.items():
            corpus = "\n\n---\n\n".join(texts)
            # Truncate very large sources to fit context
            if len(corpus) > 15000:
                corpus = corpus[:15000] + "\n\n[... truncated for context limits ...]"

            prompt = f"{WIKI_SYSTEM}\n\nSOURCE: {source}\n\n{corpus}"
            resp = self.llm.complete(
                prompt,
                temperature=0.0,
                max_tokens=self.wiki_max_tokens,
            )
            wiki_parts.append(f"# {source}\n\n{resp.text}")

        wiki = "\n\n---\n\n".join(wiki_parts)

        # Cache for reuse
        self.wiki_cache.parent.mkdir(parents=True, exist_ok=True)
        self.wiki_cache.write_text(wiki)
        return wiki

    def _get_wiki(self) -> str:
        if self._wiki is None:
            self._wiki = self._build_wiki()
        return self._wiki

    def answer(self, mcq: MCQ) -> Prediction:
        import time
        import re

        wiki = self._get_wiki()

        # Truncate wiki if too long for the answer prompt
        max_wiki_chars = 30000
        if len(wiki) > max_wiki_chars:
            wiki_truncated = wiki[:max_wiki_chars] + "\n\n[... see full wiki for more ...]"
        else:
            wiki_truncated = wiki

        prompt = ANSWER_SYSTEM.format(
            wiki=wiki_truncated,
            stem=mcq.stem,
            a=mcq.options["A"],
            b=mcq.options["B"],
            c=mcq.options["C"],
            d=mcq.options["D"],
        )

        t0 = time.perf_counter()
        resp = self.llm.complete(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        latency = (time.perf_counter() - t0) * 1000

        raw = resp.text
        if re.search(r"<answer>\s*[A-D]\s*$", raw, re.IGNORECASE):
            raw = raw + "</answer>"

        predicted = extract_answer(raw)
        return Prediction(
            qid=mcq.qid,
            predicted=predicted,
            correct=mcq.correct,
            is_correct=predicted == mcq.correct,
            raw_response=resp.text,
            latency_ms=latency,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            extras={"model": resp.model, "backend": resp.backend, "agent": "wiki"},
        )

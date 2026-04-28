"""
Multi-step reasoning agent with [doc, page, sentence]-level attribution.

Pipeline:  DECOMPOSE → RETRIEVE → WIKI_CHECK → REASON → VERIFY

Each step emits CitationRef(doc, page, sentence, text, similarity) so
the visual explorer can highlight exactly where in the source docs the
answer came from.

Drops into: src/ribo_agent/agents/multistep_agent.py

Usage in config YAML:
    agent: multistep
    llm: ...
    generation:
      temperature: 0.0
      max_tokens: 512
      top_k_retrieve: 5
      similarity_threshold: 0.70
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

from ..llm.base import LLMClient
from ..parsers.schema import MCQ
from .base import Prediction
from .zeroshot import extract_answer



def _retry_llm(llm, prompt, **kwargs):
    """Retry LLM calls up to 5 times on transient errors."""
    import time as _time
    for attempt in range(5):
        try:
            return llm.complete(prompt, **kwargs)
        except KeyboardInterrupt:
            raise
        except BaseException as e:
            if attempt == 4:
                raise
            wait = 10 * (attempt + 1)
            print(f"\n  [retry {attempt+1}/5] {type(e).__name__}: {e} — waiting {wait}s", flush=True)
            _time.sleep(wait)

# ── Domain types ─────────────────────────────────────────────────────

class StepType(str, Enum):
    DECOMPOSE    = "DECOMPOSE"
    RETRIEVE     = "RETRIEVE"
    WIKI_CHECK   = "WIKI_CHECK"
    REASON       = "REASON"
    VERIFY       = "VERIFY"


@dataclass
class CitationRef:
    """Atomic provenance unit: [doc, page, sentence]."""
    doc_id: str          # e.g. "OAP_2025", "RIB_Act_1990"
    doc_title: str       # human-readable
    page: int            # 1-indexed page in source PDF/doc
    sentence_idx: int    # 0-indexed sentence within the chunk
    sentence_text: str   # the actual sentence
    chunk_id: str        # back-reference to KB chunk
    section: str | None  # e.g. "s. 14", "Section 7"
    citation: str        # human label e.g. "OAP 1 Section 7 — Deductibles"
    similarity: float    # cosine sim to query (0-1)
    used_in_answer: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def ref_tag(self) -> str:
        """Short tag for inline citation: [OAP_2025, p.34, s.3]"""
        return f"[{self.doc_id}, p.{self.page}, s.{self.sentence_idx}]"


@dataclass
class StepTrace:
    """One reasoning step with its inputs, outputs, and citations."""
    step_type: StepType
    label: str
    description: str
    input_text: str
    output_text: str
    citations: list[CitationRef] = field(default_factory=list)
    duration_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["step_type"] = self.step_type.value
        return d


@dataclass
class MultiStepTrace:
    """Full pipeline trace for one question — serializable to JSON."""
    qid: str
    question_stem: str
    options: dict[str, str]
    correct: str
    predicted: str | None
    confidence: float
    is_correct: bool
    steps: list[StepTrace] = field(default_factory=list)
    all_citations: list[CitationRef] = field(default_factory=list)
    total_duration_ms: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        for s in d["steps"]:
            s["step_type"] = s["step_type"]  # already string from asdict
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ── Source document metadata ─────────────────────────────────────────

DOC_META = {
    "OAP_2025":               {"title": "Ontario Automobile Policy (OAP 1) 2025", "pages": 68},
    "RIBO_By-Law_1":          {"title": "RIBO By-Law No. 1",                      "pages": 33},
    "RIBO_By-Law_2":          {"title": "RIBO By-Law No. 2",                      "pages": 18},
    "RIBO_By-Law_3":          {"title": "RIBO By-Law No. 3",                      "pages": 12},
    "RIB_Act_1990":           {"title": "Registered Insurance Brokers Act, 1990", "pages": 42},
    "Ontario_Regulation_989": {"title": "Ontario Regulation 989",                 "pages": 16},
    "Ontario_Regulation_990": {"title": "Ontario Regulation 990",                 "pages": 24},
    "Ontario_Regulation_991": {"title": "Ontario Regulation 991",                 "pages": 20},
}


# ── BM25-style retrieval (no external deps) ─────────────────────────

def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _bm25_score(query_tokens: set[str], doc_tokens: set[str],
                citation_tokens: set[str]) -> float:
    """Simple overlap score with citation boost."""
    text_overlap = len(query_tokens & doc_tokens)
    cite_overlap = len(query_tokens & citation_tokens) * 2  # citation match = 2x
    return float(text_overlap + cite_overlap)


def _retrieve_chunks(query: str, chunks: list[dict],
                     top_k: int = 5) -> list[dict]:
    """BM25-style retrieval with similarity score normalization."""
    q_tokens = _tokenize(query)
    scored = []
    for c in chunks:
        text_tokens = _tokenize(c.get("text", ""))
        cite_tokens = _tokenize(c.get("citation", ""))
        score = _bm25_score(q_tokens, text_tokens, cite_tokens)
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)

    max_score = scored[0][0] if scored and scored[0][0] > 0 else 1.0
    results = []
    for score, c in scored[:top_k]:
        c_copy = dict(c)
        c_copy["_similarity"] = round(score / max_score, 3)
        results.append(c_copy)
    return results


# ── Sentence splitter for [doc, page, sentence] granularity ──────────

def _split_sentences(text: str) -> list[str]:
    """Split chunk text into sentences for sentence-level attribution."""
    sents = re.split(r'(?<=[.;:])\s+(?=[A-Z("])', text)
    return [s.strip() for s in sents if s.strip()]


def _estimate_page(chunk: dict) -> int:
    """Estimate page number from chunk metadata.
    
    The chunker stores section numbers. We estimate page from section
    position within the document. In production, PDF page offsets would
    be stored during ingestion.
    """
    source = chunk.get("source", "")
    section = chunk.get("section", "")
    meta = DOC_META.get(source, {"pages": 50})
    total_pages = meta["pages"]

    # Heuristic: section number roughly maps to page position
    try:
        sec_num = float(section.split("(")[0].split("-")[0]) if section else 1
    except (ValueError, IndexError):
        sec_num = 1

    # Different docs have different section densities
    if "OAP" in source:
        return max(1, min(total_pages, int(sec_num * 1.5) + 5))
    elif "Act" in source:
        return max(1, min(total_pages, int(sec_num * 0.8) + 3))
    elif "By-Law" in source:
        return max(1, min(total_pages, int(sec_num * 2) + 2))
    elif "Regulation" in source:
        return max(1, min(total_pages, int(sec_num * 0.5) + 2))
    return max(1, min(total_pages, int(sec_num)))


def _chunk_to_citations(chunk: dict, query_tokens: set[str],
                        used_in_answer: bool = False) -> list[CitationRef]:
    """Convert a KB chunk into sentence-level CitationRef entries."""
    source = chunk.get("source", "unknown")
    meta = DOC_META.get(source, {"title": source, "pages": 50})
    page = _estimate_page(chunk)
    sentences = _split_sentences(chunk.get("text", ""))

    refs = []
    for i, sent in enumerate(sentences):
        sent_tokens = _tokenize(sent)
        overlap = len(query_tokens & sent_tokens)
        max_possible = max(len(query_tokens), 1)
        sim = chunk.get("_similarity", overlap / max_possible)

        refs.append(CitationRef(
            doc_id=source,
            doc_title=meta["title"],
            page=page,
            sentence_idx=i,
            sentence_text=sent,
            chunk_id=chunk.get("chunk_id", ""),
            section=chunk.get("section"),
            citation=chunk.get("citation", ""),
            similarity=round(sim, 3),
            used_in_answer=used_in_answer,
        ))
    return refs


# ── Prompt templates ─────────────────────────────────────────────────

DECOMPOSE_PROMPT = """You are an expert at the Ontario RIBO Level 1 insurance exam.
Break this question into 2-3 sub-questions that each target a specific
retrievable fact from the study materials. Identify which document and
section is most likely being tested.

Question: {stem}
Options: A. {a} | B. {b} | C. {c} | D. {d}

Output format:
SUB-Q1: [sub-question targeting a specific fact]
SUB-Q2: [sub-question]
LIKELY_SOURCE: [which document/section this tests]
SEARCH_TERMS: [key phrases to search for in study materials]"""


REASON_PROMPT = """You are taking the Ontario RIBO Level 1 insurance broker licensing exam.
Answer based ONLY on the evidence provided. For each claim, cite the source
using [doc_id, page, sentence] tags.

EVIDENCE:
{evidence}

---

Question: {stem}

Options:
A. {a}
B. {b}
C. {c}
D. {d}

INSTRUCTIONS:
1. For each option, state whether the evidence supports or refutes it
2. Cite specific sentences: [doc_id, p.N, s.N]
3. Identify the BEST answer based on evidence
4. Flag if any option lacks evidence (potential corpus gap)

Think step by step, then give your final answer:
<answer>LETTER</answer>"""


VERIFY_PROMPT = """You are a verification agent. Check this draft answer against the evidence.

DRAFT ANSWER: {draft_answer}
DRAFT REASONING: {draft_reasoning}

EVIDENCE (same as what the reasoner saw):
{evidence}

Check for:
1. Does any evidence CONTRADICT the draft answer?
2. Is there stronger evidence for a DIFFERENT option?
3. Are the cited sources actually saying what the reasoner claims?
4. Is this a known exam trap? (e.g., Act vs By-Law delegation, $500 collision vs $300 comprehensive)

If the draft is correct, respond: VERIFIED [letter]
If it should change, respond: OVERRIDE [new_letter] — [reason]
Then: <answer>LETTER</answer>"""


WIKI_CHECK_PROMPT = """Given this exam question, check the study wiki for:
1. Direct answers or definitions
2. Known exam traps (common wrong answers)
3. Cross-references between documents
4. Negative knowledge: things that are NOT true but commonly believed

Question: {stem}
Options: A. {a} | B. {b} | C. {c} | D. {d}

WIKI EXCERPT:
{wiki}

Respond with:
FINDINGS: [what the wiki says about this topic]
TRAPS: [any exam traps identified]
NEGATIVE: [things that are NOT true that might appear as distractors]"""


# ── The agent ────────────────────────────────────────────────────────

class MultiStepAgent:
    """Multi-step reasoning agent with full [doc, page, sentence] attribution."""

    def __init__(
        self,
        llm: LLMClient,
        kb_path: Path | None = None,
        wiki_cache_path: Path | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_k_retrieve: int = 5,
        similarity_threshold: float = 0.70,
        wiki_max_tokens: int = 4096,
        enable_voting: bool = False,
        vote_samples: int = 3,
    ) -> None:
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k_retrieve
        self.sim_threshold = similarity_threshold
        self.wiki_max_tokens = wiki_max_tokens
        self.enable_voting = enable_voting
        self.vote_samples = vote_samples

        root = Path(__file__).resolve().parents[3]
        self.kb_path = kb_path or root / "data" / "kb" / "chunks.jsonl"
        self.wiki_cache = wiki_cache_path or root / "data" / "kb" / "wiki_compiled.md"

        self._chunks: list[dict] | None = None
        self._wiki: str | None = None

    def _get_chunks(self) -> list[dict]:
        if self._chunks is None:
            if self.kb_path.exists():
                self._chunks = [json.loads(l) for l in self.kb_path.open()]
            else:
                self._chunks = []
        return self._chunks

    def _get_wiki(self) -> str:
        if self._wiki is None:
            if self.wiki_cache.exists():
                self._wiki = self.wiki_cache.read_text()
            else:
                self._wiki = ""
        return self._wiki

    def _step_decompose(self, mcq: MCQ) -> StepTrace:
        t0 = time.perf_counter()
        prompt = DECOMPOSE_PROMPT.format(
            stem=mcq.stem,
            a=mcq.options["A"], b=mcq.options["B"],
            c=mcq.options["C"], d=mcq.options["D"],
        )
        resp = _retry_llm(self.llm, prompt, temperature=0.0, max_tokens=300)
        duration = (time.perf_counter() - t0) * 1000

        # Extract search terms from decomposition
        search_terms = ""
        for line in resp.text.splitlines():
            if line.strip().startswith("SEARCH_TERMS:"):
                search_terms = line.split(":", 1)[1].strip()

        return StepTrace(
            step_type=StepType.DECOMPOSE,
            label="Question Decomposition",
            description="Break MCQ into sub-questions targeting specific retrievable facts.",
            input_text=mcq.stem,
            output_text=resp.text.strip(),
            citations=[],
            duration_ms=duration,
            metadata={"search_terms": search_terms},
        )

    def _step_retrieve(self, mcq: MCQ, search_terms: str) -> StepTrace:
        t0 = time.perf_counter()
        chunks = self._get_chunks()

        # Build query from stem + search terms
        query = f"{mcq.stem} {search_terms}"
        q_tokens = _tokenize(query)

        # Retrieve
        retrieved = _retrieve_chunks(query, chunks, top_k=self.top_k)

        # Convert to sentence-level citations
        all_citations: list[CitationRef] = []
        output_lines = []
        for rank, chunk in enumerate(retrieved, 1):
            refs = _chunk_to_citations(chunk, q_tokens)
            all_citations.extend(refs)
            output_lines.append(
                f"[Rank {rank}] {chunk.get('citation', '?')} "
                f"(sim={chunk.get('_similarity', 0):.2f}) — "
                f"{len(refs)} sentences"
            )

        duration = (time.perf_counter() - t0) * 1000
        return StepTrace(
            step_type=StepType.RETRIEVE,
            label="Primary Retrieval (KB Chunks)",
            description=f"BM25 search over {len(chunks)} study chunks. "
                        f"Retrieved top-{self.top_k}.",
            input_text=query,
            output_text="\n".join(output_lines),
            citations=all_citations,
            duration_ms=duration,
            metadata={"n_chunks_searched": len(chunks),
                       "n_retrieved": len(retrieved)},
        )

    def _step_wiki_check(self, mcq: MCQ) -> StepTrace:
        t0 = time.perf_counter()
        wiki = self._get_wiki()

        if not wiki:
            return StepTrace(
                step_type=StepType.WIKI_CHECK,
                label="Wiki Verification",
                description="No compiled wiki available — skipped.",
                input_text="", output_text="SKIPPED: no wiki",
                duration_ms=0,
            )

        # Truncate wiki for prompt
        wiki_excerpt = wiki[:15000] if len(wiki) > 15000 else wiki

        prompt = WIKI_CHECK_PROMPT.format(
            stem=mcq.stem,
            a=mcq.options["A"], b=mcq.options["B"],
            c=mcq.options["C"], d=mcq.options["D"],
            wiki=wiki_excerpt,
        )
        resp = _retry_llm(self.llm, prompt, temperature=0.0, max_tokens=400)
        duration = (time.perf_counter() - t0) * 1000

        # Create wiki citations (synthetic — wiki doesn't have pages)
        wiki_citations = []
        for i, line in enumerate(resp.text.splitlines()):
            if line.strip() and not line.startswith(("FINDINGS:", "TRAPS:", "NEGATIVE:")):
                wiki_citations.append(CitationRef(
                    doc_id="WIKI",
                    doc_title="LLM-Compiled Knowledge Wiki",
                    page=1, sentence_idx=i,
                    sentence_text=line.strip(),
                    chunk_id="wiki", section=None,
                    citation="Wiki (compiled)",
                    similarity=0.85,
                    used_in_answer=False,
                ))

        return StepTrace(
            step_type=StepType.WIKI_CHECK,
            label="Wiki Verification & Trap Detection",
            description="Check compiled wiki for exam traps, negative knowledge, cross-references.",
            input_text=mcq.stem,
            output_text=resp.text.strip(),
            citations=wiki_citations,
            duration_ms=duration,
        )

    def _step_reason(self, mcq: MCQ, retrieve_step: StepTrace,
                     wiki_step: StepTrace) -> StepTrace:
        t0 = time.perf_counter()

        # Build evidence block from citations
        evidence_lines = []
        for ref in retrieve_step.citations:
            tag = ref.ref_tag()
            evidence_lines.append(f"{tag} [{ref.citation}]: {ref.sentence_text}")

        # Add wiki findings
        if wiki_step.output_text and wiki_step.output_text != "SKIPPED: no wiki":
            evidence_lines.append("\n--- WIKI FINDINGS ---")
            evidence_lines.append(wiki_step.output_text)

        evidence = "\n".join(evidence_lines)

        prompt = REASON_PROMPT.format(
            evidence=evidence, stem=mcq.stem,
            a=mcq.options["A"], b=mcq.options["B"],
            c=mcq.options["C"], d=mcq.options["D"],
        )
        resp = _retry_llm(self.llm, prompt, temperature=self.temperature,
                                 max_tokens=self.max_tokens)
        duration = (time.perf_counter() - t0) * 1000

        raw = resp.text
        if re.search(r"<answer>\s*[A-D]\s*$", raw, re.IGNORECASE):
            raw += "</answer>"

        # Parse which citations were actually used in the reasoning
        used_refs = set()
        for match in re.finditer(r"\[(\w+),\s*p\.(\d+),\s*s\.(\d+)\]", raw):
            used_refs.add((match.group(1), int(match.group(2)), int(match.group(3))))

        # Mark citations that were used
        cited = []
        for ref in retrieve_step.citations:
            ref_copy = CitationRef(**{k: v for k, v in asdict(ref).items()})
            if (ref.doc_id, ref.page, ref.sentence_idx) in used_refs:
                ref_copy.used_in_answer = True
            cited.append(ref_copy)

        draft_answer = extract_answer(raw)

        return StepTrace(
            step_type=StepType.REASON,
            label="Multi-Step Reasoning",
            description="LLM reasons over retrieved evidence with inline [doc, page, sentence] citations.",
            input_text=f"{len(retrieve_step.citations)} evidence sentences",
            output_text=raw.strip(),
            citations=cited,
            duration_ms=duration,
            metadata={"draft_answer": draft_answer or "",
                       "n_citations_used": len(used_refs)},
        )

    def _step_verify(self, mcq: MCQ, reason_step: StepTrace,
                     retrieve_step: StepTrace) -> StepTrace:
        t0 = time.perf_counter()
        draft = reason_step.metadata.get("draft_answer", "")
        draft_reasoning = reason_step.output_text[:2000]

        evidence_lines = []
        for ref in retrieve_step.citations:
            tag = ref.ref_tag()
            evidence_lines.append(f"{tag} [{ref.citation}]: {ref.sentence_text}")
        evidence = "\n".join(evidence_lines)

        prompt = VERIFY_PROMPT.format(
            draft_answer=draft,
            draft_reasoning=draft_reasoning,
            evidence=evidence,
        )
        resp = _retry_llm(self.llm, prompt, temperature=0.0, max_tokens=300)
        duration = (time.perf_counter() - t0) * 1000

        raw = resp.text
        if re.search(r"<answer>\s*[A-D]\s*$", raw, re.IGNORECASE):
            raw += "</answer>"

        verified_answer = extract_answer(raw)
        is_override = "OVERRIDE" in raw.upper()

        return StepTrace(
            step_type=StepType.VERIFY,
            label="Critic / Contradiction Check",
            description="Second LLM pass verifies draft against evidence. Checks for contradictions and exam traps.",
            input_text=f"Draft={draft}",
            output_text=raw.strip(),
            citations=reason_step.citations,  # reuse — verifier checks same evidence
            duration_ms=duration,
            metadata={"verified_answer": verified_answer or draft,
                       "is_override": is_override,
                       "original_draft": draft},
        )

    def answer(self, mcq: MCQ) -> Prediction:
        """Run full multi-step pipeline and return Prediction with trace."""
        t0_total = time.perf_counter()

        # Step 1: Decompose
        decompose = self._step_decompose(mcq)
        search_terms = decompose.metadata.get("search_terms", "")

        # Step 2: Retrieve from KB
        retrieve = self._step_retrieve(mcq, search_terms)

        # Step 3: Wiki check
        wiki_check = self._step_wiki_check(mcq)

        # Step 4: Reason
        reason = self._step_reason(mcq, retrieve, wiki_check)

        # Step 5: Verify
        verify = self._step_verify(mcq, reason, retrieve)

        # Determine final answer
        final_answer = verify.metadata.get("verified_answer",
                         reason.metadata.get("draft_answer", ""))

        # Optional: majority voting
        if self.enable_voting and final_answer:
            votes = Counter([final_answer])
            for _ in range(self.vote_samples - 1):
                r = self._step_reason(mcq, retrieve, wiki_check)
                ans = r.metadata.get("draft_answer", "")
                if ans:
                    votes[ans] += 1
            final_answer = votes.most_common(1)[0][0]

        total_ms = (time.perf_counter() - t0_total) * 1000

        # Collect all unique citations
        all_citations = []
        seen = set()
        for step in [retrieve, wiki_check, reason, verify]:
            for ref in step.citations:
                key = (ref.doc_id, ref.page, ref.sentence_idx)
                if key not in seen:
                    seen.add(key)
                    all_citations.append(ref)

        # Compute confidence from evidence convergence
        n_used = sum(1 for r in all_citations if r.used_in_answer)
        n_total = max(len(all_citations), 1)
        confidence = min(0.99, 0.5 + (n_used / n_total) * 0.5)
        if verify.metadata.get("is_override"):
            confidence *= 0.85  # reduce confidence on overrides

        # Build full trace
        trace = MultiStepTrace(
            qid=mcq.qid,
            question_stem=mcq.stem,
            options=mcq.options,
            correct=mcq.correct,
            predicted=final_answer if final_answer in "ABCD" else None,
            confidence=round(confidence, 3),
            is_correct=(final_answer == mcq.correct),
            steps=[decompose, retrieve, wiki_check, reason, verify],
            all_citations=all_citations,
            total_duration_ms=round(total_ms, 1),
        )

        return Prediction(
            qid=mcq.qid,
            predicted=final_answer if final_answer in "ABCD" else None,
            correct=mcq.correct,
            is_correct=(final_answer == mcq.correct),
            raw_response=reason.output_text[:5000],
            latency_ms=total_ms,
            extras={
                "agent": "multistep",
                "trace": trace.to_dict(),
                "confidence": trace.confidence,
                "n_citations": len(all_citations),
                "n_citations_used": n_used,
                "steps_completed": len(trace.steps),
            },
        )

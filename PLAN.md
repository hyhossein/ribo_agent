# 7-day plan

One commit-heavy chunk of work per day. Each day ships something
self-contained with tests and benchmarks where applicable.

## Day 1 — EDA

Understand the inputs before writing a single line of agent code.

- Inventory both zips, file types, page counts.
- Classify the question PDFs by structure: which ones are graded MCQs with
  known answers, which are study aids, which are something else.
- Characterize question length, option length, and any metadata fields
  (content domain, competency, cognitive level).
- Skim each study document and sketch its section structure — which fields
  would be worth indexing.
- Check for question leakage between the "answer-key" question sets and the
  "manual" question sets.
- Write up findings in `notebooks/day1_eda.ipynb` + `docs/EDA.md`.

**Out of scope for today:** parsing into JSONL, embeddings, any LLM call.

## Day 2 — Parsers and eval set

Turn the questions into clean, versioned JSONL. Two parsers:

1. `Sample-Questions-RIBO-Level-1-Exam.pdf` — straightforward (inline answers
   + rich metadata).
2. `695993459-Practise-RIBO-Exam.pdf` — harder (X-grid answer key whose
   column positions change between pages).
3. Manual MCQ extraction with PyMuPDF font-flag detection (the bolded
   option is the answer).

Unit tests per parser. Dedup via stem fingerprints. Confirm zero leakage
between eval set and few-shot pool.

## Day 3 — KB ingestion & chunking  ✓ shipped as v0.3.0

Turn the raw study corpus into retrieval-ready chunks:

- LibreOffice headless converts the three `.doc` regulations + the
  `.doc` RIB Act to UTF-8 text. SHA-keyed cache so reruns skip the
  slow conversion.
- Four per-format splitters: regulations (`1.` / `5.1` / `7.2`), RIB
  Act (title-on-prior-line), RIBO By-Laws (ARTICLE + x.y), OAP
  (`Section N - Title`).
- Size normaliser merges tiny neighbours and splits oversize blocks
  on paragraph → sentence → character with small overlap.
- 298 chunks across 8 sources. Every chunk carries a citation string
  suitable for quoting in an answer.
- Manual-PDF parser added alongside: 386 MCQs extracted via PyMuPDF
  bold-font detection (Calibri-Bold / flag bit 4). Dedup +
  fingerprint leakage check prove 0 overlap with the 169-Q eval set.

Tests: 64 total (was 47). New ones cover bold-answer detection
fingerprint stability, per-format splitter invariants, chunk size
bounds.

## Day 4 — Embeddings & retrieval eval

BGE-base-en-v1.5 embeddings. FAISS index. A retrieval-quality eval asking:
for questions that cite a specific section, does top-k retrieval find it?
Report recall@1/@5/@20, latency, memory.

## Day 5 — Agent v0 and v1

- **v0:** Claude Sonnet 4.6, zero-shot, no retrieval. Establishes the floor.
- **v1:** + RAG over study docs.

Eval harness scores the full 169-question held-out set. JSON traces per run
for error analysis.

## Day 6 — v2, v3 + CI

- **v2:** + few-shot exemplars retrieved from the manual question pool.
- **v3:** + self-consistency (N samples, majority vote) and/or a verifier
  pass cross-checking against the cited statute.

GitHub Actions: `pytest`, parser-checksum check, small smoke eval.

## Day 7 — Report, profiling, error analysis

- Design doc explaining every architectural choice.
- Profile the winning config (latency, token cost, $/question).
- Manual review of every missed question — what's the failure mode?
  (retrieval? reasoning? ambiguous question?)

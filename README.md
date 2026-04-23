# RIBO Level 1 Exam Agent

An AI agent that answers multiple-choice questions from the Ontario
**Registered Insurance Brokers of Ontario (RIBO) Level 1** licensing exam.

## The task

Ontario brokers must pass the RIBO exam to become licensed. The exam tests
knowledge of the RIB Act, Ontario Regulations 989/990/991, the RIBO By-Laws,
and the Ontario Automobile Policy (OAP 1), plus general insurance concepts
such as subrogation, indemnity, coverage forms, and the like.

The take-home: build an agent that gets as many questions right as possible
on a held-out set, using:

- **Study corpus** (`data/raw/study/`) — RIB Act, Ontario Regulations, three
  RIBO By-Laws, OAP 2025.
- **Question corpus** (`data/raw/questions/`) — two official RIBO sample
  papers (with answer keys) plus two larger RIBO licensing-manual question
  sets where the correct option is marked in **bold** type.

## Repo status

This repo is being built up over the course of the week. The commit history
is the story; each day adds a focused chunk of work with its own tests and
benchmarks. See [`PLAN.md`](./PLAN.md) for the 7-day breakdown.

| Day | Focus | Status |
| :-- | :---- | :----- |
| 1   | EDA — understand the data before writing any system | ✅ done |
| 2   | Parsers & eval set | ✅ done (169 MCQs, 35 tests) |
| 3   | Knowledge-base ingestion & chunking | — |
| 4   | Embeddings + retrieval eval | — |
| 5   | Zero-shot baseline (v0) and simple RAG (v1) | — |
| 6   | v2/v3 agents (few-shot, domain routing, self-consistency) + CI | — |
| 7   | Final report, profiling, error analysis | — |

## Layout

```
data/
  raw/
    questions/     # original PDFs, untouched
    study/         # original study documents, untouched
notebooks/         # EDA and analysis (one per day)
src/
  ribo_agent/      # library code (parsers, agents, retrieval)
tests/             # unit + integration tests
configs/           # agent configs
results/           # eval results and reports
```

## Running locally

Requires `poppler-utils` and Python 3.11+. Inside an activated conda env:

```bash
make install       # pip install -e . plus declared deps
make parse         # parse both PDFs -> data/parsed/*.jsonl
make test          # run pytest (currently 35 tests, ~0.4s)
```

After `make parse` you will have:

```
data/parsed/sample_questions.jsonl    # 79 MCQs with Content Domain metadata
data/parsed/practice_exam.jsonl       # 90 MCQs from the X-grid-keyed PDF
```

Day 3 adds the manual extractor and the study-doc chunker.

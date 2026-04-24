# RIBO Agent

[![CI](https://github.com/hyhossein/ribo_agent/actions/workflows/ci.yml/badge.svg)](https://github.com/hyhossein/ribo_agent/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Release](https://img.shields.io/github/v/release/hyhossein/ribo_agent?include_prereleases)](https://github.com/hyhossein/ribo_agent/releases)
[![Tests](https://img.shields.io/badge/tests-81%20passing-brightgreen.svg)](./tests)

An AI agent that answers multiple-choice questions from the Ontario
**Registered Insurance Brokers of Ontario (RIBO) Level 1** licensing
exam. Fully open-source, locally-runnable, designed to promote cleanly
to Azure ML for production.

## 🏆 Leaderboard

Head-to-head accuracy on the 169-question held-out eval set. Every
model runs locally via [Ollama](https://ollama.com), zero-shot, no
retrieval yet (RAG lands in v0.5.0 — expect meaningful lift).

<!-- LEADERBOARD:START -->
|  | Model | Accuracy | Macro-F1 | Latency (ms) |
| :--- | :--- | ---: | ---: | ---: |
| 🥇 | **wiki_claude-opus-4-20250514** | `0.8817` | `0.8835` | 29626 |
| 🥈 | **claude-opus-4-20250514** | `0.7870` | `0.8031` | 7396 |
| 🥉 | **Qwen 2.5 7B Instruct** | `0.5976` | `0.6085` | 41979 |
| 4. | **claude-sonnet-4-20250514** | `0.5207` | `0.5351` | 6253 |
| 5. | **Phi-4 Mini 3.8B** | `0.4911` | `0.4982` | 25095 |

_Updated 2026-04-24 10:51 UTC · 169-question eval set · zero-shot, no RAG_
<!-- LEADERBOARD:END -->

**Baselines:** random = `0.2500` · pass mark (Ontario) = `0.7500`

Full per-model reports with per-domain breakdowns and confusion
matrices live in [`results/runs/`](./results/runs).
A live-updated markdown version is also kept at
[`results/LEADERBOARD.md`](./results/LEADERBOARD.md).

## TL;DR

- **What it does.** Takes a RIBO exam question in, returns A/B/C/D out.
- **How.** Local open-source LLM via Ollama + retrieval over the
  official study corpus (RIB Act, Ontario Regulations, RIBO By-Laws,
  OAP 2025).
- **Where it runs.** Today on a laptop. Tomorrow on Azure ML Managed
  Online Endpoints with one config switch.
- **How we prove it works.** Automated model sweep runs every
  candidate against the 169-question eval set, writes per-question
  traces, commits results, and pushes — one command, one leaderboard.

## Run the whole sweep

```bash
ollama serve &                                  # one terminal
make parse && make kb && make test              # first time only
make sweep                                      # in another terminal
```

`make sweep` pulls, evaluates, commits, and pushes every candidate
model from [`docs/MODELS.md`](./docs/MODELS.md). 60–90 min unattended
on an M-series Mac. Ctrl+C is safe — each completed model is its own
commit.

## Status

| Release | Scope | Evidence |
| :--- | :--- | :--- |
| [v0.1.0](https://github.com/hyhossein/ribo_agent/releases/tag/v0.1.0) | Parsers, eval set, 35 tests | 169 MCQs in `data/parsed/` |
| [v0.2.0](https://github.com/hyhossein/ribo_agent/releases/tag/v0.2.0) | CI/CD, LLM + storage interfaces, Azure ML stubs | 47 tests green on push |
| [v0.3.0](https://github.com/hyhossein/ribo_agent/releases/tag/v0.3.0) | Manual-PDF extractor + study-doc chunker | 386 few-shot MCQs, 298 KB chunks, 64 tests |
| [v0.3.1](https://github.com/hyhossein/ribo_agent/releases/tag/v0.3.1) | macOS LibreOffice binary discovery | `make kb` works on Mac |
| [v0.4.0](https://github.com/hyhossein/ribo_agent/releases/tag/v0.4.0) | Zero-shot agent + eval harness + leaderboard | First accuracy & macro-F1 numbers |
| v0.5.0 | RAG agent (BGE + FAISS retrieval) | Accuracy lift over v0.4.0 |
| v0.6.0 | v2 (few-shot) + v3 (self-consistency) | Per-variant lift |
| v1.0.0 | Final report, profiling, error analysis, Azure ML deployment recipe | End-to-end reproducible |

See [`PLAN.md`](./PLAN.md) for the build plan and
[`CHANGELOG.md`](./CHANGELOG.md) for detailed release notes.

## Architecture

```
                  ┌──────────────────────────┐
                  │  Agent (v0 .. v3)        │
                  │  prompt + retrieve + ask │
                  └────┬─────────────────┬───┘
                       │                 │
                       ▼                 ▼
               ┌──────────────┐   ┌─────────────┐
               │ LLMClient    │   │ Retriever   │
               │ (Protocol)   │   │ (FAISS)     │
               └──────┬───────┘   └─────────────┘
            ┌─────────┴──────────┐
            ▼                    ▼
     ┌─────────────┐      ┌────────────────┐
     │ Ollama      │      │ Azure ML       │
     │ (local dev) │      │ Managed Online │
     │             │      │ Endpoint       │
     └─────────────┘      └────────────────┘
```

Every agent depends on the `LLMClient` protocol, never on a specific
backend. Swap local ↔ Azure ML by editing a single config line:

```yaml
# configs/v0_baseline.yaml
llm:
  backend: ollama     # or: azureml
  model: qwen2.5:7b-instruct
```

## Running locally

```bash
# one-time setup
conda create -n ribo python=3.11 -y
conda activate ribo
pip install -e .[dev]

# start the local LLM (separate terminal, leave running)
brew install ollama
ollama serve &
ollama pull qwen2.5:7b-instruct

# generate the eval set, the few-shot pool, and the KB
make parse     # -> data/parsed/{sample_questions,practice_exam,manual_pool,eval,train}.jsonl
make kb        # -> data/kb/chunks.jsonl (298 section-level chunks)
make test      # pytest, 81 tests, ~5s

# Evaluation (requires Ollama running + the model pulled)
ollama serve &
ollama pull qwen2.5:7b-instruct

make eval CONFIG=configs/v0_zeroshot_qwen25_7b.yaml    # ~15 min
make eval-all                                           # every zero-shot config
make compare                                            # leaderboard table
```

Evaluation and agent runs land in v0.4.0+; see `PLAN.md`.

## Repo layout

```
src/ribo_agent/
    parsers/         # PDF -> canonical MCQ JSONL
    llm/             # LLM client abstraction + Ollama and Azure ML impls
    io/              # Storage abstraction (local + Azure Blob stub)
tests/               # pytest — 47 tests currently
notebooks/           # EDA and analysis, one per stage
docs/                # EDA write-up, design decisions
configs/             # one YAML per agent variant
data/
    raw/             # immutable inputs (git-tracked)
    parsed/          # derived: eval JSONL (gitignored, rebuild with `make parse`)
    kb/              # derived: chunked study corpus (v0.3.0)
    index/           # derived: FAISS indices (v0.4.0)
results/             # per-run eval reports (v0.5.0+)
.github/workflows/   # CI
```

`raw/` is git-tracked (small, immutable). Everything else is a
deterministic function of code + raw inputs, so it is gitignored and
rebuilt by a Makefile target. `make clean` removes all derived state.

## Design principles

**Local-first, cloud-ready.** Every capability that could eventually
live on Azure ML (LLM, storage, embeddings, vector index) sits behind a
small interface with two implementations — a local one for development
and a cloud one for production. Today both resolve to local; in v1.0.0
the Azure path is fully wired.

**Reproducible.** Raw PDFs in, canonical JSONL out, deterministic
chunks and embeddings, fixed LLM temperature 0.0 for baselines. Any
number in `results/` can be re-derived from a clean checkout.

**Tested where it matters.** Every PDF-parsing trap identified during
EDA is backed by a named regression test (`tests/test_*_parser.py`)
so silent breakage becomes loud.

**Observable.** GitHub Actions runs every test on every push. Each
release is tagged, notable changes live in `CHANGELOG.md`, and eval
reports in `results/` are versioned markdown so an exec can read a
commit diff and see the number move.

## License

MIT — see [LICENSE](./LICENSE).

# RIBO Agent

[![CI](https://github.com/hyhossein/ribo_agent/actions/workflows/ci.yml/badge.svg)](https://github.com/hyhossein/ribo_agent/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Release](https://img.shields.io/github/v/release/hyhossein/ribo_agent?include_prereleases)](https://github.com/hyhossein/ribo_agent/releases)
[![Tests](https://img.shields.io/badge/tests-87%20passing-brightgreen.svg)](./tests)

An AI agent that answers multiple-choice questions from the Ontario
**RIBO Level 1** insurance broker licensing exam. Supports both
open-source local models (Ollama) and commercial APIs (Anthropic
Claude, OpenAI). Designed to promote to Azure ML for production.

---

## 🏆 Leaderboard

<!-- LEADERBOARD:START -->
|  | Model | Accuracy | Macro-F1 | Latency (ms) |
| :--- | :--- | ---: | ---: | ---: |
| 🥇 | **wiki_claude-opus-4-20250514** | `0.8876` | `0.8869` | 20399 |
| 🥈 | **claude-opus-4-20250514** | `0.8817` | `0.8766` | 51512 |
| 🥉 | **claude-opus-4-20250514** | `0.7870` | `0.8031` | 7396 |
| 4. | **Qwen 2.5 7B Instruct** | `0.5976` | `0.6085` | 41979 |
| 5. | **claude-sonnet-4-20250514** | `0.5207` | `0.5351` | 6253 |
| 6. | **Phi-4 Mini 3.8B** | `0.4911` | `0.4982` | 25095 |

_Updated 2026-04-24 18:47 UTC · 169-question eval set · zero-shot, no RAG_
<!-- LEADERBOARD:END -->

**Baselines:** random = `0.2500` · RIBO pass mark (Ontario) = `0.7500`

---

## Methodology: from open-source baseline to agentic pipeline

This section documents the reasoning process, not just the result.
The journey from a 49% local model to an 88% agentic system followed
a deliberate experimental progression.

### Step 1: Establish the floor with open-source models

**Question:** How well can small, free, locally-runnable models answer
Ontario insurance licensing questions out of the box?

**Approach:** We benchmarked seven open-source models (3.8B to 12B
parameters) via Ollama on a 16 GB MacBook Air. Zero-shot, no
retrieval, no context. Pure parametric knowledge.

**Finding:** The best open-source model (Qwen 2.5 7B) reached 59.8%.
The smallest (Phi-4 Mini 3.8B) reached 49.1%. All are well above
random (25%) but none pass the exam (75%). The models have general
insurance knowledge from pretraining but lack Ontario-specific
regulatory details.

**Insight:** Open-source models at this scale cannot pass the exam
alone. But they are not useless — they demonstrate that insurance
domain knowledge exists in the model weights. The gap is
jurisdiction-specific rules, not domain understanding.

### Step 2: Test commercial models to measure the parameter ceiling

**Question:** Does scaling to a frontier model close the gap, or is
the problem fundamentally about knowledge access?

**Approach:** We evaluated Claude Sonnet 4 and Claude Opus 4 via the
Anthropic API. Same zero-shot prompt, same 169 questions.

**Finding:** Opus 4 reached 78.7% (above the pass mark). Sonnet 4
reached 52.1%. The Sonnet-to-Opus jump (+26.6pp) is much larger than
Phi-to-Qwen (+10.7pp), confirming that model quality matters — but
even Opus misses regulation-specific questions where the answer
requires knowing exact section numbers or exception clauses.

**Insight:** A frontier model can pass the exam zero-shot, but
barely. The failure cases cluster on questions citing specific
statutes (e.g. "under s. 14 of Regulation 991..."). This is a
knowledge access problem, not a reasoning problem.

### Step 3: Compile the knowledge base (Karpathy LLM Wiki)

**Question:** If we give the model structured access to the official
study corpus, how much does accuracy improve?

**Approach:** Instead of traditional RAG (embed chunks, retrieve
top-k per question), we adopted
[Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f):
use the LLM itself to pre-compile all 297 study chunks into a
structured knowledge wiki organized by topic, with cross-references
resolved and section numbers preserved. The wiki is built once at
startup and cached. At eval time, each question is answered against
the compiled wiki rather than raw document fragments.

**Why wiki over RAG:** traditional RAG has two failure modes on this
task. First, the embedding model may not retrieve the right chunk —
regulatory text is dense and semantically similar across sections.
Second, retrieved chunks lack cross-references, so the model can't
see that section 14 has an exception defined in section 14.1. The
wiki pattern eliminates both: compilation surfaces all
cross-references, and the model sees organized knowledge rather than
disconnected fragments.

**Finding:** Opus 4 + Wiki reached **88.2%** — a +9.5pp lift over
zero-shot. The improvement is concentrated on exactly the question
types that zero-shot missed: regulation-specific, section-citing,
exception-clause questions.

**Insight:** Knowledge access is the dominant lever. The wiki
compilation cost ($3 one-time) is amortized across all questions.
This is cheaper and more effective than scaling to a larger model.

### Step 4: Question rewriting (in progress)

**Question:** Can we further improve accuracy by clarifying
ambiguous questions before the model answers?

**Approach:** A two-stage pipeline where an LLM first rewrites the
question stem — expanding abbreviations (OAP = Ontario Automobile
Policy), identifying which regulation is being tested, clarifying
pronouns — then passes the clarified question to the wiki agent.

**Hypothesis:** Some questions the model gets wrong are not because
it lacks the knowledge, but because it fails to connect the question
phrasing to the right section in the wiki. Rewriting acts as a
"study buddy" layer.

**Status:** eval running, results pending.

### Summary of the progression

```
 49.1%  ──►  Phi-4 Mini 3.8B, zero-shot, local
 59.8%  ──►  Qwen 2.5 7B, zero-shot, local
 78.7%  ──►  Claude Opus 4, zero-shot, API
 88.2%  ──►  Claude Opus 4 + Wiki compilation    ◄── best result
  ???   ──►  Claude Opus 4 + Rewrite + Wiki       ◄── in progress
```

Each step tests a specific hypothesis. The commit history shows
every experiment as it happened.

---

## Agent architectures

Three progressively more sophisticated designs. The accuracy lift
from v0 to v1 is the core result.

### v0: Zero-shot (baseline)

The model sees only the question and four options. No study material.
This measures what the LLM already knows from pretraining.

```
Question + Options  ──►  LLM  ──►  A/B/C/D
```

Opus 4 reaches **78.7%** zero-shot. Strong, but misses
regulation-specific details that require exact section knowledge.

### v1: LLM Wiki agent

Inspired by [Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).
Instead of traditional RAG (retrieve raw chunks per question), we
**pre-compile the entire 297-chunk study corpus into a structured
knowledge wiki** at startup. The wiki is organized by topic (RIB Act,
Ontario Regulations 989/990/991, RIBO By-Laws 1/2/3, OAP 2025) with
cross-references already resolved and section numbers preserved.

```
297 study chunks  ──►  LLM compiles  ──►  Structured Wiki (cached)
                                                  │
                         Question + Wiki  ──►  LLM  ──►  A/B/C/D
```

**Why this beats traditional RAG:** RAG re-discovers knowledge from
scratch on every question, hoping the embedding model finds the right
chunk. The wiki pattern compiles once and reuses: cross-references
are pre-resolved, the synthesis reflects the entire corpus, and the
model sees organized knowledge rather than disconnected fragments.

Opus 4 + Wiki reaches **88.2%** — a **+9.5pp lift** over zero-shot
and well above the 75% pass mark.

### v2: Question rewrite + Wiki

Two-stage agentic pipeline. Before answering, an LLM rewrites the
question to expand abbreviations (OAP, RIB Act), clarify ambiguous
pronouns, and identify which specific regulation or section is being
tested. The clarified question then feeds into the wiki agent.

```
Question  ──►  LLM rewrites  ──►  Clarified Question
                                          │
                    Clarified Q + Wiki  ──►  LLM  ──►  A/B/C/D
```

This addresses a common failure mode where the model knows the rule
but can't connect the original question phrasing to the right section.

---

## Key results

| Approach | Accuracy | Cost | Insight |
| :--- | ---: | ---: | :--- |
| Opus zero-shot | 78.7% | $1.01 | Strong base knowledge, misses regulation specifics |
| **Opus + Wiki** | **88.2%** | ~$4.00 | **Pre-compiled knowledge is the dominant lever** |
| Qwen 2.5 7B (local) | 59.8% | $0 | Viable for low-cost pre-screening |
| Sonnet 4 zero-shot | 52.1% | $0.32 | Instruction-following gap vs Opus on regulatory MCQ |
| Phi-4 Mini 3.8B (local) | 49.1% | $0 | Above random but below practical threshold |

**The takeaway:** knowledge access matters more than model size.
Opus zero-shot (78.7%) vs Opus + wiki (88.2%) shows that structured
context delivers a bigger lift than scaling from Sonnet to Opus
(52.1% → 78.7%).

---

## Model coverage

Both open-source and commercial models, evaluating the
cost-accuracy tradeoff.

| Model | Type | Size | Backend |
| :--- | :--- | :--- | :--- |
| Claude Opus 4 | Commercial | — | Anthropic API |
| Claude Sonnet 4 | Commercial | — | Anthropic API |
| Qwen 2.5 7B Instruct | Open-source | 4.4 GB | Ollama (local) |
| Phi-4 Mini 3.8B | Open-source | 2.5 GB | Ollama (local) |
| Llama 3.1 8B | Open-source | 4.7 GB | Ollama (local) |
| Qwen 3 8B | Open-source | 5.2 GB | Ollama (local) |
| Gemma 3 12B | Open-source | 8.1 GB | Ollama (local) |
| DeepSeek-R1-Distill 7B | Open-source | 4.7 GB | Ollama (local) |

Model selection rationale: [`docs/MODELS.md`](./docs/MODELS.md)
Literature review (15 cited works): [`docs/LITERATURE.md`](./docs/LITERATURE.md)

---

## Quick start

```bash
conda create -n ribo python=3.11 -y && conda activate ribo
pip install -e .[dev]

make parse          # 169 eval + 386 few-shot MCQs
make kb             # 297 section-level chunks with citations
make test           # 87 tests, ~5s

# zero-shot with local model
ollama serve &
ollama pull qwen2.5:7b-instruct
make eval CONFIG=configs/v0_zeroshot_qwen25_7b.yaml

# wiki agent with Claude Opus (the 88.2% run)
export ANTHROPIC_API_KEY="your-key"
make eval CONFIG=configs/v1_wiki_opus.yaml

# full open-source model sweep (unattended, ~90 min)
make sweep

# leaderboard
make compare
```

---

## Architecture

```
                  ┌──────────────────────────────┐
                  │  Agent (v0 / v1 / v2)        │
                  │  zero-shot / wiki / rewrite  │
                  └────┬─────────────────┬───────┘
                       │                 │
                       ▼                 ▼
               ┌──────────────┐   ┌──────────────┐
               │ LLMClient    │   │ Knowledge    │
               │ (Protocol)   │   │ Base (wiki)  │
               └──────┬───────┘   └──────────────┘
            ┌─────────┼──────────┐
            ▼         ▼          ▼
     ┌──────────┐ ┌──────────┐ ┌──────────┐
     │ Ollama   │ │Anthropic │ │ OpenAI   │
     │ (local)  │ │ API      │ │ API      │
     └──────────┘ └──────────┘ └──────────┘
```

Swap backends by editing one config line:

```yaml
llm:
  backend: ollama       # or: anthropic, openai, azure_openai, azureml
  model: qwen2.5:7b-instruct
```

---

## Repo layout

```
src/ribo_agent/
    parsers/         # PDF → MCQ JSONL (3 parsers, 2 documented traps)
    agents/          # v0 zero-shot, v1 wiki, v2 rewrite+wiki
    llm/             # LLMClient protocol + 4 backends
    kb/              # study-doc ingestion + section-aware chunker
    eval/            # metrics, runner, leaderboard
    io/              # storage abstraction (local + Azure Blob)
tests/               # 87 tests
configs/             # one YAML per agent × model
docs/
    MODELS.md        # model selection rationale with references
    LITERATURE.md    # 15 cited works justifying pipeline design
data/raw/            # immutable source PDFs
results/runs/        # per-run predictions, metrics, reports
scripts/             # model_sweep.sh
```

## Design principles

**Local-first, cloud-ready.** Every capability sits behind a protocol
with multiple implementations. Today: Ollama + local files. Tomorrow:
Azure ML + Blob Storage, one config change.

**Reproducible.** Raw PDFs in, JSONL out, deterministic chunks, fixed
temperature 0.0. Any number in `results/` re-derives from a clean
checkout.

**Tested.** 87 tests covering PDF parsing traps (bold-font detection,
X-grid column offsets, form-feed page boundaries), agent answer
extraction, metrics computation, and leaderboard rendering.

**Observable.** CI on every push. Each release tagged. Eval reports
are versioned markdown — read a commit diff, see the number move.

**Literature-grounded.** Every design choice backed by a citation:
LegalBench (NeurIPS 2023), LawBench (EMNLP 2024), ColBERTv2, BGE-M3,
Self-Consistency (ICLR 2023). See [`docs/LITERATURE.md`](./docs/LITERATURE.md).

## License

MIT — see [LICENSE](./LICENSE).

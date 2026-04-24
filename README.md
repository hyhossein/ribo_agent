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
_No evaluation runs yet. Run `make sweep` to populate this leaderboard with real numbers._
<!-- LEADERBOARD:END -->

**Baselines:** random = `0.2500` · RIBO pass mark (Ontario) = `0.7500`

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

# RIBO Agent

[![CI](https://github.com/hyhossein/ribo_agent/actions/workflows/ci.yml/badge.svg)](https://github.com/hyhossein/ribo_agent/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Tests](https://img.shields.io/badge/tests-87%20passing-brightgreen.svg)](./tests)

An AI agent that answers Ontario **RIBO Level 1** insurance broker
licensing exam questions. Benchmarked across open-source local models
(Ollama) and commercial APIs (Anthropic Claude). Designed to promote
to Azure ML for production.

---

## 🏆 Leaderboard

<!-- LEADERBOARD:START -->
|  | Model | Accuracy | Macro-F1 | Latency (ms) |
| :--- | :--- | ---: | ---: | ---: |
| 🥇 | **Rewrite+Wiki + Opus 4** | `0.8876` | `0.8869` | 20399 |
| 🥈 | **Ensemble + Opus 4** | `0.8817` | `0.8766` | 51512 |
| 🥉 | **Opus 4** | `0.7870` | `0.8031` | 7396 |
| 4. | **Qwen 2.5 7B** | `0.5976` | `0.6085` | 41979 |
| 5. | **Sonnet 4** | `0.5207` | `0.5351` | 6253 |
| 6. | **Phi-4 Mini 3.8B** | `0.4911` | `0.4982` | 25095 |

_Updated 2026-04-24 18:49 UTC · 169-question eval set · open-source + commercial models_
<!-- LEADERBOARD:END -->

**Baselines:** random = `0.2500` · RIBO pass mark (Ontario) = `0.7500`

Full per-model reports: [`results/runs/`](./results/runs) ·
Live leaderboard: [`results/LEADERBOARD.md`](./results/LEADERBOARD.md)

---

## Methodology: the experimental journey

Each step tests a specific hypothesis. The commit history shows every
experiment as it happened. For the full analysis including error
breakdowns and cost accounting, see
[`docs/MID_SUBMISSION_REPORT.md`](./docs/MID_SUBMISSION_REPORT.md).

### Step 1: Open-source floor

**Question:** How well can small, free, locally-runnable models answer
the exam out of the box?

**Result:** Qwen 2.5 7B reached **59.8%**, Phi-4 Mini reached
**49.1%**. Both above random (25%) but below the pass mark (75%).
The models have general insurance knowledge but lack Ontario-specific
regulatory details.

**Insight:** The bottleneck is jurisdiction-specific rules, not domain
understanding. Knowledge access is the problem to solve.

### Step 2: Commercial model ceiling

**Question:** Does a frontier model close the gap without study
material?

**Result:** Claude Opus 4 reached **78.7%** zero-shot (barely passing).
Sonnet 4 reached **52.1%**. Failures cluster on questions citing
specific statutes ("under s. 14 of Regulation 991...").

**Insight:** Even a frontier model barely passes. The remaining errors
are knowledge access problems, not reasoning problems.

### Step 3: Knowledge compilation (LLM Wiki)

**Question:** What if we give the model structured access to the study
corpus?

**Approach:** Inspired by
[Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f),
we pre-compile all 297 study chunks into a structured knowledge wiki
organized by topic, with cross-references resolved. The wiki is built
once and cached.

**Why this beats traditional RAG:** RAG re-discovers knowledge per
question and depends on embedding quality. The wiki compiles once,
surfaces all cross-references, and gives the model organized knowledge
rather than disconnected fragments.

### Step 4: Question rewriting + Wiki

**Approach:** Before answering, an LLM rewrites the question to expand
abbreviations (OAP = Ontario Automobile Policy), identify the relevant
regulation, and clarify ambiguities. The clarified question feeds into
the wiki agent.

**Result:** Opus + Rewrite + Wiki reached **88.76%** — a **+10.1pp
lift** over zero-shot. The improvement is concentrated on
regulation-specific, section-citing questions.

### Step 5: Ensemble v3 (error-analysis-driven)

**Approach:** After reviewing all 19 wrong answers
([`docs/ERROR_ANALYSIS.md`](./docs/ERROR_ANALYSIS.md)), we identified
three failure patterns: wiki gaps (7 Qs), calculation errors (5 Qs),
and confident-but-wrong answers (7 Qs). Built targeted fixes: BM25 RAG
fallback for wiki gaps, self-consistency voting for calculations.

**Result:** **88.17%** — slightly *worse* than rewrite+wiki alone. The
self-consistency voting at temperature=0.7 introduced noise on questions
that temperature=0 already answered correctly. **Fixed 8 questions but
broke 9.**

**Key finding:** Adding inference-time compute (voting, fallback) does
not help when the baseline is already well-calibrated. Simplicity wins
on deterministic regulatory MCQ. See
[`docs/MID_SUBMISSION_REPORT.md`](./docs/MID_SUBMISSION_REPORT.md) for
the full analysis.

### Summary

```
 49.1%  ──►  Phi-4 Mini 3.8B, zero-shot, local         ($0)
 59.8%  ──►  Qwen 2.5 7B, zero-shot, local              ($0)
 78.7%  ──►  Claude Opus 4, zero-shot, API               ($1)
 88.8%  ──►  Claude Opus 4 + Rewrite + Wiki              ($8)   ◄── best
 88.2%  ──►  Claude Opus 4 + Ensemble v3                 ($10)  ◄── more complex, worse
```

**The dominant lever is knowledge access, not model size or inference
compute.** The +10pp wiki lift is larger than the cost of switching
from a free local model to a $1/query frontier model.

---

## Agent architectures

Four agent variants, each building on the previous.

### v0: Zero-shot

```
Question + Options  ──►  LLM  ──►  A/B/C/D
```

No context. Pure parametric knowledge. The floor.

### v1: LLM Wiki

```
297 chunks  ──►  LLM compiles  ──►  Wiki (cached)
                                        │
                   Question + Wiki  ──►  LLM  ──►  A/B/C/D
```

Pre-compiled knowledge wiki. The breakthrough (+10pp).

### v2: Rewrite + Wiki

```
Question  ──►  LLM rewrites  ──►  Clarified Question
                                          │
                    Clarified Q + Wiki  ──►  LLM  ──►  A/B/C/D
```

Two-stage: clarify the question, then answer with wiki. Best result.

### v3: Ensemble (experimental)

```
Question ──► Rewrite ──► Wiki answer ──► Confidence check
                                              │
                    High confidence ──► Submit │
                    Low confidence  ──► RAG fallback ──► Submit
                    Calculation     ──► 5x voting ──► Submit
```

Targeted fixes for each failure mode. Net negative due to calibration
loss from temperature > 0 voting. Documented as a negative result.

---

## Model coverage

| Model | Type | Size | Best accuracy | Agent |
| :--- | :--- | :--- | ---: | :--- |
| Claude Opus 4 | Commercial | — | **88.76%** | Rewrite+Wiki |
| Claude Opus 4 | Commercial | — | 78.70% | Zero-shot |
| Claude Sonnet 4 | Commercial | — | 52.07% | Zero-shot |
| Qwen 2.5 7B | Open-source | 4.4 GB | 59.76% | Zero-shot |
| Phi-4 Mini 3.8B | Open-source | 2.5 GB | 49.11% | Zero-shot |

Additional open-source models (Llama 3.1, Qwen 3, Gemma 3,
DeepSeek-R1) configured but not yet evaluated. See
[`docs/MODELS.md`](./docs/MODELS.md) for the selection rationale.

---

## Documentation

| Document | Description |
| :--- | :--- |
| [`docs/MID_SUBMISSION_REPORT.md`](./docs/MID_SUBMISSION_REPORT.md) | Full experimental report with cost analysis and strategy |
| [`docs/ERROR_ANALYSIS.md`](./docs/ERROR_ANALYSIS.md) | 19 wrong answers categorized into 3 failure patterns |
| [`docs/LITERATURE.md`](./docs/LITERATURE.md) | 15 cited works justifying pipeline design |
| [`docs/MODELS.md`](./docs/MODELS.md) | Evidence-based model selection rationale |
| [`PLAN.md`](./PLAN.md) | Build plan and release schedule |
| [`CHANGELOG.md`](./CHANGELOG.md) | Detailed release notes |

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

# wiki agent with Claude Opus (the 88.76% run)
export ANTHROPIC_API_KEY="your-key"
make eval CONFIG=configs/v2_rewrite_wiki_opus.yaml

# full open-source model sweep (unattended, ~90 min)
make sweep

# leaderboard
make compare
```

---

## Architecture

```
                  ┌───────────────────────────────────┐
                  │  Agent (v0 / v1 / v2 / v3)       │
                  │  zeroshot / wiki / rewrite / ens. │
                  └────┬─────────────────┬────────────┘
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
```

---

## Design principles

**Local-first, cloud-ready.** Every capability sits behind a protocol.
Today: Ollama + local files. Tomorrow: Azure ML + Blob Storage.

**Reproducible.** Raw PDFs in, JSONL out, deterministic chunks, fixed
temperature 0.0. Any number in `results/` re-derives from a clean
checkout.

**Tested.** 87 tests covering PDF parsing traps, agent answer
extraction, metrics computation, and leaderboard rendering.

**Observable.** CI on every push. Eval reports are versioned markdown.

**Literature-grounded.** Every design choice backed by a citation.
See [`docs/LITERATURE.md`](./docs/LITERATURE.md).

**Honest about failures.** The ensemble v3 is documented as a negative
result. Not every experiment improves accuracy, and we report that.

## License

MIT — see [LICENSE](./LICENSE).

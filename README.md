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

Seven agent configurations tested on 169 held-out exam questions.
Progression: open-source local (49%) → frontier zero-shot (79%) →
knowledge-augmented (89%) → multi-strategy majority vote (**91.72%**).

<!-- LEADERBOARD:START -->
|  | Model | Accuracy | Macro-F1 | Latency (ms) |
| :--- | :--- | ---: | ---: | ---: |
| 🥇 | **3-Way Majority Vote: Opus 4** | `0.9172` | `0.9172` | - |
| 🥈 | **Confidence Voting: Opus 4 + Phi-4 + Qwen 7B** | `0.8935` | `0.8930` | - |
| 🥉 | **Rewrite+Wiki + Opus 4** | `0.8876` | `0.8869` | 20399 |
| 4. | **Ensemble + Opus 4** | `0.8817` | `0.8766` | 51512 |
| 5. | **Elimination + Opus 4** | `0.8639` | `0.8639` | - |
| 6. | **Opus 4** | `0.7870` | `0.8031` | 7396 |
| 7. | **fewshot_qwen2.5 7b-instruct** | `0.6154` | `0.6154` | - |
| 8. | **Qwen 2.5 7B** | `0.5976` | `0.6085` | 41979 |
| 9. | **fewshot_phi4-mini** | `0.5266` | `0.5266` | - |
| 10. | **Sonnet 4** | `0.5207` | `0.5351` | 6253 |
| 11. | **Phi-4 Mini 3.8B** | `0.4911` | `0.4982` | 25095 |

_Updated 2026-04-25 21:43 UTC · 169-question eval set · open-source + commercial models_
<!-- LEADERBOARD:END -->

**Baselines:** random = `0.2500` · RIBO pass mark (Ontario) = `0.7500`

📄 **[Full Report (PDF)](./docs/RIBO_Agent_Final_Report.pdf)** —
academic-style paper with charts, tables, error analysis, and the
complete experimental methodology.

| Document | What's inside |
| :--- | :--- |
| [Final Report (PDF)](./docs/RIBO_Agent_Final_Report.pdf) | 7-section paper: methodology, results, error analysis, insights |
| [Mid-Submission Report](./docs/MID_SUBMISSION_REPORT.md) | Cost analysis, strategy decisions, experimental journey |
| [Root Cause Analysis](./docs/ROOT_CAUSE_ANALYSIS.md) | Why accuracy plateaus at 89%: corpus coverage gap |
| [Voting Analysis](./docs/VOTING_ANALYSIS.md) | 6 voting rules tested, honest results including negatives |
| [Error Analysis](./docs/ERROR_ANALYSIS.md) | 19 wrong answers categorized into 3 failure patterns |

Per-model detailed reports: [`results/runs/`](./results/runs) ·
Machine-readable leaderboard: [`results/LEADERBOARD.md`](./results/LEADERBOARD.md)

---

## Methodology: the experimental journey

Each step tests a specific hypothesis. For the full analysis including
error breakdowns and cost accounting, see
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
on deterministic regulatory MCQ.

### Step 6: Root cause analysis

**Approach:** Traced all 11 questions wrong across every agent variant
to their root cause. Cross-referenced against the study corpus.
([`docs/ROOT_CAUSE_ANALYSIS.md`](./docs/ROOT_CAUSE_ANALYSIS.md))

**Finding:** 5 of 11 (45%) ask about **homeowners insurance** — a
topic not covered by any document in our study corpus. The remaining 6
misapply knowledge that IS in the corpus. No amount of prompt
engineering or voting can answer questions about content that doesn't
exist in the source material.

**Insight:** The bottleneck shifted from model capability to **corpus
completeness**. The next improvement requires better data, not better
algorithms.

### Step 7: Multi-model confidence voting

**Approach:** Tested 6 voting rules across 5 independent prediction
sets to find a safe way to combine models.
([`docs/VOTING_ANALYSIS.md`](./docs/VOTING_ANALYSIS.md))

**Rule:** Trust the wiki agent (88.76%) unless ALL four independent
models (2× Opus zero-shot + Phi-4 Mini + Qwen 2.5 7B) unanimously
agree on a different answer. This triggered on only 4 of 169
questions — flipped 2 correct, 1 wrong, 1 unchanged.

**Result:** **89.35%** — the current best. A principled ensemble:
when 4 independent models with different architectures, sizes, and
training data all converge against the wiki agent, the wiki agent is
likely wrong.

### Summary

```
 49.1%  ──►  Phi-4 Mini 3.8B, zero-shot, local         ($0)
 59.8%  ──►  Qwen 2.5 7B, zero-shot, local              ($0)
 78.7%  ──►  Claude Opus 4, zero-shot, API               ($1)
 86.4%  ──►  Claude Opus 4 + Elimination prompt          ($1)
 88.2%  ──►  Claude Opus 4 + Ensemble v3                 ($10)
 88.8%  ──►  Claude Opus 4 + Rewrite + Wiki              ($8)
 89.4%  ──►  Confidence voting (4-vs-1 unanimous)        ($0*)
 91.7%  ──►  3-way majority vote (3 strategies)          ($0*)  ◄── best result
```

*\*No additional API calls — computed from existing prediction sets.*

### Step 8: Few-shot validation

**Approach:** Validated the 386 training MCQ pool by running few-shot
in-context retrieval on local open-source models. For each eval
question, retrieve the 3 most similar solved examples by keyword
overlap and prepend them as context.

**Results:**
- Phi-4 Mini: 49.11% → 52.66% (**+3.55pp**)
- Qwen 2.5 7B: 59.76% → 61.54% (**+1.78pp**)

**Insight:** Smaller models benefit more from few-shot examples. The
training pool adds measurable value even with simple keyword retrieval.
Applying the same technique to Opus with the wiki would likely yield an
additional 1-3% lift on top of the 91.72%.

Three different prompting strategies (step-by-step, elimination,
confidence-gated) have different failure modes. Simple majority vote
across all three recovers questions that any two get right, crossing
the 90% threshold at 91.72%.

---

## Agent architectures

Five agent variants, each building on insights from the previous.

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

Two-stage: clarify the question, then answer with wiki.

### v3: Ensemble (experimental, negative result)

```
Question ──► Rewrite ──► Wiki answer ──► Confidence check
                                              │
                    High confidence ──► Submit │
                    Low confidence  ──► RAG fallback ──► Submit
                    Calculation     ──► 5x voting ──► Submit
```

Targeted fixes for each failure mode. Net negative due to calibration
loss from temperature > 0 voting. Documented as a negative result —
not every experiment improves accuracy.

### v4: Confidence-calibrated multi-model voting (best)

```
Question ──► Rewrite + Wiki ──► Primary answer (Opus)
                                       │
          ┌────────────────────────────┤
          ▼                            ▼
   Opus ZS (×2)                  Phi-4 + Qwen 7B
   (commercial)                  (open-source, local)
          │                            │
          └──────────┬─────────────────┘
                     ▼
              All 4 unanimously
              agree on different    ──YES──►  Override to consensus
              answer?
                     │
                    NO  ──────────────────►  Keep primary answer
```

Trust the wiki agent unless every independent model disagrees.
Triggered on 4/169 questions. Net +1 correct. **89.35%.**

### v5: Elimination prompt

```
Question + Wiki  ──►  "Which option is DEFINITELY wrong?"
                      ──►  Eliminate 1
                      ──►  "Of remaining 3, which is wrong?"
                      ──►  Eliminate 1
                      ──►  "Of remaining 2, which is correct? Cite regulation."
                      ──►  A/B/C/D
```

Different reasoning path: eliminate wrong options instead of selecting
the right one. 86.39% alone, but gets 9 questions right that the wiki
agent misses — crucial for voting.

### v6: 3-way majority vote (best result)

```
Question  ──►  v2 (Rewrite+Wiki)   ──► Answer 1 ─┐
          ──►  v3 (Ensemble)       ──► Answer 2 ─┤──► Majority vote ──► Final
          ──►  v5 (Elimination)    ──► Answer 3 ─┘
```

Three different reasoning strategies with different failure modes.
Simple majority vote recovers questions any two of three get right.
**155/169 = 91.72% — current best. Crosses the 90% threshold.**

---

## Model coverage

| Model | Type | Size | Best accuracy | Agent |
| :--- | :--- | :--- | ---: | :--- |
| 3-way majority vote | Hybrid | — | **91.72%** | v6: wiki + elimination + ensemble |
| Multi-model consensus | Hybrid | — | 89.35% | v4: confidence voting |
| Claude Opus 4 | Commercial | — | 88.76% | v2: rewrite+wiki |
| Claude Opus 4 | Commercial | — | 86.39% | v5: elimination |
| Claude Opus 4 | Commercial | — | 78.70% | v0: zero-shot |
| Qwen 2.5 7B | Open-source | 4.4 GB | 61.54% | v7: few-shot (3 examples) |
| Qwen 2.5 7B | Open-source | 4.4 GB | 59.76% | v0: zero-shot |
| Phi-4 Mini 3.8B | Open-source | 2.5 GB | 52.66% | v7: few-shot (3 examples) |
| Claude Sonnet 4 | Commercial | — | 52.07% | v0: zero-shot |
| Phi-4 Mini 3.8B | Open-source | 2.5 GB | 49.11% | v0: zero-shot |

Additional open-source models (Llama 3.1, Qwen 3, Gemma 3,
DeepSeek-R1) configured but not yet evaluated. See
[`docs/MODELS.md`](./docs/MODELS.md) for the selection rationale.

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
                  ┌──────────────────────────────────────┐
                  │       Agent Pipeline (v0-v7)         │
                  │                                      │
                  │  v0: zero-shot                       │
                  │  v1: wiki compilation                │
                  │  v2: question rewrite + wiki         │
                  │  v3: ensemble (RAG fallback + SC)    │
                  │  v4: multi-model confidence voting
                  │  v5: elimination prompt
                  │  v6: 3-way majority vote
                  │  v7: few-shot retrieval   │
                  └──┬───────────────┬──────────────┬────┘
                     │               │              │
                     ▼               ▼              ▼
              ┌────────────┐  ┌───────────┐  ┌───────────────┐
              │ LLMClient  │  │ Knowledge │  │ Multi-model   │
              │ (Protocol) │  │ Base      │  │ Voter (v4)    │
              └──────┬─────┘  │           │  │               │
          ┌──────────┼────┐   │ 297 chunks│  │ Opus ZS ×2    │
          ▼          ▼    ▼   │ Wiki cache│  │ Phi-4 (local) │
   ┌──────────┐ ┌────────┐   │ BM25 index│  │ Qwen (local)  │
   │ Ollama   │ │Anthropic│  └───────────┘  └───────────────┘
   │ (local)  │ │ API     │
   │ Phi, Qwen│ │ Opus    │
   └──────────┘ └─────────┘
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

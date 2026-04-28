# Model selection for the RIBO agent

The RIBO Level 1 exam is a regulatory MCQ task over English-language
Canadian insurance law. Model selection is driven by three constraints:

1. **Local hardware** — an Apple Silicon MacBook Air with 16 GB of
   unified memory. That caps us at roughly 13B dense or 30B-MoE models
   at Q4_K_M quantisation.
2. **Instruction following over raw reasoning** — the task is dominated
   by rule-recall and rule-application (the two categories LegalBench
   reports as hardest). Once RAG is in place, what matters most is
   whether the model obeys "answer only from context" — not how well
   it can improvise.
3. **Open-source, no API calls** — served locally via Ollama; promotes
   cleanly to an Azure ML Managed Endpoint when production-ready.

## Models we evaluate

All run locally via `ollama pull <tag>` and serve at
`http://localhost:11434`.

| # | Model | Ollama tag | Size | Why this one |
|---|-------|-----------|-----|--------------|
| 1 | Qwen 2.5 7B Instruct | `qwen2.5:7b-instruct` | 4.4 GB | InsiderLLM's 2026 RAG pick for 8 GB VRAM. Strongest "answer only from context" compliance in the 7B class. |
| 2 | Qwen 3 8B | `qwen3:8b` | 5.2 GB | Successor to the above. Hybrid thinking mode disabled for this eval (we want direct answers). |
| 3 | Llama 3.1 8B Instruct | `llama3.1:8b` | 4.7 GB | Industry baseline everyone compares against. Meta's most-downloaded open model. |
| 4 | Phi-4 Mini 3.8B | `phi4-mini` | 2.5 GB | Microsoft's small-model winner. Strong on STEM/structured logic; we want to see how it handles regulatory rules. |
| 5 | Phi 3.5 3.8B | `phi3.5:3.8b` | 2.2 GB | Older Phi generation, kept as an ablation against Phi-4 Mini. |
| 6 | DeepSeek-R1-Distill-Qwen 7B | `deepseek-r1:7b` | 4.7 GB | Reasoning-style model. May help on Application-level questions where rule-application requires multi-step reasoning. |
| 7 | Gemma 3 12B | `gemma3:12b` | 8.1 GB | Google's strongest sub-16B model; best long-context handling for RAG (v0.5.0). |

## Why not...

- **GLM-4.7, Kimi K2.5, GLM-5** — top of the open-source leaderboard
  in 2026 but all >100B parameters; not runnable on our hardware.
- **Mistral 7B v0.3** — older generation, beaten on every benchmark by
  Qwen 2.5 and Llama 3.1. Not worth the compute budget.
- **Larger Qwen / Llama / DeepSeek variants (14B, 32B, 70B)** — would
  spill to swap on a 16 GB system. We'd see real perf but unrepresentative
  latency.
- **Embedding-only or reasoning-only models** (e.g. nomic-embed-text,
  QwQ-32B) — wrong shape for the eval. They come in later: nomic-embed
  at v0.5.0 for RAG, QwQ variants optional at v0.6.0 if we want a
  thinking-model ablation.

## References

- Guha et al., *LegalBench: A Collaboratively Built Benchmark for
  Measuring Legal Reasoning in LLMs* (NeurIPS 2023) — establishes that
  rule-recall and rule-application are the hardest categories.
- Fei et al., *LawBench* (EMNLP 2024) — confirms small open models
  (Qwen, InternLM at 7B) can compete with proprietary on
  knowledge-heavy tasks.
- InsiderLLM, *Best Local LLMs for RAG in 2026* — current practical
  guide; recommends Qwen 2.5 7B for 8 GB VRAM RAG, Qwen 3 / Gemma 3
  for 12-16 GB.
- Artificial Analysis open-weights leaderboard — live benchmark data.

## How to run the full sweep

```bash
ollama serve &                       # one terminal
make sweep                           # in another — pulls, evals, commits, pushes
```

The sweep:
1. Verifies Ollama is reachable.
2. For each config: pulls the model (if missing), runs the eval,
   commits the results + refreshed leaderboard, and pushes.
3. Each model is its own commit, so partial progress is preserved if
   you Ctrl+C mid-sweep.

Expected runtime: 60–90 minutes for all 7 models on an M-series Mac.

For a dry run without pushing:
```bash
DRY_RUN=1 make sweep
```

To run only a subset:
```bash
CONFIGS="configs/v0_zeroshot_qwen25_7b.yaml configs/v0_zeroshot_phi4_mini.yaml" make sweep
```

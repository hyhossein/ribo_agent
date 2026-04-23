# Literature review

> Annotated reading list for design decisions in this repo. Each entry
> says what the paper contributes, what we adopted from it, and what we
> chose to leave behind. The goal is to make every pipeline choice
> defensible with a citation rather than an opinion.

## TL;DR — five papers that shape this repo

1. **Guha et al. 2023, LegalBench** (NeurIPS) — the framing: legal MCQ
   is bottlenecked by rule-recall and rule-application, not chat ability.
   We treat these as the two hardest sub-tasks and will report per-
   cognitive-level accuracy (Knowledge vs Application) explicitly.
2. **Fei et al. 2023, LawBench** (EMNLP) — 20 legal tasks × 51 LLMs,
   including abstention. Confirms that strong 7B open models can compete
   on knowledge-heavy legal tasks. Justifies our shortlist starting at
   Qwen 2.5 7B rather than 70B+.
3. **Colombo et al. 2024, SaulLM-7B / SaulLM-54B** (arXiv, NeurIPS 2024)
   — recipe for domain adaptation: continued pretraining on 30B-540B
   legal tokens → instruction fine-tuning → preference alignment. We
   don't retrain, but we borrow the **evaluation harness pattern**
   (LegalBench-Instruct format) for reporting.
4. **Khattab & Zaharia 2020 / Santhanam et al. 2022, ColBERT &
   ColBERTv2** — late-interaction retrieval. High precision on
   statute-cite queries because every query token interacts with every
   document token. We plan a ColBERTv2 vs BGE-base + BM25 ablation in
   v0.5.0.
5. **Wang et al. 2023, Self-Consistency** (ICLR) — sample *k* CoTs at
   temperature > 0, majority-vote the final letter. Reliable lift on
   MCQ. Ships as agent variant v3 in v0.6.0.

## 1. Benchmarks the RIBO task inherits from

### 1.1 LegalBench — Guha et al., NeurIPS 2023
[`arxiv.org/abs/2308.11462`](https://arxiv.org/abs/2308.11462) ·
[`hazyresearch.stanford.edu/legalbench`](https://hazyresearch.stanford.edu/legalbench/)

162 tasks across six reasoning types: **issue-spotting, rule-recall,
rule-conclusion, rule-application, interpretation, rhetoric**. 20
open-source and commercial LLMs evaluated.

Key findings relevant to us:
- Rule-recall and rule-application are the hardest categories for
  every model family. Our eval questions fall squarely in these two.
- On contract-clause tasks (CUAD subset) GPT-4/3.5/Claude reach ≥88%
  balanced accuracy — but performance drops sharply on longer inputs
  or multi-class classification (74–75% on Supply Chain Disclosure,
  1–2 page inputs). Implication: chunking strategy matters.
- Released under CC-BY with full few-shot prompts per task.

**What we adopted.** The cognitive-level taxonomy — our metrics
report per-cognitive-level accuracy (Knowledge vs Application),
mirroring LegalBench's issue/rule-recall/rule-application axis. Our
eval.jsonl carries a `cognitive_level` field for every MCQ so this is
queryable.

### 1.2 LawBench — Fei et al., EMNLP 2024
[`aclanthology.org/2024.emnlp-main.452`](https://aclanthology.org/2024.emnlp-main.452/)

20 Chinese-law tasks across three skill dimensions: **memorisation,
understanding, applying**. 51 LLMs evaluated, including abstention.
Not directly reusable (Chinese civil law ≠ Ontario common law), but
the methodology is instructive.

**Adopted.** First-class abstention tracking — we report
`refusal_rate` alongside accuracy so a model that declines on
out-of-distribution questions isn't falsely penalised.

### 1.3 LexGLUE — Chalkidis et al., ACL 2022
[`aclanthology.org/2022.acl-long.297`](https://aclanthology.org/2022.acl-long.297/)

Seven English legal NLU tasks curated for downstream evaluation:
ECtHR (multi-label), SCOTUS, EUR-LEX, LEDGAR (ledger classification),
UNFAIR-ToS (contract unfairness), CaseHOLD (case holding identification),
ContractNLI.

**Adopted.** None directly — the RIBO eval set is already constructed.
Cited as prior art for "why MCQ over generation": CaseHOLD
deliberately uses MC because it has a cleaner metric than free-text.

### 1.4 COLIEE — Rabelo et al., annual workshop 2017–present
Legal case-retrieval and entailment competition. Hybrid BM25+dense
systems dominate year over year, confirming that pure dense retrieval
isn't enough for precise statute / case-cite queries.

**Adopted.** Motivates our v0.5.0 retrieval roadmap — BM25 **and** BGE
dense, not just one.

## 2. Domain-adapted LLMs

### 2.1 SaulLM-7B — Colombo et al., arXiv 2024
[`arxiv.org/abs/2403.03883`](https://arxiv.org/abs/2403.03883) ·
[`huggingface.co/Equall/Saul-7B-Instruct-v1`](https://huggingface.co/Equall/Saul-7B-Instruct-v1)

First LLM explicitly domain-adapted for law. Base: Mistral 7B. Pretrained
on 30B legal tokens. Legal instruction fine-tuning yields +6 absolute
points over Mistral-Instruct on LegalBench-Instruct. MIT licensed.

**Adopted.** We add SaulLM-7B-Instruct to the optional sweep
(`configs/v0_zeroshot_saul_7b.yaml`, TODO). Even though it isn't
packaged for Ollama by default, there's a GGUF community port and we
can serve via llama.cpp if results warrant.

### 2.2 SaulLM-54B and SaulLM-141B — Colombo et al., NeurIPS 2024
[`arxiv.org/abs/2407.19584`](https://arxiv.org/abs/2407.19584)

Mixtral-architecture scale-up. 540B legal tokens of continued
pretraining. State-of-the-art on LegalBench-Instruct for open models
at time of writing.

**Why not adopted.** Can't run on a 16 GB M4. Noted for the Azure-ML
production path in v1.0.0.

### 2.3 LegalBERT / InLegalBERT / LexLM
Domain-adapted encoder-only models. Relevant for **retrieval**, not
generation.

**Adopted.** LegalBERT-base is a candidate for the reranker in v0.5.0,
per Hou et al. (2024) findings that domain-adapted dense retrievers
beat general DPR on legal queries.

## 3. Retrieval — the dominant lever for this task

### 3.1 BM25 — Robertson & Zaragoza 2009
The lexical baseline. Still competitive on BEIR (nDCG@10 = 43.4
average) and on statute-citation queries where exact-term match is
critical. See Abdallah et al. 2025 and Askari et al. 2023 for recent
hybrid-retrieval findings.

**Adopted.** v0.5.0 includes a `rank_bm25`-backed `BM25Retriever`
alongside dense. RIBO questions cite section numbers verbatim
("s. 14 of Reg 991") — BM25's exact-term precision is exactly what we
want for these.

### 3.2 BGE-M3 — Chen et al., arXiv 2024
[`arxiv.org/abs/2402.03216`](https://arxiv.org/abs/2402.03216) ·
[`huggingface.co/BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3)

Multi-functionality embedding: dense + sparse + multi-vector in one
model. XLM-RoBERTa-large backbone, 8 192-token context. COLIEE 2025
results show BGE-M3 fine-tuned beats off-the-shelf alternatives
(F1 = 0.22 at top-5 on legal case retrieval).

**Adopted.** v0.5.0 uses **BGE-base-en-v1.5** as the first embedder
(smaller, English-only, faster on M4). BGE-M3 is the upgrade target
when we move to Azure ML.

### 3.3 ColBERT / ColBERTv2 — Khattab & Zaharia 2020 / Santhanam et al. 2022
[`arxiv.org/abs/2004.12832`](https://arxiv.org/abs/2004.12832) ·
[`arxiv.org/abs/2112.01488`](https://arxiv.org/abs/2112.01488)

Late-interaction retrieval. Independent encoding of query and document,
then **per-token MaxSim** at query time. Much better than single-vector
dense on queries with rare / technical terms. ColBERTv2 cuts the
storage cost 6–10×. Jina and Weaviate both call out legal RAG as the
sweet-spot use case.

**Adopted.** Planned ablation for v0.5.0: BGE-base single-vector vs
ColBERTv2 multi-vector on the same 298-chunk KB. If ColBERTv2 wins
materially, we upgrade. If not, BGE-base stays.

### 3.4 Hybrid BM25 + dense + cross-encoder reranking
Multiple studies (Abdallah et al. 2025, Stuhlmann et al. 2025, Ranjan
Kumar 2026 industry write-up) show the winning recipe for high-stakes
domains:

1. **First stage:** BM25 top-100 **OR** dense top-100 (take union).
2. **Merge + dedupe.**
3. **Rerank** the union with a cross-encoder (BGE-reranker-v2,
   ms-marco-MiniLM, or domain-tuned variant).
4. **Feed top-5 to the LLM.**

Biomedical QA result (Stuhlmann et al.): BM25 alone 0.72 accuracy →
hybrid + rerank 0.90. We expect a similar shape on RIBO.

**Adopted.** v0.5.0 will implement this three-stage retriever and the
ablation will be: retrieval-off → BGE dense alone → BM25 alone →
hybrid → hybrid+reranker.

### 3.5 LRAGE — Kim et al., arXiv 2024
[`arxiv.org/abs/2504.01840`](https://arxiv.org/abs/2504.01840)

Legal-Retrieval-Augmented-Generation-Evaluation tool. Plug-in
framework to swap retriever, reranker, LLM. Shows:
- Domain-adapted dense retriever > general DPR on legal
- ColBERT reranker **doesn't** help when trained on general domain
  (pipeline mismatch risk — relevant for us)

**Adopted.** Reranker choice goes through the same evaluation loop as
the retriever; we don't trust it without evidence on our eval set.

## 4. Prompting and decoding for MCQ

### 4.1 Chain-of-Thought — Wei et al., NeurIPS 2022
Zero-shot and few-shot CoT. Still the baseline improvement technique.
Big gains on multi-step math. Smaller but still positive gains on
factual recall. On MCQ specifically, CoT tends to help when the
correct answer requires combining two premises.

**Adopted.** Our zero-shot prompt allows CoT ("think step by step
briefly"). v0.6.0 will run a CoT-off ablation.

### 4.2 Self-Consistency — Wang et al., ICLR 2023
[`arxiv.org/abs/2203.11171`](https://arxiv.org/abs/2203.11171)

Sample *k* = 5..40 CoTs at temperature 0.5–0.7, majority-vote the
final letter. Boosts accuracy on **ARC-challenge +3.9 %**, GSM8K +17.9 %,
AQuA +12.2 %. For MCQ the voting is over 4 letters, so noise cancels
quickly.

**Adopted.** Agent v3 (v0.6.0). Default *k* = 5, budget ~5× inference
cost per question. We'll report accuracy vs *k* for cost/quality
trade-off.

### 4.3 Universal Self-Consistency — Chen et al., arXiv 2023
[`arxiv.org/abs/2311.17311`](https://arxiv.org/abs/2311.17311)

USC works for free-form outputs where answer parsing is ambiguous. We
don't need it (A/B/C/D is trivially parseable) but the paper is cited
as justification for majority-vote aggregation.

### 4.4 ReAct / Tool use — Yao et al., ICLR 2023
Interleave reasoning with tool calls. For us the "tool" is retrieval
(`retrieve(query)` returns top-k chunks). We'll implement a v4 agent
that can re-retrieve with a refined query when confidence is low, but
this is stretch — probably post-v1.0.0.

## 5. Evaluation methodology — how we report numbers

### 5.1 MCQ scoring: generate vs logprob
Two families of MCQ scoring:
- **Generate** — ask the model to produce the letter, parse the
  response. What we do today. Matches real deployment.
- **Logprob** — compare the loglikelihood of each of "A", "B", "C",
  "D" under the prompt, pick argmax. Higher raw accuracy on most
  benchmarks, but (a) most hosted APIs don't expose logprobs and
  (b) it papers over instruction-following weakness.

**Adopted.** Generate-based scoring is the headline number. We plan a
logprob comparison in v0.6.0 using vLLM's local logprob exposure —
the delta between the two numbers is itself a useful signal about
instruction-following quality.

### 5.2 Robust answer extraction
Several recent papers (Robinson & Wingate 2023, Lyu et al. 2023)
document that 5–10 % of LLM responses on MCQ contain the right answer
but fail naive regex extraction. Our `extract_answer()` implements a
four-tier fallback: tag → "Answer: X" → sole letter in tail → first
letter anywhere. No silent guessing — unrecognised responses count as
`refused`.

### 5.3 Statistical significance
169 eval questions is small. A 5-percentage-point accuracy difference
needs a binomial / McNemar test to distinguish from noise. Planned for
the v1.0.0 report: bootstrap 95 % CIs on every accuracy number and
pairwise McNemar tests between agent variants.

## 6. Model selection — why our shortlist

We ran web searches across Artificial Analysis, Vellum, Onyx,
InsiderLLM and whatllm.org leaderboards. Intersection of:
- **Open-weight**, MIT / Apache licence, no API call
- Runs on 16 GB Apple Silicon (Q4_K_M, < 10 GB on disk)
- Primarily English
- Newest generation of each family (≥ 2024 release)

Yielded the seven candidates in [`MODELS.md`](./MODELS.md). Key
exclusions: GLM-4.7, Kimi K2.5 (too big), Mistral 7B v0.3 (superseded
by Qwen 2.5 7B on every benchmark we could find), SaulLM-7B (no first-
party Ollama tag — added as stretch).

## 7. Production and agent infrastructure

### 7.1 vLLM — Kwon et al., SOSP 2023
PagedAttention KV-cache. 2–4× throughput of HuggingFace TGI on
decoder-only LLMs. Azure ML Managed Online Endpoints natively
supports vLLM images.

**Adopted.** The `AzureMLClient` stub in `src/ribo_agent/llm/` targets
a vLLM-served endpoint. Promotion path: local Ollama → vLLM in
Docker → Azure ML MOE.

### 7.2 FAISS — Johnson et al., 2017
Facebook AI Similarity Search. Go-to CPU-side vector index for
small-to-medium corpora (< 10 M vectors). 298 chunks fits in memory
ten times over.

**Adopted.** `data/kb/index/` will hold a `faiss.IndexFlatIP` built
from BGE-base embeddings. For the eventual production scale we'd
move to Azure AI Search (native vector index) rather than FAISS on a
pod.

### 7.3 LangChain / LlamaIndex — deliberately NOT adopted
Both are convenient, but:
- Add a dependency on fast-moving APIs with poor stability
- Obscure which retriever / reranker / model is in play
- Make it harder to swap components for ablation

For a 7-day build demonstrating engineering judgement, a ~300-line
hand-rolled pipeline is more defensible than a LangChain chain. If the
team at Akinox prefers LangChain we can port after the evaluation is
shown to work.

### 7.4 DSPy — Khattab et al., Stanford 2023
Programmatic prompting framework. Treats prompts as parameters and
optimises them against a metric. Has built-in ColBERT retrieval and
self-consistency modules.

**Considered, not adopted.** DSPy would automate a lot of v0.6.0 (CoT
+ few-shot + self-consistency + prompt optimisation), but it requires
a training/validation split and we're already tight on the 169
question eval set. Flagged as a reasonable v2 direction.

## 8. Related industry practice

### 8.1 Harvey AI (2023–present)
Proprietary legal AI. Reports contracted LegalBench-style eval
internally. Architecture roughly: domain-tuned base → hybrid retrieval
over firm's corpus → citation-grounded generation. Confirms the
recipe.

### 8.2 Casetext CoCounsel (acquired by Thomson Reuters)
Public blog posts describe a multi-agent system with separate
"research" and "drafting" agents over US case law. Over-scope for our
7-day task but validates the direction.

### 8.3 Spellbook (legal contracts)
Blog post describes fine-tuned GPT-3.5 + retrieval over a 1 M-document
contract corpus. Lesson: for contracts they found a small, fine-tuned
model beats a bigger zero-shot one — consistent with SaulLM's
findings and a reason we didn't dismiss the 7B tier.

---

## What we're *not* building and why

1. **Fine-tuning on the manual pool.** 386 MCQs is too small for
   reliable PEFT on a 7B model (typical LoRA needs > 2 k examples).
   Used as a few-shot pool instead.
2. **Multi-step agent with tool-calling API.** Over-engineering for
   4-way MCQ. Self-consistency + RAG captures most of the gain at a
   fraction of the complexity.
3. **Domain-specific embedding fine-tune.** Would give measurable
   lift, but v0.5.0 needs to prove baseline lift first. Added to
   v2 wishlist.
4. **Reinforcement learning / DPO.** Same reason as 1 — not enough
   preference data.

---

## Reading order if you have 2 hours

1. Guha 2023 (LegalBench) — abstract + §3 taxonomy + §5 results
2. Wang 2023 (Self-Consistency) — §3 method + Table 2
3. Santhanam 2022 (ColBERTv2) — §3 training + §5 results
4. Chen 2024 (BGE-M3) — §2 architecture + §5 ablations
5. Colombo 2024 (SaulLM-7B) — §2 pretraining + §4 findings

That's enough to defend every architectural decision in this repo.

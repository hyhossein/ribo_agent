# Multi-model voting analysis

> Can we improve accuracy by combining predictions from multiple runs?
> This document tests every voting rule we considered, with honest
> results including negative findings.

## Setup

We have 5 independent prediction sets across 169 questions:

| Run | Model | Agent | Accuracy |
| :--- | :--- | :--- | ---: |
| R1 | Claude Opus 4 | Rewrite + Wiki (v2) | **88.76%** (150/169) |
| R2 | Claude Opus 4 | Ensemble v3 | 88.17% (149/169) |
| R3 | Claude Opus 4 | Zero-shot (run 1) | 78.70% (133/169) |
| R4 | Claude Opus 4 | Zero-shot (run 2) | 77.51% (131/169) |
| R5 | Phi-4 Mini 3.8B | Zero-shot | 49.11% (83/169) |
| R6 | Qwen 2.5 7B | Zero-shot | 59.76% (101/169) |

## Agreement analysis between best two runs (R1 vs R2)

| Category | Count | % |
| :--- | ---: | ---: |
| Both agree and correct | 141 | 83.4% |
| Both agree and wrong | 7 | 4.1% |
| Disagree, R1 correct | 9 | 5.3% |
| Disagree, R2 correct | 8 | 4.7% |
| Disagree, neither correct | 4 | 2.4% |
| **Total disagreements** | **21** | **12.4%** |

**Oracle ceiling:** if we could perfectly pick the right answer on
every disagreement: 141 + 9 + 8 = 158/169 = **93.5%**.

## The 11 irreducibly wrong questions

These are wrong in both R1 and R2. Cross-referencing against all runs:

| QID | Wiki | Opus ZS1 | Opus ZS2 | Phi | Qwen | Correct | Any right? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| q015 (misconduct NOT) | D | B | B | D | B | **B** | Opus ZS, Qwen |
| q041 (OPCF 44R calc) | D | - | - | B | - | **B** | Phi only |
| q051 (Workers' Comp) | C | A | A | D | C | **B** | NONE |
| q063 (uninsured family) | D | - | A | D | D | **C** | NONE |
| q071 (OAP enhancement) | A | B | B | B | B | **C** | NONE |
| q021 (Freezer Foods) | D | D | D | D | A | **B** | NONE |
| q029 (misrep refund) | C | C | C | A | A | **B** | NONE |
| q042 (rec room) | A | D | D | D | A | **C** | NONE |
| q051p (Fine Arts) | D | B | B | B | B | **B** | Opus ZS, Phi, Qwen |
| q063p (replacement cost) | C | C | C | A | C | **B** | NONE |
| q081 (snowmobile) | D | D | D | C | D | **A** | NONE |

**Only 3 of 11 are recoverable** from existing runs: q015, q041, q051p.
The other 8 are wrong across ALL models — no voting can fix them.

## Voting rules tested

### Rule 1: Hedging-based confidence (citations minus hedges)

**Logic:** When R1 and R2 disagree, pick the answer from whichever
response has more citations and fewer hedging phrases.

**Result:** 146/169 = 86.39% (−2.37pp). **Made things worse.**

The hedging signal is inversely correlated with correctness — the
ensemble (R2) produces longer, more citation-heavy responses that
are often confidently wrong.

### Rule 2: Always pick R1 on disagreement

**Logic:** R1 (rewrite+wiki) is the most calibrated run. When in
doubt, trust it.

**Result:** 150/169 = 88.76%. No change — this is R1's accuracy.

### Rule 3: Always pick R2 on disagreement

**Result:** 149/169 = 88.17%. Slightly worse.

### Rule 4: Loose consensus (ZS + 1 open-source agree, with hedging)

**Logic:** Override wiki answer when:
1. Opus wiki and Opus zero-shot disagree
2. Zero-shot + at least one open-source model agree
3. Wiki response contains hedging language

**Result:** Triggers 5 times. Fixed 2, broke 2. **Net: 0.**

| QID | Wiki | Override to | Result |
| :--- | :--- | :--- | :--- |
| q018 | A → D | **FIXED** |
| q026 | D → A | **BROKE** |
| q071 | A → B | SAME-WRONG |
| q051p | D → B | **FIXED** |
| q058 | B → A | **BROKE** |

### Rule 5: Strict consensus (ZS + BOTH Phi AND Qwen agree, with hedging)

**Logic:** Same as Rule 4 but require both open-source models to
agree with zero-shot.

**Result:** Triggers 5 times. Fixed 2, broke 2. **Net: 0.**

Same as Rule 4 because the additional Qwen filter didn't remove
either broken case.

### Rule 6: Unanimous 4-vs-1 (all non-wiki models agree)

**Logic:** Override wiki answer only when ALL four non-wiki models
(2× Opus ZS + Phi + Qwen) unanimously agree on a different answer.
No hedging requirement — pure consensus.

**Result:** Triggers 4 times. Fixed 2, broke 1. **Net: +1.**

| QID | Wiki | Override to | Result |
| :--- | :--- | :--- | :--- |
| q018 | A → D | **FIXED** |
| q026 | D → A | **BROKE** |
| q071 | A → B | SAME-WRONG |
| q051p | D → B | **FIXED** |

**Final: 151/169 = 89.35%.** The only rule with a positive net effect.

The one BROKE case (q026) is notable: all four non-wiki models
unanimously agree on A, but the correct answer is D. The crowd is
confidently wrong — this question involves a regulatory detail where
the wiki (correctly) has the right answer but the non-wiki models
all share the same wrong pretraining knowledge.

## Why voting can't reach 90%

The arithmetic is simple:
- 150 correct in the best run (R1)
- 3 recoverable from other runs (q015, q041, q051p)
- Maximum with perfect voting: 153/169 = **90.53%**
- But every rule that recovers q015 and q051p also risks q026 and
  q058 (false positives with identical signal shape)
- The best safe rule (Rule 6, unanimous 4-vs-1) reaches 89.35%

**To reach 90% via voting, we would need either:**
1. A new run that gets q026 or q071 right (breaking the unanimous
   wrong consensus)
2. A signal that distinguishes q026 (crowd wrong) from q018/q051p
   (crowd right) — no such signal exists in the response text

## Pattern analysis: when is the crowd right vs wrong?

We examined whether response features (hedging count, citation count,
response length) predict which side is correct in disagreements.

| When rewrite wins (9 cases) | When ensemble wins (8 cases) |
| :--- | :--- |
| Avg rewrite hedges: 2.8 | Avg rewrite hedges: 1.6 |
| Avg rewrite citations: 4.6 | Avg rewrite citations: 4.9 |
| Avg ensemble hedges: 2.2 | Avg ensemble hedges: 2.6 |
| Avg ensemble citations: 11.9 | Avg ensemble citations: 5.0 |

**No reliable signal.** More citations does not predict correctness.
The ensemble often generates long, citation-heavy responses that are
confidently wrong. This is consistent with the literature on LLM
calibration — model confidence and response length are poor proxies
for correctness on knowledge-intensive tasks (Kadavath et al. 2022).

## Conclusion

Multi-model voting provides at most +1 question (+0.59pp) via the
unanimous 4-vs-1 rule. The remaining gap to 90% cannot be closed by
voting because:

1. **8 of 11 hard questions are wrong across ALL models** — no
   voting scheme can recover an answer no model has
2. **5 of those 8 are outside our study corpus** (homeowners
   insurance) — a data problem, not a model problem
3. **Response-surface confidence signals don't predict correctness**
   — hedging, citations, and length are unreliable

The bottleneck has shifted from model capability to **corpus
completeness**. See `docs/ROOT_CAUSE_ANALYSIS.md` for the full
data coverage analysis.

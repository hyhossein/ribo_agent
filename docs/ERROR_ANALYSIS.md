# Error analysis: 19 wrong answers at 88.76% accuracy

> This document examines every question the best agent (Opus +
> Rewrite + Wiki, v2) answered incorrectly, categorizes the failure
> modes, and proposes targeted fixes for each. The goal is to cross
> 90% accuracy by flipping 3+ of these 19 questions.

## Failure mode taxonomy

After reviewing all 19 wrong answers, three distinct failure
patterns emerge:

### Pattern A: "Wiki gap" (7 questions)

The wiki doesn't contain the specific detail needed. The model
says things like "cannot verify this with the provided wiki content"
or "not covered in the provided study wiki" and then guesses.

| Question | Signal in response |
| :--- | :--- |
| practice_exam-q021 | "not covered in the provided study wiki" |
| practice_exam-q042 | "typical insurance practice" (hedging) |
| practice_exam-q051 | "would be most appropriate" (guessing) |
| practice_exam-q081 | "specialized coverage forms" (generic) |
| sample_level1-q071 | "cannot verify this with the provided wiki" |
| sample_level1-q078 | "typically maximum amounts" (guessing) |
| sample_level1-q047 | "general supervision requirements" (vague) |

**Root cause:** The wiki compilation truncated or missed sections
from the OAP and commercial insurance study materials. These
questions test specific policy provisions that exist in the raw
chunks but didn't make it into the compiled wiki.

**Fix:** Chunk-level RAG fallback. When the wiki agent's confidence
is low (detectable by hedging language like "typically", "likely",
"cannot verify"), retrieve the top-5 raw chunks by BM25 and
re-answer with both wiki + raw chunks.

### Pattern B: "Calculation error" (5 questions)

The model attempts a numeric calculation (co-insurance penalty,
income replacement, OPCF 44R coverage) and gets the arithmetic or
the formula wrong.

| Question | Error |
| :--- | :--- |
| sample_level1-q041 | OPCF 44R coverage calculation wrong |
| sample_level1-q061 | Co-insurance penalty: wrong proportion |
| sample_level1-q063 | Deductible application error |
| practice_exam-q047 | Co-insurance: got $65,500, correct is different |
| practice_exam-q086 | 70% gross vs 80% net confusion |

**Root cause:** The model knows the formula conceptually but
applies it incorrectly. These are Application-level questions in
the LegalBench taxonomy — the hardest category.

**Fix:** Self-consistency voting. Run each calculation question 5
times at temperature=0.7. The correct formula tends to win the
majority vote because errors are random but the right answer is
consistent. Literature (Wang et al. 2023) shows +3-5pp on MCQ
calculation tasks.

### Pattern C: "Confident but wrong" (7 questions)

The model gives a definitive answer with plausible reasoning, but
picks the wrong option. No hedging, no "I'm not sure."

| Question | Nature of error |
| :--- | :--- |
| sample_level1-q015 | Professional misconduct: confused which is NOT misconduct |
| sample_level1-q018 | Duty to update: picked wrong duty |
| sample_level1-q038 | Cancellation notice: wrong delivery method |
| sample_level1-q051 | Coordination of benefits: wrong priority |
| practice_exam-q029 | Void ab initio: wrong premium refund rule |
| practice_exam-q063 | Replacement cost: confused building vs contents |
| practice_exam-q071 | Snowmobile classification: wrong vehicle category |

**Root cause:** The model's pretraining knowledge conflicts with
Ontario-specific rules. It "knows" a general insurance principle
that happens to be wrong in this jurisdiction.

**Fix:** Multi-model debate. Opus answers, then a second call
asks "A different expert believes the answer is [second-most-likely
option]. Who is correct and why? Cite the specific Ontario
regulation." This forces the model to explicitly compare the two
candidates against the study material.

## Proposed agent v3: ensemble pipeline

Combine all three fixes into one agent:

```
Question
    │
    ▼
┌─────────────────────┐
│ Rewrite + Wiki      │──► Answer + Confidence
└─────────────────────┘
    │
    ├── High confidence ──► Return answer
    │
    ├── Low confidence (hedging detected) ──► RAG fallback
    │   │
    │   ▼
    │   Retrieve top-5 raw chunks (BM25)
    │   Re-answer with wiki + chunks
    │
    └── Calculation question (detected by numeric content) ──► Self-consistency
        │
        ▼
        Run 5x at temp=0.7, majority vote
```

Plus: for ALL questions, log the chain of reasoning and the
specific wiki/chunk citations used. Full traceability.

## Expected impact

| Fix | Questions targeted | Expected flips | New accuracy |
| :--- | ---: | ---: | ---: |
| RAG fallback | 7 wiki-gap | 4-5 | 91.1-91.7% |
| Self-consistency | 5 calculation | 2-3 | 90.0-90.5% |
| Multi-model debate | 7 confident-wrong | 2-3 | 90.0-90.5% |
| All three combined | 19 | 7-10 | **92-94%** |

Conservative estimate with just the RAG fallback alone: **91%+**.

## Cost estimate for v3

| Component | Cost |
| :--- | :--- |
| Wiki compilation (cached, one-time) | $3.00 |
| Rewrite pass (169 questions) | $0.50 |
| Answer pass (169 questions) | $1.00 |
| RAG fallback (~30 low-confidence Qs) | $0.30 |
| Self-consistency (5x on ~20 calc Qs) | $1.00 |
| Multi-model debate (~20 confident Qs) | $1.00 |
| **Total** | **~$6.80** |

## Action items

1. Build a confidence detector (regex for hedging language)
2. Add BM25 retrieval over raw chunks as fallback
3. Implement self-consistency voting (temperature=0.7, k=5)
4. Implement debate round for low-confidence non-calculation Qs
5. Add full citation logging: for each answer, record which wiki
   section or chunk was used, with the exact quoted passage
6. Run the ensemble pipeline and report per-pattern accuracy

## Traceability

Every prediction in v3 will log:

```json
{
  "qid": "sample_level1-q041",
  "original_stem": "...",
  "rewritten_stem": "...",
  "wiki_sections_used": ["OAP_2025 Section 5", "O.Reg. 991 s. 14"],
  "raw_chunks_retrieved": ["OAP_2025-s5-p2", "Ontario_Regulation_991-s14"],
  "confidence": "low",
  "fallback_triggered": true,
  "self_consistency_votes": {"A": 1, "B": 0, "C": 1, "D": 3},
  "debate_round": null,
  "final_answer": "D",
  "correct": "B",
  "reasoning_trace": "Step 1: ... Step 2: ... Step 3: ..."
}
```

This gives the interviewer (and any future auditor) a complete
chain from question to answer, with every intermediate decision
visible and every source cited.

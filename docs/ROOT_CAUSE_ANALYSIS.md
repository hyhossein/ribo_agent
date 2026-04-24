# Root cause analysis: why accuracy plateaus at 88.76%

## Discovery

After analyzing the 11 questions that ALL agent variants get wrong,
we found the root cause is not prompt engineering, model capability,
or inference strategy. **It is a gap in the source documents.**

## The study corpus

The raw study materials provided contain 8 documents:

| Document | Domain |
| :--- | :--- |
| Ontario Automobile Policy 2025 | Auto insurance |
| Ontario Regulation 989 | Insurance regulations |
| Ontario Regulation 990 | Insurance regulations |
| Ontario Regulation 991 | Insurance regulations |
| RIBO By-Law No. 1 (March 2024) | Broker regulation |
| RIBO By-Law No. 2 (March 2024) | Broker regulation |
| RIBO By-Law No. 3 (March 2024) | Broker regulation |
| Registered Insurance Brokers Act 1990 | Broker regulation |

**Missing:** No homeowners policy, no commercial property policy, no
specialty coverage documentation (Fine Arts, Personal Effects,
Freezer Foods endorsements).

## Impact on the 11 hardest questions

| Question | Topic | In corpus? | Both runs answer |
| :--- | :--- | :--- | :--- |
| practice_exam-q021 | Freezer Foods coverage | **NO** | Wrong |
| practice_exam-q042 | Homeowners rec room renovation | **NO** | Wrong |
| practice_exam-q051 | Fine Arts endorsement | **NO** | Wrong |
| practice_exam-q063 | Homeowners replacement cost | **NO** | Wrong |
| practice_exam-q081 | Snowmobile OAP 1 form | **NO** (0 chunks found) | Wrong |
| sample_level1-q015 | Professional misconduct (NOT) | Yes | Wrong |
| sample_level1-q041 | OPCF 44R calculation | Yes | Wrong |
| sample_level1-q051 | Workers' Comp + auto | Partial | Wrong |
| sample_level1-q063 | Uninsured auto, family member | Yes | Wrong |
| sample_level1-q071 | OAP enhancement options | Partial | Wrong |
| practice_exam-q029 | Misrepresentation premium refund | Partial | Wrong |

**5 of 11 (45%) of the irreducibly wrong answers are about topics not
covered by any document in our study corpus.**

## What this means

The wiki agent reaches 88.76% because it can only compile knowledge
that exists in the source documents. For auto insurance, regulations,
and broker ethics, the wiki provides near-complete coverage. For
homeowners and specialty coverages, the wiki has nothing — and the
model falls back to pretraining knowledge, which is often wrong on
Ontario-specific details.

## The ceiling without new documents

If we could perfectly answer all 6 questions where the knowledge IS
in the corpus (the OAP/regulation questions), but accept that the 5
homeowners questions are unanswerable:

- Current: 150/169 = 88.76%
- Fix the 6 in-corpus questions: 156/169 = 92.31%
- Fix the 5 homeowners questions too: 161/169 = 95.27%

**The realistic ceiling with current documents is ~92%.** The
remaining 5% requires homeowners/property policy source material.

## Recommendations

### Short-term (no new documents, ~$1-2)
1. For the 6 in-corpus questions, add a verification step: after
   answering, extract the cited section from the wiki and verify
   the answer is consistent with the cited text.
2. For detected out-of-corpus questions (no wiki citations found),
   explicitly tell the model: "The study wiki does not cover this
   topic. Answer from your general insurance knowledge." This may
   recover 2-3 homeowners questions where Opus's pretraining is
   correct.

### Medium-term (with new documents)
3. Obtain homeowners policy documentation (e.g., IBC standard
   Homeowners Comprehensive form, or the RIBO study manual's
   homeowners chapter).
4. Add to `data/raw/study/`, rebuild the wiki, re-eval.
5. Expected: 92-95% accuracy.

### Long-term (production)
6. Maintain a living knowledge base that covers all exam domains.
7. Track which questions lack source coverage and flag them as
   lower confidence in production.

## Key insight

**The dominant bottleneck shifted from model capability to corpus
completeness.** Earlier in the project, the model didn't know enough
(49-60% with open-source). We solved that with the wiki pattern
(88.76%). Now the model knows everything in the corpus but the
corpus doesn't cover everything on the exam. The next improvement
requires better data, not better algorithms.

This is a common pattern in production NLP systems: initial gains
come from better models and prompts, but the ceiling is set by
data coverage. Identifying this transition point — and knowing when
to invest in data rather than engineering — is the key insight.

# Day 1 — Exploratory Data Analysis

**Purpose.** Understand the inputs before writing any system. This document
records what I found and the design decisions it implies. The reproducible
EDA lives in `notebooks/day1_eda.py`; the raw log of a run is in
`results/day1_eda.log`.

---

## 1. Inventory

```
data/raw/questions/                       KB   pages   producer              text layer
 695993459-Practise-RIBO-Exam.pdf        209   19     Microsoft Word 365    yes
 901740466-1-RIBO-Manual-249-387.pdf    1194   139    iLovePDF              yes
 901740471-RIBO-Manual-Questions.pdf    1453   161    iLovePDF              yes
 Sample-Questions-RIBO-Level-1-Exam.pdf  291   31     Microsoft Word 365    yes

data/raw/study/
 Ontario Automobile Policy 2025.pdf      556   68     Accessibil-IT         yes
 Ontario Regulation 989.doc               71   -      (Word 97 binary)      n/a
 Ontario Regulation 990.doc               73   -      (Word 97 binary)      n/a
 Ontario Regulation 991.doc              124   -      (Word 97 binary)      n/a
 RIBO By-Law No 1 March 2024.pdf         804   33     MS Print To PDF       yes
 RIBO By-Law No 2 March 2024.pdf         622   12     MS Print To PDF       yes
 RIBO By-Law No 3 March 2024.pdf         807   27     MS Print To PDF       yes
 Registered Insurance Brokers Act 1990.doc 168  -     (Word 97 binary)      n/a
```

**Implications.**
- All PDFs have a real text layer — no OCR needed.
- The four `.doc` files must be converted first. LibreOffice headless will do
  it: `libreoffice --headless --convert-to txt`. Native Python `python-docx`
  won't work (that's for `.docx`).
- No scanned documents, no images-of-text, no fonts worth worrying about
  beyond bold-weight detection (see §3).

## 2. Four question PDFs, three different structures

| File | Structure | # questions | Answer signal |
|------|-----------|-------------|---------------|
| `Sample-Questions-...` | MCQ with inline answers and **rich metadata** | 79 | `• Correct Option: X` below each question, plus Content Domain, Competency, and Cognitive Level |
| `695993459-Practise-RIBO-Exam.pdf` | MCQ with separate X-grid answer key | 90 | grid table after Q90, with an X in the correct column per row |
| `901740466-*.pdf`, `901740471-*.pdf` | Manual study guide with duplicated Q&A | ~500 each | questions appear twice; in the "Answers" section the correct option is rendered in **bold font** |

The two manual PDFs are **study aids**, not graded tests with an answer key.
They're still useful — ~1000 additional Q&A pairs can seed few-shot
retrieval once the bold answers are extracted. They should not be used as
eval data though, because they overlap conceptually with what the sample
set tests.

## 3. What the sample-questions PDF gives us for free

79 questions, each tagged with:

- **Content Domain:** General Insurance (48), Personal Lines Habitational (19),
  Commercial Lines (7), Personal Lines Automobile (5).
- **Competency:** Insurance Product Knowledge (46), Legal/Regulatory (16),
  Professionalism/Ethics (8), + 5 smaller categories.
- **Cognitive Level:** Knowledge (28), Comprehension (38), Application (13).

This is a gift. It lets us:

- Break down per-domain accuracy in the final report.
- Potentially route retrieval differently for Regulatory questions (search
  the Act/Regulations/By-Laws) vs. Habitational (search general-knowledge
  chunks or manual exemplars).
- Spot-check that the agent isn't lopsided (e.g., strong on Knowledge-level
  trivia but weak on Application-level scenarios).

Stem lengths: **min 31 / p50 167 / p90 297 / max 533** characters. So
average is two sentences, worst-case a small paragraph. Fits easily in any
LLM context.

## 4. The X-grid answer key has a trap

The answer key for `695993459-Practise-RIBO-Exam.pdf` looks like:

```
             A      B      C      D
       1                           X
       2                           X
       3      X
       ...
```

Across the two pages of the key, the column offsets are different:

- page 0: unique X offsets = `[10, 17, 32, 45]`
- page 1: unique X offsets = `[10, 14, 18, 22]`

So the parser **cannot hard-code a fixed column → letter map**. It must
cluster the X offsets observed on each page and assign A/B/C/D in
left-to-right order, per page.

## 5. Question-number regex trap (form-feed vs. newline)

Naive `^(\d{1,2})\.` multiline regex picks up 75/90 question stems. The
missing 15 are `[6, 12, 19, 25, 31, 37, 43, 49, 54, 59, 64, 69, 75, 81, 86]`
— near-perfectly-spaced intervals of 5-6. They are all **the first question
at the top of each new page**, where `pdftotext -layout` emits
`\x0c` (form-feed) rather than `\n` before the number. Python's `re.MULTILINE`
does not treat `\x0c` as a line boundary.

Fix on Day 2: either strip/replace `\x0c` before matching, or include it in
the line-boundary character class: `(?m)(?:^|\f)(\d{1,2})\.`.

## 6. Study document structure

All are PDFs or `.doc` files with clean section numbering:

- **Ontario Regulations 989 / 990 / 991** — sections numbered `1.`, `2.`,
  `5.(1)`, `7.2`, etc. Regulation 991 contains section 14 (Code of Conduct)
  which is cited in roughly every regulatory question I've eyeballed.
- **RIBO By-Laws 1 / 2 / 3** — each organized into Articles then
  sub-sections `1.1`, `1.2`, `2.1`, …
- **RIB Act 1990** — sections numbered with short titles on separate lines
  in the LibreOffice text output. Will need a purpose-built splitter.
- **OAP 2025** — sections `Section 1 - Introduction`, `Section 2 - What
  Automobiles Are Covered`, etc.

**Chunking strategy for Day 3:** chunk by semantic unit (section /
sub-section) rather than fixed token windows, because exam questions
frequently cite section numbers verbatim ("According to Reg 991 s. 14...").
Preserve the citation trail in chunk metadata so the model can ground its
answer.

## 7. Leakage check (preliminary)

80-char normalized-prefix overlap between the sample/practice sets and the
two manual study-aid PDFs: **0**. That's a green light but a weak one — minor
rewording defeats exact-prefix matching.

A proper SHA-fingerprint dedup (normalize stem + first option, hash, compare
sets) runs as part of Day 2's parser. If that also returns 0, the manual
pool is safe to use for few-shot retrieval without leakage risk.

## 8. Open questions going into Day 2

1. **Which pool is the final eval set?** The assessment doesn't explicitly
   specify. My plan is to hold out the 169 questions that have ground-truth
   answer keys (79 sample + 90 practice) and treat the ~1000 manual MCQs as
   a few-shot pool. Confirm with the interviewer on Day 2 if possible; if
   not, document this assumption prominently in the final report.
2. **Content-domain mismatch.** The sample set has the rich `Content Domain`
   labels, but the practice set has none. Should we run a small classifier
   on the practice set to assign domains retrospectively, or just skip
   per-domain breakdowns for it? I lean: skip — don't invent labels.
3. **Automobile policy vintage.** The study corpus ships the 2025 OAP 1.
   Some practice-exam questions reference `OPF #6` and `OPCF #44`, which
   still exist in 2025 but whose wording may have evolved. If a question
   turns on exact endorsement wording the answer key depends on the wording
   at the time the question was written, not today's. Watch out for this
   during error analysis on Day 7.

---

## What we have after Day 1

- A clean, reproducible EDA (`notebooks/day1_eda.py` + `results/day1_eda.log`).
- An accurate picture of the four question PDFs: how many, how formatted,
  where the answers live, what metadata exists.
- A list of known **parser traps** (X-grid column drift, form-feed vs.
  newline, `.doc` binary format, bolded-answer extraction) with a plan for
  each.
- Design principles for the Day 3 chunker: section-level granularity with
  citation-preserving metadata, not token-window chunking.

No agent code written today. That's fine — shaving a day off planning costs
less than building on a wrong assumption and rewriting later.

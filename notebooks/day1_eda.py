"""Day 1 EDA — exploratory data analysis of the RIBO corpus.

Run: `python notebooks/day1_eda.py`. The script is written as a notebook
would be, with `# %%` cell markers, so it can also be opened in VS Code /
Jupytext and explored interactively.

Goals for today:

1. Know exactly what files we have, how big they are, and whether the PDFs
   are text-based or scanned.
2. Classify the four question PDFs by structure (graded MCQs with known
   answers vs. study-aid material).
3. Profile question length, option length, and any per-question metadata
   (content domain, competency, cognitive level).
4. Eyeball the structure of each study document — what sections are there,
   are section numbers citable, is OCR needed.
5. Produce findings that drive Day 2's design decisions, written to
   `docs/EDA.md`.

No LLM calls, no parsing into JSONL, no embeddings. Just looking.
"""

# %% imports & setup
from __future__ import annotations

import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
QUESTIONS = RAW / "questions"
STUDY = RAW / "study"


def sh(cmd: list[str]) -> str:
    """Run a shell command, return stdout as str, raise on non-zero."""
    r = subprocess.run(cmd, capture_output=True, check=True)
    return r.stdout.decode("utf-8", errors="replace")


def pdfinfo(path: Path) -> dict:
    """Parse `pdfinfo` output into a dict."""
    out = sh(["pdfinfo", str(path)])
    d: dict[str, str] = {}
    for line in out.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            d[k.strip()] = v.strip()
    return d


def pdf_text(path: Path, layout: bool = True) -> str:
    cmd = ["pdftotext"]
    if layout:
        cmd.append("-layout")
    cmd.extend([str(path), "-"])
    return sh(cmd)


# %% 1. file inventory
print("=" * 70)
print("1. FILE INVENTORY")
print("=" * 70)


def inventory(dir_: Path) -> None:
    for p in sorted(dir_.iterdir()):
        size_kb = p.stat().st_size / 1024
        ext = p.suffix.lower()
        if ext == ".pdf":
            info = pdfinfo(p)
            pages = info.get("Pages", "?")
            producer = info.get("Producer", "?")
            # is the PDF text-extractable?
            first_page = pdf_text(p, layout=False)[:500]
            is_text = len(first_page.strip()) > 50 and any(c.isalpha() for c in first_page)
            flag = "text" if is_text else "scanned?"
            print(f"  {p.name:55s} {size_kb:7.1f} KB  {pages:>4s}p  [{flag}]  {producer[:30]}")
        else:
            print(f"  {p.name:55s} {size_kb:7.1f} KB  ({ext[1:]})")


print("\n-- data/raw/questions/")
inventory(QUESTIONS)
print("\n-- data/raw/study/")
inventory(STUDY)


# %% 2. classify question files
print("\n" + "=" * 70)
print("2. QUESTION-FILE CLASSIFICATION")
print("=" * 70)
# Quick heuristic: look at the first and last slices of each PDF to decide
# what structure we're dealing with.
for pdf in sorted(QUESTIONS.glob("*.pdf")):
    full = pdf_text(pdf)
    head = full[:8000]
    tail = full[-4000:]
    n_correct_option = full.count("Correct Option")
    n_option_markers = len(re.findall(r"\b[oO]\s+[A-D]\.", head))
    n_lowercase_opts = len(re.findall(r"^\s*[a-d]\)", head, flags=re.MULTILINE))
    has_xgrid = "Broker Practice Exam Answers" in tail or "Broker Practice Exam Answers" in full

    if n_correct_option > 5:
        kind = f"MCQ with inline answers + metadata ({n_correct_option} 'Correct Option' markers)"
    elif has_xgrid:
        kind = "MCQ with separate X-grid answer key"
    elif n_lowercase_opts > 3:
        kind = f"Manual-style MCQ a/b/c/d ({n_lowercase_opts} option-markers on first page; answers likely bolded)"
    else:
        kind = "UNCLASSIFIED"
    print(f"  {pdf.name[:55]:55s} -> {kind}")


# %% 3. profile the "sample questions" PDF (has richest metadata)
print("\n" + "=" * 70)
print("3. SAMPLE QUESTIONS PROFILE (rich metadata)")
print("=" * 70)
sample = QUESTIONS / "Sample-Questions-RIBO-Level-1-Exam (1).pdf"
text = pdf_text(sample)
# count questions, answers, metadata occurrences
n_questions = len(re.findall(r"^Question \d+\.", text, flags=re.MULTILINE))
n_correct = len(re.findall(r"Correct Option:\s*[A-D]", text))
domains = Counter(re.findall(r"Content Domain:\s*(.+)", text))
comps = Counter(re.findall(r"Competency:\s*(.+)", text))
cogs = Counter(re.findall(r"Cognitive Level:\s*(.+)", text))

print(f"  questions found:  {n_questions}")
print(f"  correct options:  {n_correct}")
print(f"\n  content domains:")
for k, v in domains.most_common():
    print(f"    {v:3d}  {k}")
print(f"\n  competencies ({sum(comps.values())} total):")
for k, v in comps.most_common():
    print(f"    {v:3d}  {k}")
print(f"\n  cognitive levels:")
for k, v in cogs.most_common():
    print(f"    {v:3d}  {k}")

# stem length distribution
stems = re.findall(
    r"Question \d+\.\s*(.*?)(?=o A\.|\n\s*o A\.)", text, flags=re.DOTALL
)
stem_chars = [len(" ".join(s.split())) for s in stems]
if stem_chars:
    stem_chars.sort()
    print(f"\n  stem char counts: min={min(stem_chars)} "
          f"p50={stem_chars[len(stem_chars)//2]} "
          f"p90={stem_chars[int(len(stem_chars)*0.9)]} "
          f"max={max(stem_chars)}")


# %% 4. profile the Practice Exam answer grid
print("\n" + "=" * 70)
print("4. PRACTICE EXAM ANSWER GRID INSPECTION")
print("=" * 70)
prac = QUESTIONS / "695993459-Practise-RIBO-Exam.pdf"
ptext = pdf_text(prac)
# where does the answer grid start?
idx = ptext.find("Broker Practice Exam Answers")
grid = ptext[idx:]
# count X's and look at column positions per-page
pages = grid.split("\f")
print(f"  answer key spans {len(pages)} pages (after first)")
for i, p in enumerate(pages):
    x_positions: list[int] = []
    for line in p.splitlines():
        if re.match(r"^\s+\d{1,3}\b", line):
            for m in re.finditer(r"X", line):
                x_positions.append(m.start())
    if x_positions:
        uniq = sorted(set(x_positions))
        print(f"    page {i}: {len(x_positions):2d} X's, "
              f"{len(uniq)} unique column offsets: {uniq}")

# The design implication: answer-key column positions are NOT stable across
# pages, so the parser cannot assume a fixed A/B/C/D → column mapping.
# Each page needs to be clustered independently.


# %% 5. head-vs-tail question numbering to detect gaps
print("\n" + "=" * 70)
print("5. PRACTICE EXAM QUESTION NUMBER COVERAGE")
print("=" * 70)
# Which question numbers can we pick up with naive regex, and where do
# we lose them?
body = ptext[:idx]
# accept "N." or "N " followed by an uppercase letter (captures weird cases
# like "51.Your..." and "52 Your..." that appeared in the raw text).
qnums = [int(m.group(1)) for m in re.finditer(
    r"^(\d{1,2})[\.\s]+[A-Z]", body, flags=re.MULTILINE
)]
# dedup (some numbers appear inside body text)
# we'll take the first occurrence only and require that the number is <= 90
# and arrives in roughly-sorted order.
seen = set()
ordered_unique = []
last = 0
for n in qnums:
    if 1 <= n <= 90 and n not in seen and n >= last - 3:
        seen.add(n)
        ordered_unique.append(n)
        last = n
missing = [n for n in range(1, 91) if n not in seen]
print(f"  question numbers 1-90, found via naive regex: {len(seen)}")
print(f"  missing: {missing}")
# Hypothesis: missing numbers appear with punctuation quirks (no space after
# period, or no period at all) or line wraps.


# %% 6. quick peek at the two big manual PDFs
print("\n" + "=" * 70)
print("6. RIBO MANUAL QUESTION PDFs")
print("=" * 70)
for name in [
    "901740466-1-RIBO-Manual-249-387 (1).pdf",
    "901740471-RIBO-Manual-Questions (1).pdf",
]:
    pdf = QUESTIONS / name
    t = pdf_text(pdf)
    # count question-start markers like "1)"
    n_qstart = len(re.findall(r"^\s*\d{1,3}\)\s", t, flags=re.MULTILINE))
    # count option markers like "a)"
    n_opts = len(re.findall(r"^\s*[a-d]\)\s", t, flags=re.MULTILINE))
    # approximate number of MCQ items (opts / 4)
    approx_mcq = n_opts // 4
    # find section-like labels
    subsec = re.findall(r"\b(?:Principles & Practices|RIBO Rules|Travel Health|"
                        r"Automobile Insurance|Habitational Insurance|"
                        r"Commercial Insurance|Co-Insurance)[^(\n]*\(\d+\)", t)
    print(f"\n  {name}")
    print(f"    question-start markers: {n_qstart}")
    print(f"    option markers:         {n_opts}")
    print(f"    approx MCQ items:       {approx_mcq}")
    print(f"    subsection labels seen: {Counter(subsec)}")


# %% 7. study-doc structure preview
print("\n" + "=" * 70)
print("7. STUDY DOCUMENT STRUCTURE PREVIEW")
print("=" * 70)
print("Only the top-of-document lines per file. Goal: confirm each document")
print("has a clean section structure we can chunk by.\n")
for p in sorted(STUDY.iterdir()):
    print(f"-- {p.name}")
    if p.suffix.lower() == ".pdf":
        head = pdf_text(p)[:600]
    else:
        # .doc files — use libreoffice to convert on the fly
        try:
            with p.open("rb") as f:
                raw_bytes = f.read(4000)
            # crude check: print the first few printable substrings we find
            ascii_ = raw_bytes.decode("latin-1", errors="ignore")
            head = re.sub(r"[^\x20-\x7E\n]+", " ", ascii_)[:400]
        except Exception as e:
            head = f"(could not read: {e})"
    # show only the first few non-empty lines
    lines = [ln.strip() for ln in head.splitlines() if ln.strip()]
    for ln in lines[:6]:
        print(f"    {ln[:110]}")
    print()


# %% 8. leakage check — do manual questions appear verbatim in the sample set?
print("=" * 70)
print("8. LEAKAGE CHECK (manual vs. sample set)")
print("=" * 70)
# Extract the first 80 chars of every stem-looking line and compare sets.
sample_stems: set[str] = set()
for m in re.finditer(r"Question \d+\.\s*(.{20,120})", text):
    norm = re.sub(r"\s+", " ", m.group(1).lower().strip())
    sample_stems.add(norm[:80])

manual_stems: set[str] = set()
for name in [
    "901740466-1-RIBO-Manual-249-387 (1).pdf",
    "901740471-RIBO-Manual-Questions (1).pdf",
]:
    tt = pdf_text(QUESTIONS / name)
    for m in re.finditer(r"^\s*\d{1,3}\)\s*(.{20,120})", tt, flags=re.MULTILINE):
        norm = re.sub(r"\s+", " ", m.group(1).lower().strip())
        manual_stems.add(norm[:80])

overlap = sample_stems & manual_stems
print(f"  unique sample-set stem prefixes:  {len(sample_stems)}")
print(f"  unique manual stem prefixes:      {len(manual_stems)}")
print(f"  exact-prefix overlap:             {len(overlap)}")
if overlap:
    for s in list(overlap)[:3]:
        print(f"    overlap sample: {s!r}")

# Per-80-char prefix is an imperfect check (minor wording differences
# won't match) but it catches the "same question copy-pasted across files"
# case cheaply. A proper SHA-fingerprint dedup comes on Day 2.

print("\nDay 1 EDA complete. See docs/EDA.md for the write-up.")

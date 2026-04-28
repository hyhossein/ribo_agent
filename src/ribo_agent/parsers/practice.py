"""Parse `695993459-Practise-RIBO-Exam.pdf`.

Format: 90 MCQs with options A/B/C/D, an unrelated essay section, then an
answer-key grid:

    Broker Practice Exam Answers
              A      B      C      D
       1                           X
       2                           X
       ...

Day 1 EDA (see docs/EDA.md §§4–5) surfaced two real traps this parser must
design around:

1. **X-grid column offsets drift between pages.** Page 1 of the key has
   offsets [10, 17, 32, 45], page 2 has [10, 14, 18, 22]. No fixed
   column->letter map is safe. We cluster offsets per page, sort left-to-
   right, and assign A/B/C/D.

2. **Form-feed before first question of each page.** 15 question numbers
   (6, 12, 19, 25, 31, 37, 43, 49, 54, 59, 64, 69, 75, 81, 86) sit
   immediately after `\f` rather than `\n`, defeating naive ^-anchored
   regex. We replace `\f` with `\n` before matching.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .schema import MCQ


# line-of-interest patterns --------------------------------------------------

# "1. Following ..." / "51.Your ..." (no space) / "52 Your ..." (no period)
QSTART_RE = re.compile(r"^(\d{1,2})[\.\s]+([A-Z].*)")

# " A. Option text" — indent 0-3 spaces, letter A-D, period, space
OPT_RE = re.compile(r"^\s{0,3}([A-D])\.\s+(.*)")

ANSWER_HEADER = "Broker Practice Exam Answers"

# ignorable page noise (running header, date stamps, footers)
_NOISE = [
    re.compile(r"^RIBO Practice Exam Licensing Course$"),
    re.compile(r"^\d+/\d+/\d+\s*$"),
    re.compile(r"^\d+/\d+/\d+\s+\d+$"),
]


def _pdftotext_layout(pdf: Path) -> str:
    r = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"],
        capture_output=True, check=True,
    )
    return r.stdout.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# answer-key extraction
# ---------------------------------------------------------------------------

def _extract_answer_key(text: str) -> dict[int, str]:
    """Return {question_number: 'A'|'B'|'C'|'D'} by scanning the grid pages.

    Strategy: split the post-header region by form-feed. For each page,
    gather (qnum, x_offset_of_X) pairs; the four unique offsets cluster
    cleanly and map left-to-right to A/B/C/D.
    """
    idx = text.find(ANSWER_HEADER)
    if idx < 0:
        raise RuntimeError("answer key header not found")
    tail = text[idx:]

    answers: dict[int, str] = {}
    for page in tail.split("\f"):
        rows: list[tuple[int, int]] = []
        for line in page.splitlines():
            # line must START with an indented 1-3 digit number
            mn = re.match(r"^\s+(\d{1,3})\b", line)
            if not mn:
                continue
            qnum = int(mn.group(1))
            if not 1 <= qnum <= 200:
                continue
            for xm in re.finditer(r"X", line):
                rows.append((qnum, xm.start()))

        if not rows:
            continue
        offsets = sorted({off for _, off in rows})
        # If more than 4 distinct offsets (unlikely) keep the 4 most populated.
        if len(offsets) > 4:
            from collections import Counter
            top4 = {off for off, _ in Counter(off for _, off in rows).most_common(4)}
            offsets = sorted(top4)
        letter_for = {off: "ABCD"[i] for i, off in enumerate(offsets)}
        for qnum, off in rows:
            if off in letter_for:
                answers[qnum] = letter_for[off]
    return answers


# ---------------------------------------------------------------------------
# MCQ body extraction
# ---------------------------------------------------------------------------

def _is_noise(s: str) -> bool:
    return any(p.match(s) for p in _NOISE)


def _walk_body(body: str) -> list[dict]:
    """Extract MCQs from the pre-answer-key body as plain dicts."""
    # Replace form-feed with newline so multi-line regex sees page-top numbers.
    body = body.replace("\f", "\n")
    lines = [ln for ln in body.splitlines() if not _is_noise(ln.strip())]

    out: list[dict] = []
    current: dict | None = None
    current_letter: str | None = None

    def flush() -> None:
        if current is not None and len(current["options"]) == 4:
            current["stem"] = " ".join(current.pop("stem_parts")).strip()
            out.append(current)

    for i, ln in enumerate(lines):
        stripped = ln.strip()

        qm = QSTART_RE.match(stripped)
        if qm:
            qnum = int(qm.group(1))
            # Require 1..90 AND a blank line immediately before (protects
            # against sentences inside stems that happen to start with a
            # small integer).
            prev_blank = i == 0 or not lines[i - 1].strip()
            if 1 <= qnum <= 90 and prev_blank:
                flush()
                current = {
                    "qnum": qnum,
                    "stem_parts": [qm.group(2).strip()],
                    "options": {},
                }
                current_letter = None
                continue

        if current is None:
            continue

        om = OPT_RE.match(ln)
        if om:
            current_letter = om.group(1)
            current["options"][current_letter] = om.group(2).strip()
            continue

        # continuation line
        if stripped:
            if current_letter is None:
                current["stem_parts"].append(stripped)
            else:
                current["options"][current_letter] = (
                    current["options"][current_letter] + " " + stripped
                ).strip()

    flush()
    return out


def parse(pdf_path: Path, source: str = "practice_exam") -> list[MCQ]:
    text = _pdftotext_layout(pdf_path)

    # split off the answer-key region
    idx = text.find(ANSWER_HEADER)
    if idx < 0:
        raise RuntimeError("answer key header not found")
    body_text = text[:idx]

    answers = _extract_answer_key(text)
    raw = _walk_body(body_text)

    out: list[MCQ] = []
    for q in raw:
        correct = answers.get(q["qnum"])
        if not correct:
            continue  # no key => skip; never emit a fake answer
        out.append(MCQ(
            qid=f"{source}-q{q['qnum']:03d}",
            source=source,
            stem=q["stem"],
            options=q["options"],
            correct=correct,
        ))
    return out

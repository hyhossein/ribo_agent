"""Parse `Sample-Questions-RIBO-Level-1-Exam.pdf`.

Format (79 questions, consistent throughout):

    Question N. <stem ...>
        o A. <option ...>
        o B. <option ...>
        o C. <option ...>
        o D. <option ...>
    Review Information
    • Correct Option: X
    • Content Domain: ...
    • Competency: ...
    • Cognitive Level: ...

Stems and options can wrap across lines and occasionally across pages. The
parser walks the `pdftotext -layout` output as a simple state machine.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .schema import MCQ


QUESTION_RE = re.compile(r"^Question (\d+)\.\s*(.*)")
OPTION_RE = re.compile(r"^\s*o\s+([A-D])\.\s*(.*)")
CORRECT_RE = re.compile(r"Correct Option:\s*([A-D])")
DOMAIN_RE = re.compile(r"Content Domain:\s*(.+)")
COMPETENCY_RE = re.compile(r"Competency:\s*(.+)")
COGNITIVE_RE = re.compile(r"Cognitive Level:\s*(.+)")

_PAGE_HEADER_PATTERNS = [
    re.compile(r"^RIBO Level 1 Exam"),
    re.compile(r"^Page \d+ of \d+$"),
    re.compile(r"^LEVEL 1: ENTRY-LEVEL EXAM SAMPLE QUESTIONS$"),
]


def _pdftotext_layout(pdf: Path) -> str:
    r = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"],
        capture_output=True, check=True,
    )
    return r.stdout.decode("utf-8", errors="replace")


def _clean_lines(text: str) -> list[str]:
    # Strip form-feed page markers (see docs/EDA.md §5 — `\f` breaks line
    # boundaries otherwise).
    text = text.replace("\f", "\n")
    out: list[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if any(p.match(s) for p in _PAGE_HEADER_PATTERNS):
            continue
        out.append(ln)
    return out


def parse(pdf_path: Path, source: str = "sample_level1") -> list[MCQ]:
    """Return every MCQ in the Sample Questions PDF, in document order."""
    lines = _clean_lines(_pdftotext_layout(pdf_path))
    n = len(lines)
    out: list[MCQ] = []
    i = 0

    while i < n:
        m = QUESTION_RE.match(lines[i].strip())
        if not m:
            i += 1
            continue

        qnum = int(m.group(1))
        stem_parts: list[str] = [m.group(2).strip()]
        options: dict[str, str] = {}
        i += 1

        # Collect stem lines until the first option.
        while i < n and not OPTION_RE.match(lines[i]):
            s = lines[i].strip()
            if s:
                stem_parts.append(s)
            i += 1

        # Collect options until "Review Information".
        current_letter: str | None = None
        while i < n:
            om = OPTION_RE.match(lines[i])
            if om:
                current_letter = om.group(1)
                options[current_letter] = om.group(2).strip()
                i += 1
                continue
            s = lines[i].strip()
            if s.startswith("Review Information"):
                break
            if current_letter and s:
                options[current_letter] = (options[current_letter] + " " + s).strip()
            i += 1

        # Walk Review Information until the next Question or EOF.
        correct: str | None = None
        domain = competency = cognitive = None
        while i < n and not QUESTION_RE.match(lines[i].strip()):
            s = lines[i].strip()
            if mm := CORRECT_RE.search(s):
                correct = mm.group(1)
            elif mm := DOMAIN_RE.search(s):
                domain = mm.group(1).strip()
            elif mm := COMPETENCY_RE.search(s):
                competency = mm.group(1).strip()
            elif mm := COGNITIVE_RE.search(s):
                cognitive = mm.group(1).strip()
            i += 1

        if correct and len(options) == 4:
            out.append(MCQ(
                qid=f"{source}-q{qnum:03d}",
                source=source,
                stem=" ".join(stem_parts).strip(),
                options=options,
                correct=correct,
                content_domain=domain,
                competency=competency,
                cognitive_level=cognitive,
            ))
    return out

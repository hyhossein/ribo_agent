"""Extract MCQs from the two RIBO Licensing Program manual PDFs.

These PDFs are study aids. Each question appears twice: once plain, and
once in an "Answers" section where the correct option is rendered in
**bold font** (Calibri-Bold, or PyMuPDF span flag bit 4 / value 16).

We only process the Answers regions and infer the correct letter from
font weight. Output joins the ~1000 extracted MCQs into a few-shot /
RAG exemplar pool — NOT the eval set.

Two different layouts:

1. `901740471-RIBO-Manual-Questions.pdf` — explicit "Answers to <topic>
   Sample Exam Questions" page headers.

2. `901740466-1-RIBO-Manual-249-387.pdf` — one big "More Sample Exam
   Questions & Case Studies (Answers)" region whose sub-sections use
   labels like "Principles & Practices of Insurance (40)".

Both use the same (a)/(b)/(c)/(d) option format underneath.

Hard rule: if no option is bold, we SKIP the question — never guess.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF

from .schema import MCQ


# explicit answer-section titles in the 471 PDF
_ANSWER_HEADERS_471 = [
    "Answers to Principles & Practices of Insurance Sample Exam Questions",
    "Answers to RIBO Rules & Regulations Sample Exam Questions",
    "Answers to Travel Health Insurance Sample Exam Questions",
    "Answers to Automobile Insurance Sample Exam Questions",
    "Answers to Habitational Insurance Sample Exam Questions",
    "Answers to Co-Insurance & Commercial Insurance Sample Exam Questions",
]

# headers that mark the end of an MCQ region
_STOP_SUBSTRINGS = [
    "Case Studies Answers",
    "Case Study Questions",
    "Answers to 50 Things You Need to Know",
    "Things You Need to Know",
]

# In the 466 PDF the answers region begins here and is subdivided by labels
# like "Principles & Practices of Insurance (40)".
_ANSWERS_ROOT_466 = "More Sample Exam Questions & Case Studies (Answers)"
_CASE_STUDIES_HEADER_466 = "Case Studies Answers"
_SUBSECTION_LABEL_RE = re.compile(r"^([A-Z][A-Za-z &,\'\-]+)\s\((\d+)\)$")

# topic running-heads — used to truncate 471 ranges before they bleed into
# adjacent study prose
_TOPIC_HEADS = [
    "Principles & Practices of Insurance",
    "RIBO Rules & Regulations",
    "Travel Health Insurance",
    "Automobile Insurance",
    "Habitational Insurance",
    "Commercial Insurance",
    "Co-Insurance",
]

_QSTART_RE = re.compile(r"^(\d{1,3})\)\s*(.*)")
_OPT_RE = re.compile(r"^\s*([a-d])\)\s*(.*)")

_NOISE_LINES = {
    "RIBO Licensing Program",
    "More Sample Exam Questions & Case Studies",
    "Principles & Practices of Insurance",
    "RIBO Rules & Regulations",
    "Travel Health Insurance",
    "Automobile Insurance",
    "Habitational Insurance",
    "Commercial Insurance",
    "Case Studies Answers",
}

# bold detection threshold — an option with < MIN_BOLD_FRAC bold characters
# is treated as plain. Rules out decorative bolding that sneaks into stems.
MIN_BOLD_FRAC = 0.25


# ---------------------------------------------------------------------------
# span / line helpers
# ---------------------------------------------------------------------------

def _is_bold_span(span: dict) -> bool:
    font = span.get("font", "")
    if "Bold" in font or "Black" in font:
        return True
    # PyMuPDF flag bit 4 (value 16) = synthetic bold.
    return bool(span.get("flags", 0) & 16)


def _page_lines(page) -> list[tuple[str, int, int]]:
    """Return (line_text, bold_char_count, total_char_count) per visual line."""
    d = page.get_text("dict", sort=True)
    out: list[tuple[str, int, int]] = []
    for block in d["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            text_parts: list[str] = []
            bold_chars = 0
            total_chars = 0
            for span in line["spans"]:
                text = span["text"]
                if not text:
                    continue
                text_parts.append(text)
                stripped_len = len(text.strip())
                total_chars += stripped_len
                if _is_bold_span(span):
                    bold_chars += stripped_len
            s = "".join(text_parts)
            if s.strip():
                out.append((s, bold_chars, total_chars))
    return out


def _is_noise(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    if s in _NOISE_LINES:
        return True
    if re.fullmatch(r"\d+", s):  # bare page number
        return True
    return False


# ---------------------------------------------------------------------------
# range finders
# ---------------------------------------------------------------------------

def _find_ranges_471(doc) -> list[tuple[str, int, int]]:
    """Return (title, start_page, end_page) triples for the 471 PDF."""
    hits: list[tuple[int, str, str]] = []  # (page, title, kind)
    for i in range(doc.page_count):
        top = "\n".join(doc[i].get_text().splitlines()[:8])
        for title in _ANSWER_HEADERS_471:
            if title in top:
                hits.append((i, title, "start"))
        for stop in _STOP_SUBSTRINGS:
            if stop in top:
                hits.append((i, stop, "stop"))

    # dedup preserving order
    seen: set[tuple[int, str]] = set()
    deduped: list[tuple[int, str, str]] = []
    for page, title, kind in sorted(hits):
        key = (page, title)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((page, title, kind))

    def parent_topic(title: str) -> str | None:
        for t in _TOPIC_HEADS:
            if t in title:
                return t
        return None

    def page_topic(idx: int) -> str | None:
        top = [ln.strip() for ln in doc[idx].get_text().splitlines()[:5] if ln.strip()]
        for ln in top:
            if ln in _TOPIC_HEADS:
                return ln
        return None

    ranges: list[tuple[str, int, int]] = []
    for idx, (page, title, kind) in enumerate(deduped):
        if kind != "start":
            continue
        end = doc.page_count - 1
        for p2, _t2, k2 in deduped[idx + 1:]:
            if k2 in ("start", "stop"):
                end = p2 - 1
                break
        # further constrain by running-head: as soon as the topic head
        # changes, we've left the MCQ answers region.
        parent = parent_topic(title)
        if parent is not None:
            for p_check in range(page + 1, end + 1):
                t_here = page_topic(p_check)
                if t_here is not None and t_here != parent:
                    end = p_check - 1
                    break
        ranges.append((title, page, end))
    return ranges


def _find_ranges_466(doc) -> list[tuple[str, int, int]]:
    """Return (subsection_title, start_page, end_page) for the 466 PDF."""
    root_start: int | None = None
    stop_page = doc.page_count - 1
    subsections: list[tuple[int, str]] = []

    for i in range(doc.page_count):
        top_lines = [ln.strip() for ln in doc[i].get_text().splitlines()[:15] if ln.strip()]
        top_joined = "\n".join(top_lines)
        if root_start is None and _ANSWERS_ROOT_466 in top_joined:
            root_start = i
            continue
        if root_start is not None and _CASE_STUDIES_HEADER_466 in top_joined:
            stop_page = i - 1
            break
        if root_start is not None:
            for ln in top_lines:
                if _SUBSECTION_LABEL_RE.match(ln):
                    subsections.append((i, ln))

    if root_start is None:
        return []

    ranges: list[tuple[str, int, int]] = []
    for idx, (page, title) in enumerate(subsections):
        end = (
            subsections[idx + 1][0] - 1
            if idx + 1 < len(subsections)
            else stop_page
        )
        ranges.append((title, page, end))
    return ranges


# ---------------------------------------------------------------------------
# MCQ extraction from a page range
# ---------------------------------------------------------------------------

def _extract_from_range(
    doc, title: str, start: int, end: int, source: str
) -> list[MCQ]:
    # collect all lines across the range
    lines: list[tuple[str, int, int]] = []
    for i in range(start, end + 1):
        lines.extend(_page_lines(doc[i]))
        # insert a blank separator so per-page boundaries are visible
        lines.append(("", 0, 0))

    # drop noise lines
    lines = [t for t in lines if not _is_noise(t[0])]

    out: list[MCQ] = []
    current: dict | None = None
    current_letter: str | None = None
    emit_seq = 0  # monotonic counter for qid uniqueness — the manual
                  # re-uses question numbers across case-study sub-blocks.

    def flush() -> None:
        nonlocal current, emit_seq
        if current is None or len(current["options"]) != 4:
            current = None
            return
        best_letter: str | None = None
        best_frac = 0.0
        for letter, (text, bold, total) in current["options"].items():
            if total == 0:
                continue
            frac = bold / total
            if frac > best_frac:
                best_frac = frac
                best_letter = letter
        if best_letter is None or best_frac < MIN_BOLD_FRAC:
            current = None
            return
        emit_seq += 1
        qid = (
            f"{source}-{title[:40].replace(' ', '_').replace(',', '')}"
            f"-n{emit_seq:04d}"
        )
        out.append(MCQ(
            qid=qid,
            source=source,
            stem=current["stem"].strip(),
            options={k.upper(): v[0].strip() for k, v in current["options"].items()},
            correct=best_letter.upper(),
            extras={
                "section": title,
                "local_qnum": current["qnum"],
            },
        ))
        current = None

    for line, bold_chars, total in lines:
        s = line.strip()
        qm = _QSTART_RE.match(s)
        if qm:
            qnum = int(qm.group(1))
            if 1 <= qnum <= 200:
                flush()
                current = {
                    "qnum": qnum,
                    "stem": qm.group(2),
                    "options": {},
                }
                current_letter = None
                continue

        if current is None:
            continue

        om = _OPT_RE.match(s)
        if om:
            current_letter = om.group(1)
            current["options"][current_letter] = [om.group(2), bold_chars, total]
            continue

        if not s:
            continue

        if current_letter is None:
            current["stem"] = (current["stem"] + " " + s).strip()
        else:
            opt = current["options"][current_letter]
            opt[0] = (opt[0] + " " + s).strip()
            opt[1] += bold_chars
            opt[2] += total

    flush()
    return out


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------

def parse_file(pdf_path: Path, source: str) -> list[MCQ]:
    doc = fitz.open(pdf_path)
    try:
        if "901740471" in pdf_path.name:
            ranges = _find_ranges_471(doc)
        elif "901740466" in pdf_path.name:
            ranges = _find_ranges_466(doc)
        else:
            return []
        all_qs: list[MCQ] = []
        for title, start, end in ranges:
            all_qs.extend(_extract_from_range(doc, title, start, end, source))
        return all_qs
    finally:
        doc.close()


def parse_all(raw_questions_dir: Path) -> list[MCQ]:
    """Parse both manual PDFs. Return deduped MCQs."""
    files = [
        (raw_questions_dir / "901740466-1-RIBO-Manual-249-387 (1).pdf", "manual_466"),
        (raw_questions_dir / "901740471-RIBO-Manual-Questions (1).pdf", "manual_471"),
    ]
    out: list[MCQ] = []
    for path, source in files:
        if path.exists():
            out.extend(parse_file(path, source))
    return out


def answer_distribution(qs: list[MCQ]) -> Counter:
    return Counter(q.correct for q in qs)

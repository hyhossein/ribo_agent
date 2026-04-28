"""Chunk study documents into retrieval-friendly pieces.

Chunks are semantic units (section / sub-section / article), not fixed
token windows. Reason: exam stems cite section numbers verbatim ("under
s. 14 of Regulation 991..."), so retrieval works much better when one
chunk maps to one statutory unit and the unit number is in the metadata.

Size policy (MIN_CHARS <= len(chunk) <= MAX_CHARS):
  - Tiny sibling sections merge so retrieval doesn't drown in 50-char
    stubs.
  - Oversize sections split on paragraph, then sentence, then char,
    keeping a small overlap so a split mid-provision is still searchable.

Each chunk carries:
  chunk_id   stable id
  source     source document label (e.g. 'RIB_Act_1990')
  citation   human-readable, e.g. 'RIB Act s. 14'
  section    section number as string, e.g. '14', '14(2)', '1.1'
  title      section title when known, else None
  text       the chunk body
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, asdict, field
from pathlib import Path

from .ingest import doc_to_text, pdf_to_text


MAX_CHARS = 2500
MIN_CHARS = 250
OVERLAP = 150


@dataclass
class Chunk:
    chunk_id: str
    source: str
    citation: str
    section: str | None
    title: str | None
    text: str
    page_number: int | None = None
    page_range: tuple[int, int] | None = None   # (start_page, end_page) 1-indexed
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# per-format splitters
# ---------------------------------------------------------------------------

_REG_SECTION_RE = re.compile(r"^(\d+(?:\.\d+)?)\.\s+(.*)")
_BYLAW_SECTION_RE = re.compile(r"^(\d+\.\d+)\s+(.*)")
_BYLAW_ARTICLE_RE = re.compile(r"^ARTICLE\s+(\d+)(?:\s+(.*))?$")
_OAP_SECTION_RE = re.compile(r"^Section\s+(\d+)\s*[-–]\s*(.*)$")
_ACT_SECTION_HEAD_RE = re.compile(r"^(\d+(?:\.\d+)?)\s+([A-Z(].*)")


def _split_regulation(text: str, source: str, citation_prefix: str) -> list[Chunk]:
    """Ontario regulation .txt files from LibreOffice."""
    lines = text.splitlines()
    chunks: list[Chunk] = []
    current_num: str | None = None
    current_title: str | None = None
    current_body: list[str] = []

    def flush() -> None:
        nonlocal current_num, current_title, current_body
        if current_num is None:
            return
        body = "\n".join(current_body).strip()
        if not body:
            return
        chunks.append(Chunk(
            chunk_id=f"{source}-s{current_num}",
            source=source,
            citation=f"{citation_prefix} s. {current_num}",
            section=current_num,
            title=current_title,
            text=body,
        ))
        current_num = None
        current_title = None
        current_body = []

    for raw in lines:
        s = raw.strip()
        if not s or s == "Back to top":
            current_body.append(raw)
            continue
        m = _REG_SECTION_RE.match(s)
        if m and len(m.group(1).split(".")[0]) <= 3:
            flush()
            current_num = m.group(1)
            rest = m.group(2).strip()
            # title is short and doesn't contain sentence markers
            if len(rest) < 60 and rest.endswith(":"):
                current_title = rest.rstrip(":")
                current_body = []
            elif len(rest) < 40 and not any(
                w in rest.lower() for w in [" the ", " shall ", " means "]
            ):
                current_title = rest
                current_body = []
            else:
                current_title = None
                current_body = [rest]
            continue
        current_body.append(s)
    flush()
    return chunks


def _split_act(text: str, source: str) -> list[Chunk]:
    """Parse the RIB Act — LibreOffice puts each section title on its own
    line just before the numbered body line."""
    lines = text.splitlines()

    # skip TOC by finding the first "1 In this Act," line past line 30
    body_start: int | None = None
    for i, ln in enumerate(lines):
        if re.match(r"^1\s+In\s+this\s+Act,", ln.strip()) and i > 30:
            body_start = i
            break
    if body_start is None:
        return []

    chunks: list[Chunk] = []
    current_num: str | None = None
    current_title: str | None = None
    current_body: list[str] = []
    prev_nonblank = ""

    def flush() -> None:
        nonlocal current_num, current_title, current_body
        if current_num is None:
            return
        body = "\n".join(current_body).strip()
        if not body:
            return
        chunks.append(Chunk(
            chunk_id=f"{source}-s{current_num}",
            source=source,
            citation=f"RIB Act s. {current_num}",
            section=current_num,
            title=current_title,
            text=body,
        ))
        current_num = None
        current_title = None
        current_body = []

    def looks_like_title(s: str) -> bool:
        if not s or len(s) > 80 or s.endswith((".", ",")):
            return False
        if s[0].isdigit() or not s[0].isupper():
            return False
        if re.search(r"\b(the|and|shall|means|that)\b", s):
            return False
        return True

    for ln in lines[body_start:]:
        s = ln.strip()
        if not s:
            continue
        m = _ACT_SECTION_HEAD_RE.match(s)
        if m:
            num = m.group(1)
            if 1 <= int(num.split(".")[0]) <= 100:
                flush()
                current_num = num
                current_title = prev_nonblank if looks_like_title(prev_nonblank) else None
                current_body = [m.group(2).strip()]
                prev_nonblank = s
                continue
        current_body.append(s)
        prev_nonblank = s
    flush()
    return chunks


def _split_bylaw(text: str, source: str, label: str) -> list[Chunk]:
    """RIBO By-Laws — Article headings + x.y sub-sections. Strip the TOC
    by jumping to the second occurrence of 'ARTICLE 1'."""
    art_starts = [m.start() for m in re.finditer(r"\n\s*ARTICLE\s+1\b", text)]
    body = text[art_starts[1]:] if len(art_starts) >= 2 else text

    chunks: list[Chunk] = []
    current_section: str | None = None
    current_title: str | None = None
    current_article = ""
    current_body: list[str] = []

    def flush() -> None:
        nonlocal current_section, current_title, current_body
        if current_section is None:
            return
        text_ = "\n".join(current_body).strip()
        if not text_:
            return
        citation = f"{label} s. {current_section}"
        if current_article:
            citation += f" ({current_article})"
        chunks.append(Chunk(
            chunk_id=f"{source}-s{current_section}",
            source=source,
            citation=citation,
            section=current_section,
            title=current_title,
            text=text_,
            extras={"article": current_article} if current_article else {},
        ))
        current_section = None
        current_title = None
        current_body = []

    lines_list = body.splitlines()
    i = 0
    while i < len(lines_list):
        ln = lines_list[i]
        s = ln.strip()
        if not s:
            current_body.append(ln)
            i += 1
            continue
        am = _BYLAW_ARTICLE_RE.match(s)
        if am:
            flush()
            art_title = (am.group(2) or "").strip()
            # Title sometimes lives on the next non-blank line (all-caps).
            if not art_title:
                for j in range(i + 1, min(i + 4, len(lines_list))):
                    nxt = lines_list[j].strip()
                    if nxt and nxt.isupper() and len(nxt) < 60:
                        art_title = nxt.title()
                        i = j
                        break
            current_article = f"Article {am.group(1)} - {art_title}".rstrip(" -")
            i += 1
            continue
        m = _BYLAW_SECTION_RE.match(s)
        if m:
            flush()
            current_section = m.group(1)
            rest = m.group(2).strip()
            if len(rest) < 70 and rest and rest[0].isupper():
                current_title = rest
                current_body = []
            else:
                current_title = None
                current_body = [rest]
            i += 1
            continue
        current_body.append(s)
        i += 1
    flush()
    return chunks


def _split_oap(text: str, source: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    current_section: str | None = None
    current_title: str | None = None
    current_body: list[str] = []

    def flush() -> None:
        nonlocal current_section, current_title, current_body
        if current_section is None:
            return
        text_ = "\n".join(current_body).strip()
        if not text_:
            return
        citation = f"OAP 1 Section {current_section}"
        if current_title:
            citation += f" — {current_title}"
        chunks.append(Chunk(
            chunk_id=f"{source}-s{current_section}",
            source=source,
            citation=citation,
            section=current_section,
            title=current_title,
            text=text_,
        ))
        current_section = None
        current_title = None
        current_body = []

    for ln in text.splitlines():
        s = ln.strip()
        m = _OAP_SECTION_RE.match(s)
        if m:
            flush()
            current_section = m.group(1)
            current_title = m.group(2).strip()
            current_body = []
            continue
        current_body.append(s)
    flush()
    return chunks


# ---------------------------------------------------------------------------
# size normalisation
# ---------------------------------------------------------------------------

def _normalise(chunks: list[Chunk]) -> list[Chunk]:
    # Merge tiny consecutive chunks from the same source.
    merged: list[Chunk] = []
    buf: Chunk | None = None
    for c in chunks:
        if buf is None:
            buf = c
            continue
        if len(buf.text) < MIN_CHARS and c.source == buf.source:
            combined = buf.text + "\n\n"
            if c.title:
                combined += c.title + "\n"
            combined += c.text
            buf = Chunk(
                chunk_id=buf.chunk_id + f"+s{c.section}",
                source=buf.source,
                citation=f"{buf.citation} / {c.citation}",
                section=buf.section,
                title=buf.title or c.title,
                text=combined,
                extras=buf.extras,
            )
            continue
        merged.append(buf)
        buf = c
    if buf is not None:
        merged.append(buf)

    # Split oversize chunks on paragraph -> sentence -> char.
    final: list[Chunk] = []
    for c in merged:
        if len(c.text) <= MAX_CHARS:
            final.append(c)
            continue
        paras = re.split(r"\n\s*\n", c.text)
        # Hard-cap any paragraph that exceeds MAX_CHARS.
        bounded: list[str] = []
        for p in paras:
            if len(p) <= MAX_CHARS:
                bounded.append(p)
                continue
            sents = re.split(r"(?<=[.;])\s+", p)
            cur = ""
            for st in sents:
                if len(cur) + len(st) + 1 > MAX_CHARS and cur:
                    bounded.append(cur)
                    cur = st
                else:
                    cur = (cur + " " + st).strip() if cur else st
            if cur:
                while len(cur) > MAX_CHARS:
                    bounded.append(cur[:MAX_CHARS])
                    cur = cur[MAX_CHARS - OVERLAP:]
                bounded.append(cur)

        # Re-pack bounded pieces into MAX_CHARS buckets with small overlap.
        cur_parts: list[str] = []
        cur_len = 0
        sub_idx = 0
        for p in bounded:
            if cur_len + len(p) > MAX_CHARS and cur_parts:
                sub_idx += 1
                final.append(Chunk(
                    chunk_id=f"{c.chunk_id}-p{sub_idx}",
                    source=c.source,
                    citation=c.citation,
                    section=c.section,
                    title=c.title,
                    text="\n\n".join(cur_parts),
                    extras=c.extras,
                ))
                last = cur_parts[-1]
                cur_parts = (
                    [last[-OVERLAP:], p] if len(last) > OVERLAP else [last, p]
                )
                cur_len = sum(len(x) for x in cur_parts)
            else:
                cur_parts.append(p)
                cur_len += len(p) + 2
        if cur_parts:
            sub_idx += 1
            final.append(Chunk(
                chunk_id=f"{c.chunk_id}-p{sub_idx}",
                source=c.source,
                citation=c.citation,
                section=c.section,
                title=c.title,
                text="\n\n".join(cur_parts),
                extras=c.extras,
            ))
    # Guarantee chunk_id uniqueness. Collisions can happen after merging
    # (e.g. `s4+s5` and `s5+s6` both normalise to the same key in rare
    # cases). Append a suffix only where needed so stable cases keep their
    # clean IDs.
    seen: dict[str, int] = {}
    for c in final:
        if c.chunk_id in seen:
            seen[c.chunk_id] += 1
            c.chunk_id = f"{c.chunk_id}#{seen[c.chunk_id]}"
        else:
            seen[c.chunk_id] = 0
    return final


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------

def chunk_corpus(raw_study_dir: Path, cache_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []

    # Regulations (.doc)
    reg_specs = [
        ("Ontario Regulation 989.doc", "O.Reg. 308/98 (Reg. 989)", "Ontario_Regulation_989"),
        ("Ontario Regulation 990.doc", "R.R.O. 1990 Reg. 990",     "Ontario_Regulation_990"),
        ("Ontario Regulation 991.doc", "R.R.O. 1990 Reg. 991",     "Ontario_Regulation_991"),
    ]
    for fname, citation_prefix, label in reg_specs:
        p = raw_study_dir / fname
        if not p.exists():
            continue
        text = doc_to_text(p, cache_dir=cache_dir)
        chunks.extend(_split_regulation(text, source=label, citation_prefix=citation_prefix))

    # RIB Act (.doc)
    act_path = raw_study_dir / "Registered Insurance Brokers Act 1990 (1).doc"
    if act_path.exists():
        text = doc_to_text(act_path, cache_dir=cache_dir)
        chunks.extend(_split_act(text, source="RIB_Act_1990"))

    # By-Laws (.pdf)
    for fname, label in [
        ("RIBO By-Law No. 1 March 2024.pdf", "RIBO By-Law 1"),
        ("RIBO By-Law No 2 March 2024.pdf",  "RIBO By-Law 2"),
        ("RIBO By-Law No 3 March 2024.pdf",  "RIBO By-Law 3"),
    ]:
        p = raw_study_dir / fname
        if not p.exists():
            continue
        text = pdf_to_text(p, cache_dir=cache_dir)
        chunks.extend(_split_bylaw(text, source=label.replace(" ", "_"), label=label))

    # OAP 2025 (.pdf)
    oap_path = raw_study_dir / "Ontario Automobile Policy 2025 (1).pdf"
    if oap_path.exists():
        text = pdf_to_text(oap_path, cache_dir=cache_dir)
        chunks.extend(_split_oap(text, source="OAP_2025"))

    return _normalise(chunks)


def summarise(chunks: list[Chunk]) -> dict:
    lengths = [len(c.text) for c in chunks]
    return {
        "total_chunks": len(chunks),
        "by_source": dict(__import__("collections").Counter(c.source for c in chunks)),
        "chunk_len_min": min(lengths) if lengths else 0,
        "chunk_len_p50": int(statistics.median(lengths)) if lengths else 0,
        "chunk_len_p90": sorted(lengths)[int(len(lengths) * 0.9)] if lengths else 0,
        "chunk_len_max": max(lengths) if lengths else 0,
        "chunk_len_mean": int(statistics.mean(lengths)) if lengths else 0,
    }

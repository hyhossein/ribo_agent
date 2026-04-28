"""Deduplicate MCQs by content fingerprint.

A fingerprint is sha256(normalised stem + first option text), truncated
to 16 hex chars. Catches "same question copy-pasted across files" while
being robust to whitespace and capitalisation drift.

Used two ways:
  1. Drop duplicates within a single parsed set.
  2. Subtract the eval-set fingerprints from the manual set so we never
     accidentally train / few-shot on questions we're grading on.
"""
from __future__ import annotations

import hashlib
import re

from .schema import MCQ


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return s


def fingerprint(q: MCQ) -> str:
    first_option_text = next(iter(q.options.values()))
    key = _norm(q.stem) + "||" + _norm(first_option_text)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def dedup(qs: list[MCQ]) -> list[MCQ]:
    seen: set[str] = set()
    out: list[MCQ] = []
    for q in qs:
        fp = fingerprint(q)
        if fp in seen:
            continue
        seen.add(fp)
        out.append(q)
    return out


def subtract(qs: list[MCQ], against: list[MCQ]) -> tuple[list[MCQ], list[MCQ]]:
    """Return (kept, removed) from qs, removing any whose fingerprint is
    also present in `against`."""
    block: set[str] = {fingerprint(q) for q in against}
    kept: list[MCQ] = []
    removed: list[MCQ] = []
    for q in qs:
        (removed if fingerprint(q) in block else kept).append(q)
    return kept, removed

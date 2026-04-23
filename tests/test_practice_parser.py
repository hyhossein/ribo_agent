"""Tests for the Practice Exam parser (695993459-Practise-RIBO-Exam.pdf).

These tests exist specifically to catch the two traps documented in
docs/EDA.md §§4-5:

  Trap A — answer-key column offsets drift between pages, so we can't
           hard-code a column->letter map. Any naive fixed-map parser will
           get roughly half the answers wrong.
  Trap B — 15 question numbers sit right after a form-feed character
           instead of a newline; naive ^-anchored regex misses all of them.

Regression of either of these trips concrete spot-check assertions below.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ribo_agent.parsers import practice
from ribo_agent.parsers.schema import MCQ


@pytest.fixture(scope="session")
def practice_pdf(raw_questions_dir: Path) -> Path:
    p = raw_questions_dir / "695993459-Practise-RIBO-Exam.pdf"
    if not p.exists():
        pytest.skip(f"practice pdf missing: {p}")
    return p


@pytest.fixture(scope="module")
def parsed(practice_pdf: Path) -> list[MCQ]:
    return practice.parse(practice_pdf)


# -- structural -----------------------------------------------------------

def test_expected_question_count(parsed: list[MCQ]) -> None:
    assert len(parsed) == 90


def test_every_mcq_has_four_options_and_a_correct_letter(parsed: list[MCQ]) -> None:
    for q in parsed:
        assert set(q.options) == {"A", "B", "C", "D"}, q.qid
        assert q.correct in {"A", "B", "C", "D"}, q.qid


def test_qids_are_sequential(parsed: list[MCQ]) -> None:
    assert [q.qid for q in parsed] == [
        f"practice_exam-q{n:03d}" for n in range(1, 91)
    ]


def test_answer_distribution_is_balanced(parsed: list[MCQ]) -> None:
    # If column clustering mis-assigned letters the distribution will
    # collapse (e.g. 0 of one letter, 40 of another). A legitimate exam is
    # well-mixed; require each letter appears at least 10 times.
    from collections import Counter
    counts = Counter(q.correct for q in parsed)
    for letter in "ABCD":
        assert counts[letter] >= 10, f"{letter}: {counts[letter]} (suspicious)"


# -- Trap B regression ---------------------------------------------------

# These specific numbers are the ones identified in docs/EDA.md §5 that sit
# immediately after a form-feed. If `\f -> \n` replacement is removed from
# the parser, all of these disappear.

@pytest.mark.parametrize("qnum", [6, 12, 19, 25, 31, 37, 43, 49, 54, 59, 64, 69, 75, 81, 86])
def test_form_feed_question_is_captured(parsed: list[MCQ], qnum: int) -> None:
    qids = {q.qid for q in parsed}
    assert f"practice_exam-q{qnum:03d}" in qids, (
        f"Q{qnum} missing — likely a form-feed handling regression"
    )


# -- Trap A regression (column-clustering per page) ----------------------

# Q50-Q54 live on the second answer-key page where offsets are [10,14,18,22]
# — tight spacing. Q1-Q49 live on the first page with offsets [10,17,32,45].
# A fixed-column mapping trained on page 1 would mis-label everything in
# Q50+. Spot-check a few answers that straddle this boundary.

def test_q49_on_page_one(parsed: list[MCQ]) -> None:
    # Grid shows X in the B column on page 1 for Q49.
    assert parsed[48].correct == "B"


def test_q50_on_page_two(parsed: list[MCQ]) -> None:
    # Grid shows X at offset 22 on page 2 — the rightmost cluster = D.
    assert parsed[49].correct == "D"


def test_q90_final_question(parsed: list[MCQ]) -> None:
    # Q90 ice-fishing / partial-loss-less-deductible = C.
    q = parsed[-1]
    assert "ice fishing" in q.stem.lower()
    assert q.correct == "C"


# -- content sanity -----------------------------------------------------

def test_q1_is_about_status_changes(parsed: list[MCQ]) -> None:
    q = parsed[0]
    assert "registration" in q.stem.lower() or "insurance broker" in q.stem.lower()
    assert q.correct == "D"


def test_no_option_is_empty(parsed: list[MCQ]) -> None:
    for q in parsed:
        for letter, text in q.options.items():
            assert text.strip(), f"{q.qid} option {letter} is empty"

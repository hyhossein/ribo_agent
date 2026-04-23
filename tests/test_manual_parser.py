"""Tests for the manual-PDF parser and the dedup/leakage logic."""
from __future__ import annotations

from pathlib import Path

import pytest

from ribo_agent.parsers import manual, practice, sample
from ribo_agent.parsers.dedup import dedup, fingerprint, subtract
from ribo_agent.parsers.schema import MCQ


@pytest.fixture(scope="module")
def manual_all(raw_questions_dir: Path) -> list[MCQ]:
    return manual.parse_all(raw_questions_dir)


# -- structural -------------------------------------------------------------

def test_manual_parser_extracts_hundreds_of_mcqs(manual_all: list[MCQ]) -> None:
    # The two PDFs together carry ~300-600 MCQs depending on dedup. A
    # regression where font-flag detection breaks drops this to ~0.
    assert len(manual_all) >= 300, (
        f"only {len(manual_all)} MCQs — bold-answer detection may be broken"
    )


def test_every_manual_mcq_has_four_options_and_valid_correct(manual_all: list[MCQ]) -> None:
    for q in manual_all:
        assert set(q.options) == {"A", "B", "C", "D"}, q.qid
        assert q.correct in {"A", "B", "C", "D"}, q.qid


def test_manual_answer_distribution_is_plausible(manual_all: list[MCQ]) -> None:
    # Real exams have reasonably mixed answer distributions. If bold
    # detection flips to choose, say, option D every time we'd see a
    # degenerate distribution.
    dist = manual.answer_distribution(manual_all)
    for letter in "ABCD":
        assert dist[letter] >= 20, (
            f"{letter}: {dist[letter]} — suspiciously low (bold detection drift?)"
        )


def test_manual_qids_are_unique(manual_all: list[MCQ]) -> None:
    assert len({q.qid for q in manual_all}) == len(manual_all)


def test_manual_sections_tagged(manual_all: list[MCQ]) -> None:
    # Each MCQ should carry which manual sub-section it came from in
    # `extras`, so retrieval can filter by topic later.
    assert all(q.extras.get("section") for q in manual_all)


# -- dedup / leakage --------------------------------------------------------

def _fake(qid: str, stem: str, first: str, correct: str = "A") -> MCQ:
    return MCQ(
        qid=qid,
        source="test",
        stem=stem,
        options={"A": first, "B": "x", "C": "y", "D": "z"},
        correct=correct,
    )


def test_fingerprint_is_stable_across_whitespace_and_case() -> None:
    a = _fake("a", "What is indemnity?", "Payment of a loss")
    b = _fake("b", "what   IS    indemnity?", "PAYMENT OF A LOSS")
    assert fingerprint(a) == fingerprint(b)


def test_fingerprint_differs_when_content_differs() -> None:
    a = _fake("a", "What is indemnity?", "Payment of a loss")
    b = _fake("b", "What is subrogation?", "Payment of a loss")
    assert fingerprint(a) != fingerprint(b)


def test_dedup_drops_duplicates_keeps_first() -> None:
    a1 = _fake("a1", "Stem one", "First")
    a2 = _fake("a2", "Stem  one", "first")  # same fingerprint
    b = _fake("b", "Stem two", "First")
    out = dedup([a1, a2, b])
    assert [q.qid for q in out] == ["a1", "b"]


def test_subtract_removes_fingerprints_present_in_other_set() -> None:
    shared_stem = "Does the policy cover X?"
    shared_opt = "Yes."
    a = _fake("a", shared_stem, shared_opt)
    b = _fake("b", shared_stem, shared_opt)
    c = _fake("c", "Entirely different", "Nope")
    kept, removed = subtract([a, c], against=[b])
    assert [q.qid for q in kept] == ["c"]
    assert [q.qid for q in removed] == ["a"]


def test_eval_and_train_pools_have_no_overlap(
    sample_pdf: Path, raw_questions_dir: Path
) -> None:
    # End-to-end check: the produced train set must share zero fingerprints
    # with the eval set, otherwise agents that use train as a few-shot
    # pool would be peeking at the answer key.
    practice_pdf = raw_questions_dir / "695993459-Practise-RIBO-Exam.pdf"
    eval_qs = sample.parse(sample_pdf) + practice.parse(practice_pdf)
    manual_qs = manual.parse_all(raw_questions_dir)
    train_kept, train_removed = subtract(manual_qs, eval_qs)
    # Assert no overlap remains.
    eval_fps = {fingerprint(q) for q in eval_qs}
    train_fps = {fingerprint(q) for q in train_kept}
    assert eval_fps.isdisjoint(train_fps)

"""Tests for the Sample-Questions parser.

These tests assert on **specific known answers** from the PDF so a parser
regression flips a concrete test, not just an item count.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ribo_agent.parsers import sample
from ribo_agent.parsers.schema import MCQ


@pytest.fixture(scope="module")
def parsed(sample_pdf: Path) -> list[MCQ]:
    return sample.parse(sample_pdf)


# -- structural -----------------------------------------------------------

def test_expected_question_count(parsed: list[MCQ]) -> None:
    # The Sample Questions PDF has 79 numbered questions (confirmed via EDA
    # — see results/day1_eda.log §3).
    assert len(parsed) == 79


def test_every_mcq_has_four_options_and_a_correct_letter(parsed: list[MCQ]) -> None:
    for q in parsed:
        assert set(q.options) == {"A", "B", "C", "D"}, q.qid
        assert q.correct in {"A", "B", "C", "D"}, q.qid


def test_every_mcq_has_metadata(parsed: list[MCQ]) -> None:
    for q in parsed:
        assert q.content_domain, q.qid
        assert q.competency, q.qid
        assert q.cognitive_level, q.qid


def test_qids_are_unique(parsed: list[MCQ]) -> None:
    assert len({q.qid for q in parsed}) == len(parsed)


def test_qids_are_sequential_and_zero_padded(parsed: list[MCQ]) -> None:
    expected = [f"sample_level1-q{n:03d}" for n in range(1, 80)]
    assert [q.qid for q in parsed] == expected


# -- spot checks for specific known answers ------------------------------
# If any of these break, the parser mis-aligned stems, options, or answers.

def test_q1_dishonesty_of_employee_bond(parsed: list[MCQ]) -> None:
    # Q1: "Which form of coverage could protect your client against a loss
    #      due to the dishonesty of an employee?" -> 3-D bond = A
    q = parsed[0]
    assert "dishonesty of an employee" in q.stem.lower()
    assert "3-D bond" in q.options["A"]
    assert q.correct == "A"
    assert q.content_domain == "Commercial Lines"


def test_q3_coinsurance_clause(parsed: list[MCQ]) -> None:
    # Q3: Co-insurance clause may DECREASE the amount paid -> C
    q = parsed[2]
    assert "co-insurance" in q.stem.lower()
    assert q.correct == "C"


def test_q4_business_interruption(parsed: list[MCQ]) -> None:
    # Q4: Business Interruption insurance -> B
    q = parsed[3]
    assert "loss of\nincome" in q.stem.lower() or "loss of income" in q.stem.lower()
    assert "Business Interruption" in q.options[q.correct]
    assert q.correct == "B"


# -- invariants we'll rely on downstream ---------------------------------

def test_no_option_is_empty(parsed: list[MCQ]) -> None:
    for q in parsed:
        for letter, text in q.options.items():
            assert text.strip(), f"{q.qid} option {letter} is empty"


def test_stem_is_non_trivial(parsed: list[MCQ]) -> None:
    for q in parsed:
        assert len(q.stem) >= 20, q.qid


def test_metadata_values_are_from_a_small_closed_set(parsed: list[MCQ]) -> None:
    # Confirms we're not accidentally picking up body text as metadata.
    cognitive_levels = {q.cognitive_level for q in parsed}
    assert cognitive_levels <= {"Knowledge", "Comprehension", "Application"}

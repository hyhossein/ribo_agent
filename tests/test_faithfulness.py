"""Tests for faithfulness validator."""
from ribo_agent.eval.faithfulness import (
    check_faithfulness,
    FaithfulnessVerdict,
    _extract_citation_refs,
    _extract_quoted_phrases,
)


def test_no_citations_gives_no_context():
    result = check_faithfulness("The answer is B.", citations=[])
    assert result.verdict == FaithfulnessVerdict.NO_CONTEXT
    assert result.grounding_score == 0.0


def test_grounded_response():
    citations = [
        {
            "source": "RIB_Act_1990",
            "citation": "RIB Act s. 14",
            "snippet": (
                "Every registered insurance broker shall act in good faith "
                "and deal fairly with the insured in all transactions."
            ),
            "score": 0.92,
        }
    ]
    response = (
        "According to RIB Act s. 14, every registered insurance broker "
        "shall act in good faith and deal fairly with the insured. "
        "Therefore the answer is B."
    )
    result = check_faithfulness(response, citations)
    assert result.verdict in (FaithfulnessVerdict.GROUNDED, FaithfulnessVerdict.PARTIAL)
    assert result.grounding_score > 0.0


def test_ungrounded_response():
    citations = [
        {
            "source": "OAP_2025",
            "citation": "OAP 1 Section 3",
            "snippet": "This section covers liability coverage for automobiles.",
            "score": 0.5,
        }
    ]
    response = (
        "The quantum entanglement principle clearly states that insurance "
        "policies must follow the laws of thermodynamics. The molecular "
        "structure of the policy determines the coverage amount."
    )
    result = check_faithfulness(response, citations)
    assert result.grounding_score < 0.4


def test_extract_citation_refs():
    text = "Under s. 14 of the RIB Act, and also Section 7 of Regulation 991..."
    refs = _extract_citation_refs(text)
    assert any("14" in r for r in refs)
    assert any("991" in r for r in refs)


def test_extract_quoted_phrases():
    text = 'The statute says "every broker shall act in good faith" which means...'
    quotes = _extract_quoted_phrases(text)
    assert len(quotes) == 1
    assert "good faith" in quotes[0]

"""Tests for the zero-shot agent and eval metrics.

Uses a MockLLM so we don't need Ollama in CI.
"""
from __future__ import annotations

import pytest

from ribo_agent.agents import ZeroShotAgent
from ribo_agent.agents.base import Prediction
from ribo_agent.agents.zeroshot import extract_answer
from ribo_agent.eval.metrics import compute_metrics, format_report
from ribo_agent.llm.base import LLMResponse
from ribo_agent.parsers.schema import MCQ


class MockLLM:
    """Returns a scripted response per call."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._i = 0

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        r = self._responses[self._i]
        self._i += 1
        return LLMResponse(
            text=r,
            prompt_tokens=100,
            completion_tokens=20,
            latency_ms=50.0,
            model="mock",
            backend="mock",
        )

    def health(self) -> bool:
        return True


def _fake_mcq(qid: str, correct: str = "A", domain: str | None = None) -> MCQ:
    return MCQ(
        qid=qid,
        source="test",
        stem="What is indemnity?",
        options={"A": "a", "B": "b", "C": "c", "D": "d"},
        correct=correct,
        content_domain=domain,
    )


# -- extract_answer regression ---------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("<answer>A</answer>", "A"),
        ("Reasoning... <answer>c</answer>", "C"),
        ("The answer is B.", "B"),
        ("After consideration, Answer: D.", "D"),
        ("So the correct choice is B.", "B"),
        ("Looking at this carefully, I choose A.", "A"),
        ("", None),
        ("This is a question about insurance, nothing decisive here", None),
        ("<answer>X</answer>", None),       # invalid letter
        ("A good point. The answer is D", "D"),
    ],
)
def test_extract_answer(text: str, expected: str | None) -> None:
    assert extract_answer(text) == expected


# -- ZeroShotAgent ---------------------------------------------------------

def test_zeroshot_agent_scores_correct_answer() -> None:
    agent = ZeroShotAgent(MockLLM(["<answer>A</answer>"]))
    pred = agent.answer(_fake_mcq("q1", correct="A"))
    assert pred.predicted == "A"
    assert pred.is_correct is True
    assert pred.latency_ms == 50.0


def test_zeroshot_agent_scores_wrong_answer() -> None:
    agent = ZeroShotAgent(MockLLM(["<answer>B</answer>"]))
    pred = agent.answer(_fake_mcq("q1", correct="A"))
    assert pred.predicted == "B"
    assert pred.is_correct is False


def test_zeroshot_agent_handles_refusal() -> None:
    agent = ZeroShotAgent(MockLLM(["I cannot determine an answer."]))
    pred = agent.answer(_fake_mcq("q1", correct="A"))
    assert pred.predicted is None
    assert pred.is_correct is False


# -- metrics ---------------------------------------------------------------

def _p(qid: str, pred: str | None, correct: str) -> Prediction:
    return Prediction(
        qid=qid,
        predicted=pred,
        correct=correct,
        is_correct=(pred == correct),
        latency_ms=10.0,
    )


def test_metrics_accuracy_and_f1() -> None:
    preds = [
        _p("q1", "A", "A"),   # correct
        _p("q2", "B", "B"),   # correct
        _p("q3", "A", "B"),   # wrong
        _p("q4", None, "C"),  # refusal -> wrong
    ]
    m = compute_metrics(preds)
    assert m.n == 4
    assert m.n_answered == 3
    assert m.accuracy == 0.5
    assert m.refusal_rate == 0.25
    # micro-F1 == accuracy for single-label
    assert m.micro_f1 == 0.5
    # macro-F1 over classes A/B/C/D
    assert 0.0 < m.macro_f1 <= 1.0


def test_metrics_confusion_matrix() -> None:
    preds = [_p("q1", "A", "A"), _p("q2", "B", "A"), _p("q3", None, "C")]
    m = compute_metrics(preds)
    assert m.confusion["A"]["A"] == 1
    assert m.confusion["A"]["B"] == 1
    assert m.confusion["C"]["REFUSED"] == 1


def test_metrics_per_domain_breakdown_uses_mcq_metadata() -> None:
    preds = [_p("q1", "A", "A"), _p("q2", "A", "B")]
    mcqs = [
        _fake_mcq("q1", "A", domain="Commercial Lines"),
        _fake_mcq("q2", "B", domain="Commercial Lines"),
    ]
    m = compute_metrics(preds, mcqs=mcqs)
    assert m.per_domain == {
        "Commercial Lines": {"n": 2, "accuracy": 0.5},
    }


def test_format_report_is_markdown() -> None:
    preds = [_p("q1", "A", "A"), _p("q2", "B", "B")]
    out = format_report(compute_metrics(preds), title="Test Run")
    assert "# Test Run" in out
    assert "accuracy" in out
    assert "| A | " in out  # table row

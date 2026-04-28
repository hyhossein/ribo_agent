"""Generate rich per-question markdown reports with citations and faithfulness.

Produces a detailed report for each eval run showing:
  - Per-question: answer, citations, retrieved docs, faithfulness verdict
  - Summary: grounding stats, accuracy by faithfulness bucket
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from ..agents.base import Prediction
from .faithfulness import check_faithfulness, FaithfulnessResult


def _verdict_emoji(verdict: str) -> str:
    return {
        "GROUNDED": "✅",
        "PARTIAL": "⚠️",
        "UNGROUNDED": "❌",
        "NO_CONTEXT": "➖",
    }.get(verdict, "❓")


def format_detailed_report(
    predictions: list[Prediction],
    *,
    title: str = "Evaluation Report",
) -> str:
    """Generate a markdown report with per-question citations and faithfulness."""
    lines: list[str] = [f"# {title}", ""]

    # Run faithfulness on all predictions
    faith_results: list[FaithfulnessResult] = []
    for pred in predictions:
        fr = check_faithfulness(pred.raw_response, pred.citations)
        faith_results.append(fr)

    # Summary stats
    n = len(predictions)
    n_correct = sum(1 for p in predictions if p.is_correct)
    n_grounded = sum(1 for f in faith_results if f.verdict.value == "GROUNDED")
    n_partial = sum(1 for f in faith_results if f.verdict.value == "PARTIAL")
    n_ungrounded = sum(1 for f in faith_results if f.verdict.value == "UNGROUNDED")
    n_no_ctx = sum(1 for f in faith_results if f.verdict.value == "NO_CONTEXT")

    lines += [
        "## Summary", "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Total questions | {n} |",
        f"| Correct | {n_correct} ({n_correct/max(n,1):.1%}) |",
        f"| ✅ Grounded | {n_grounded} ({n_grounded/max(n,1):.1%}) |",
        f"| ⚠️ Partially grounded | {n_partial} ({n_partial/max(n,1):.1%}) |",
        f"| ❌ Ungrounded | {n_ungrounded} ({n_ungrounded/max(n,1):.1%}) |",
        f"| ➖ No context (zero-shot) | {n_no_ctx} ({n_no_ctx/max(n,1):.1%}) |",
        "",
    ]

    # Accuracy by faithfulness bucket
    buckets: dict[str, list[bool]] = {}
    for pred, fr in zip(predictions, faith_results):
        v = fr.verdict.value
        buckets.setdefault(v, []).append(pred.is_correct)

    if any(v != "NO_CONTEXT" for v in buckets):
        lines += [
            "## Accuracy by Faithfulness", "",
            "| Verdict | N | Accuracy |",
            "| --- | --- | --- |",
        ]
        for v in ["GROUNDED", "PARTIAL", "UNGROUNDED", "NO_CONTEXT"]:
            if v in buckets:
                items = buckets[v]
                acc = sum(items) / max(len(items), 1)
                lines.append(
                    f"| {_verdict_emoji(v)} {v} | {len(items)} | {acc:.1%} |"
                )
        lines.append("")

    # Per-question detail
    lines += ["## Per-Question Details", ""]
    for i, (pred, fr) in enumerate(zip(predictions, faith_results), 1):
        status = "✅" if pred.is_correct else "❌"
        lines += [
            f"### Q{i}: {pred.qid}",
            "",
            f"**Answer:** {pred.predicted or 'REFUSED'} "
            f"(correct: {pred.correct}) {status}",
            "",
            f"**Faithfulness:** {_verdict_emoji(fr.verdict.value)} "
            f"{fr.verdict.value} (score: {fr.grounding_score:.2f})",
            "",
        ]

        # Citations
        if pred.citations:
            lines += ["**Retrieved Documents:**", ""]
            for j, cit in enumerate(pred.citations, 1):
                source = cit.get("source", "?")
                citation = cit.get("citation", "?")
                score = cit.get("score", 0)
                section = cit.get("section", "")
                snippet = cit.get("snippet", "")[:200]
                lines += [
                    f"  {j}. **{citation}** (source: `{source}`, "
                    f"section: {section}, relevance: {score:.2f})",
                    f"     > {snippet}...",
                    "",
                ]

        # Faithfulness details
        if fr.details:
            lines.append(f"**Grounding:** {fr.details}")
        if fr.unmatched_claims:
            lines += ["", "**⚠️ Ungrounded claims:**"]
            for claim in fr.unmatched_claims:
                lines.append(f"  - _{claim}_")
        lines += ["", "---", ""]

    # Model reasoning (first 3 questions as examples)
    lines += ["## Example Model Reasoning (first 3)", ""]
    for i, pred in enumerate(predictions[:3], 1):
        lines += [
            f"### Q{i}: {pred.qid}",
            "",
            "```",
            pred.raw_response[:500],
            "```",
            "",
        ]

    return "\n".join(lines)


def write_detailed_report(
    predictions: list[Prediction],
    output_path: Path,
    *,
    title: str = "Evaluation Report",
) -> Path:
    """Write a detailed markdown report to disk."""
    report = format_detailed_report(predictions, title=title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    return output_path

"""Classification metrics for MCQ predictions.

All metrics are pure-Python (no sklearn dep) — keeps the install light
and runs in tens of microseconds on 169 rows.

Reported:
  - accuracy (exact match)
  - per-class precision/recall/F1 over A/B/C/D
  - macro-F1 (unweighted mean of per-class F1)
  - micro-F1 (== accuracy for single-label)
  - per-domain accuracy (only where the MCQ carries content_domain)
  - per-cognitive-level accuracy (only where it carries cognitive_level)
  - refusal rate (predicted is None)
  - confusion matrix
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any

from ..agents.base import Prediction
from ..parsers.schema import MCQ


LABELS = ("A", "B", "C", "D")


@dataclass
class Metrics:
    n: int
    n_answered: int
    accuracy: float
    macro_f1: float
    micro_f1: float
    refusal_rate: float
    per_class: dict[str, dict[str, float]]
    per_domain: dict[str, dict[str, float]]
    per_cognitive: dict[str, dict[str, float]]
    confusion: dict[str, dict[str, int]]
    latency_ms_p50: float | None = None
    latency_ms_p90: float | None = None
    latency_ms_mean: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((len(xs) - 1) * p))))
    return xs[k]


def _f1(precision: float, recall: float) -> float:
    return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)


def compute_metrics(
    preds: list[Prediction], mcqs: list[MCQ] | None = None
) -> Metrics:
    """Compute all metrics. `mcqs` is optional; if provided we include the
    per-domain and per-cognitive-level breakdowns that rely on MCQ
    metadata."""
    n = len(preds)
    by_qid: dict[str, MCQ] = {m.qid: m for m in (mcqs or [])}

    answered = [p for p in preds if p.predicted is not None]
    n_answered = len(answered)
    refusal_rate = 0.0 if n == 0 else (n - n_answered) / n

    # Overall accuracy (refusals count as wrong)
    n_correct = sum(1 for p in preds if p.is_correct)
    accuracy = n_correct / n if n else 0.0

    # Per-class confusion + PRF
    confusion: dict[str, dict[str, int]] = {
        gt: {pred: 0 for pred in (*LABELS, "REFUSED")} for gt in LABELS
    }
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    for p in preds:
        gt = p.correct
        pred = p.predicted if p.predicted in LABELS else "REFUSED"
        if gt in confusion and pred in confusion[gt]:
            confusion[gt][pred] += 1
        if pred == gt:
            tp[gt] += 1
        else:
            fn[gt] += 1
            if pred in LABELS:
                fp[pred] += 1

    per_class: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    for label in LABELS:
        support = sum(1 for p in preds if p.correct == label)
        precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0.0
        f1 = _f1(precision, recall)
        f1s.append(f1)
        per_class[label] = {
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    macro_f1 = round(sum(f1s) / len(f1s), 4)
    micro_f1 = round(accuracy, 4)  # single-label micro-F1 == accuracy

    # Per-domain and per-cognitive-level (only where metadata exists)
    def _breakdown(key: str) -> dict[str, dict[str, float]]:
        buckets: dict[str, list[Prediction]] = defaultdict(list)
        for p in preds:
            mcq = by_qid.get(p.qid)
            val = getattr(mcq, key, None) if mcq else None
            if val:
                buckets[val].append(p)
        out = {}
        for val, items in sorted(buckets.items()):
            correct = sum(1 for p in items if p.is_correct)
            out[val] = {
                "n": len(items),
                "accuracy": round(correct / len(items), 4),
            }
        return out

    per_domain = _breakdown("content_domain")
    per_cognitive = _breakdown("cognitive_level")

    # Latency
    latencies = [p.latency_ms for p in preds if p.latency_ms is not None]
    return Metrics(
        n=n,
        n_answered=n_answered,
        accuracy=round(accuracy, 4),
        macro_f1=macro_f1,
        micro_f1=micro_f1,
        refusal_rate=round(refusal_rate, 4),
        per_class=per_class,
        per_domain=per_domain,
        per_cognitive=per_cognitive,
        confusion=confusion,
        latency_ms_p50=_percentile(latencies, 0.5),
        latency_ms_p90=_percentile(latencies, 0.9),
        latency_ms_mean=(sum(latencies) / len(latencies)) if latencies else None,
    )


def format_report(metrics: Metrics, *, title: str = "") -> str:
    """Markdown-formatted summary suitable for committing to results/."""
    lines: list[str] = []
    if title:
        lines += [f"# {title}", ""]
    lines += [
        "## Overall",
        "",
        f"| metric | value |",
        f"| --- | --- |",
        f"| n | {metrics.n} |",
        f"| accuracy | **{metrics.accuracy:.4f}** |",
        f"| macro-F1 | **{metrics.macro_f1:.4f}** |",
        f"| micro-F1 | {metrics.micro_f1:.4f} |",
        f"| refusal rate | {metrics.refusal_rate:.4f} |",
    ]
    if metrics.latency_ms_mean is not None:
        lines += [
            f"| latency mean (ms) | {metrics.latency_ms_mean:.0f} |",
            f"| latency p50 (ms) | {metrics.latency_ms_p50:.0f} |",
            f"| latency p90 (ms) | {metrics.latency_ms_p90:.0f} |",
        ]
    lines += ["", "## Per class (A/B/C/D)", "",
              "| class | support | precision | recall | F1 |",
              "| --- | --- | --- | --- | --- |"]
    for label, vals in metrics.per_class.items():
        lines.append(
            f"| {label} | {vals['support']} | {vals['precision']:.4f} | "
            f"{vals['recall']:.4f} | {vals['f1']:.4f} |"
        )
    if metrics.per_domain:
        lines += ["", "## Per content domain (sample-set only)", "",
                  "| domain | n | accuracy |",
                  "| --- | --- | --- |"]
        for d, vals in metrics.per_domain.items():
            lines.append(f"| {d} | {vals['n']} | {vals['accuracy']:.4f} |")
    if metrics.per_cognitive:
        lines += ["", "## Per cognitive level", "",
                  "| level | n | accuracy |",
                  "| --- | --- | --- |"]
        for c, vals in metrics.per_cognitive.items():
            lines.append(f"| {c} | {vals['n']} | {vals['accuracy']:.4f} |")
    lines += ["", "## Confusion matrix (rows = truth, cols = predicted)", ""]
    header = " | ".join(["truth\\pred"] + list(LABELS) + ["REFUSED"])
    sep = " | ".join(["---"] * (len(LABELS) + 2))
    lines += [f"| {header} |", f"| {sep} |"]
    for gt in LABELS:
        row = [str(metrics.confusion[gt][c]) for c in (*LABELS, "REFUSED")]
        lines.append("| " + " | ".join([gt, *row]) + " |")
    return "\n".join(lines) + "\n"

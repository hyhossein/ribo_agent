"""Parse every question PDF into data/parsed/*.jsonl.

Produces three raw sets plus two derived splits:

  sample_questions.jsonl    79 MCQs with rich metadata           (eval input)
  practice_exam.jsonl       90 MCQs from the X-grid-keyed PDF    (eval input)
  manual_pool.jsonl         ~1000 MCQs from the manual PDFs      (few-shot pool)

  eval.jsonl                sample + practice = 169 held-out questions
  train.jsonl               manual_pool minus any eval fingerprints

Usage:
  python -m ribo_agent.parsers.run_parse           # all
  python -m ribo_agent.parsers.run_parse sample    # just the sample PDF
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import manual, practice, sample
from .dedup import dedup, subtract

ROOT = Path(__file__).resolve().parents[3]
RAW_Q = ROOT / "data" / "raw" / "questions"
OUT = ROOT / "data" / "parsed"


def _write_jsonl(records: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def run_sample() -> list:
    pdf = RAW_Q / "Sample-Questions-RIBO-Level-1-Exam (1).pdf"
    qs = sample.parse(pdf)
    _write_jsonl(qs, OUT / "sample_questions.jsonl")
    print(f"sample_questions  {len(qs):4d} -> data/parsed/sample_questions.jsonl")
    return qs


def run_practice() -> list:
    pdf = RAW_Q / "695993459-Practise-RIBO-Exam.pdf"
    qs = practice.parse(pdf)
    _write_jsonl(qs, OUT / "practice_exam.jsonl")
    print(f"practice_exam     {len(qs):4d} -> data/parsed/practice_exam.jsonl")
    return qs


def run_manual() -> list:
    qs = manual.parse_all(RAW_Q)
    deduped = dedup(qs)
    _write_jsonl(deduped, OUT / "manual_pool.jsonl")
    print(
        f"manual_pool       {len(deduped):4d} -> data/parsed/manual_pool.jsonl"
        f"  ({len(qs) - len(deduped)} intra-pool dups removed)"
    )
    return deduped


def build_splits(eval_pool: list, train_pool: list) -> None:
    _write_jsonl(eval_pool, OUT / "eval.jsonl")
    train_clean, leaked = subtract(train_pool, eval_pool)
    _write_jsonl(train_clean, OUT / "train.jsonl")
    print(f"eval.jsonl        {len(eval_pool):4d}")
    print(
        f"train.jsonl       {len(train_clean):4d}"
        f"  ({len(leaked)} removed via leakage subtract)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "target",
        choices=["sample", "practice", "manual", "all"],
        default="all",
        nargs="?",
    )
    args = ap.parse_args()

    sample_qs = run_sample() if args.target in ("sample", "all") else []
    practice_qs = run_practice() if args.target in ("practice", "all") else []
    manual_qs = run_manual() if args.target in ("manual", "all") else []

    if args.target == "all":
        build_splits(sample_qs + practice_qs, manual_qs)


if __name__ == "__main__":
    main()

"""Parse all question PDFs into data/parsed/*.jsonl.

Usage: python -m ribo_agent.parsers.run_parse [sample|all]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import sample

ROOT = Path(__file__).resolve().parents[3]
RAW_Q = ROOT / "data" / "raw" / "questions"
OUT = ROOT / "data" / "parsed"


def _write_jsonl(records: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def run_sample() -> int:
    pdf = RAW_Q / "Sample-Questions-RIBO-Level-1-Exam (1).pdf"
    qs = sample.parse(pdf)
    _write_jsonl(qs, OUT / "sample_questions.jsonl")
    print(f"sample_questions: {len(qs)} questions -> {OUT / 'sample_questions.jsonl'}")
    return len(qs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=["sample", "all"], default="all", nargs="?")
    args = ap.parse_args()

    if args.target in ("sample", "all"):
        run_sample()
    # additional targets (practice, manual) land on day 3


if __name__ == "__main__":
    main()

"""Build the knowledge base: ingest + chunk.

Usage: python -m ribo_agent.kb.build_kb
"""
from __future__ import annotations

import json
from pathlib import Path

from .chunker import chunk_corpus, summarise


ROOT = Path(__file__).resolve().parents[3]
RAW_STUDY = ROOT / "data" / "raw" / "study"
CACHE = ROOT / "data" / "interim" / "study_txt"
OUT = ROOT / "data" / "kb" / "chunks.jsonl"
REPORT = ROOT / "data" / "kb" / "summary.json"


def main() -> None:
    chunks = chunk_corpus(RAW_STUDY, cache_dir=CACHE)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    summary = summarise(chunks)
    REPORT.write_text(json.dumps(summary, indent=2))
    print(f"wrote {len(chunks)} chunks -> {OUT.relative_to(ROOT)}")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

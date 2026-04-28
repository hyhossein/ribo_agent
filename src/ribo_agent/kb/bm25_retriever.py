"""BM25 sparse retriever for keyword-level matching.

Dense (FAISS) retrieval is great for semantic similarity but misses
exact statutory phrases like "s. 14(2)" or "Regulation 991". BM25
catches these by matching on raw tokens.

Used as one signal in the hybrid retriever (reciprocal rank fusion).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from .chunker import Chunk


@dataclass
class BM25Hit:
    chunk: Chunk
    score: float

    @property
    def citation(self) -> str:
        return self.chunk.citation

    @property
    def source(self) -> str:
        return self.chunk.source

    @property
    def text(self) -> str:
        return self.chunk.text


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)*", text.lower())


class BM25Retriever:
    """BM25Okapi over KB chunks."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        corpus = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(corpus)
        print(f"[bm25] index built — {len(chunks)} documents")

    def search(self, query: str, k: int = 10) -> list[BM25Hit]:
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [
            BM25Hit(chunk=self.chunks[i], score=float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

    @classmethod
    def from_chunks_jsonl(cls, path: str | Path) -> BM25Retriever:
        path = Path(path)
        chunks: list[Chunk] = []
        for line in path.open():
            d = json.loads(line)
            chunks.append(Chunk(**d))
        return cls(chunks)

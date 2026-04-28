"""FAISS-backed retriever over KB chunks.

Embeds chunks with sentence-transformers, indexes with FAISS, and
returns the top-k most relevant chunks for a given query — each with
its source citation so the agent can reference documents.

Usage:
    retriever = Retriever.from_chunks_jsonl("data/kb/chunks.jsonl")
    hits = retriever.search("What is the duty of an insurance broker?", k=5)
    for hit in hits:
        print(hit.citation, hit.score)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .chunker import Chunk


@dataclass
class RetrievalHit:
    """A single retrieval result with score and full chunk metadata."""
    chunk: Chunk
    score: float          # cosine similarity (higher = better)

    @property
    def citation(self) -> str:
        return self.chunk.citation

    @property
    def source(self) -> str:
        return self.chunk.source

    @property
    def text(self) -> str:
        return self.chunk.text


class Retriever:
    """Semantic search over KB chunks using sentence-transformers + FAISS."""

    def __init__(
        self,
        chunks: list[Chunk],
        model_name: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        self.chunks = chunks
        self.model_name = model_name
        print(f"[retriever] loading embedding model: {model_name}")
        self._encoder = SentenceTransformer(model_name)

        # embed all chunks
        texts = [c.text for c in chunks]
        print(f"[retriever] embedding {len(texts)} chunks ...")
        embeddings = self._encoder.encode(
            texts, show_progress_bar=True, normalize_embeddings=True,
        )
        self._embeddings = np.array(embeddings, dtype=np.float32)

        # build FAISS index (inner product on normalised vectors = cosine)
        dim = self._embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(self._embeddings)
        print(f"[retriever] index built — {self._index.ntotal} vectors, dim={dim}")

    def search(self, query: str, k: int = 5) -> list[RetrievalHit]:
        """Return the top-k most relevant chunks for `query`."""
        q_emb = self._encoder.encode(
            [query], normalize_embeddings=True,
        ).astype(np.float32)
        scores, indices = self._index.search(q_emb, k)
        hits: list[RetrievalHit] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            hits.append(RetrievalHit(chunk=self.chunks[idx], score=float(score)))
        return hits

    @classmethod
    def from_chunks_jsonl(
        cls,
        path: str | Path,
        model_name: str = "BAAI/bge-small-en-v1.5",
    ) -> Retriever:
        """Load chunks from a JSONL file and build the index."""
        path = Path(path)
        chunks: list[Chunk] = []
        for line in path.open():
            d = json.loads(line)
            chunks.append(Chunk(**d))
        return cls(chunks, model_name=model_name)

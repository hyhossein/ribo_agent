"""FAISS-backed retriever over KB chunks.

Embeds chunks with sentence-transformers, indexes with FAISS, and
returns the top-k most relevant chunks for a given query — each with
its source citation so the agent can reference documents.

Caches embeddings to disk (data/kb/embeddings.npy) so subsequent
loads skip the expensive encoding step entirely.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .chunker import Chunk

_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "kb"


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

        # Build a cache key from chunk count + first/last chunk ids
        cache_key = f"{model_name}_{len(chunks)}"
        if chunks:
            cache_key += f"_{chunks[0].chunk_id}_{chunks[-1].chunk_id}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
        cache_path = _CACHE_DIR / f"embeddings_{cache_hash}.npy"

        if cache_path.exists():
            print(f"[retriever] loading cached embeddings from {cache_path.name}")
            self._embeddings = np.load(cache_path)
        else:
            texts = [c.text for c in chunks]
            print(f"[retriever] embedding {len(texts)} chunks ...")
            embeddings = self._encoder.encode(
                texts, show_progress_bar=True, normalize_embeddings=True,
                batch_size=64,
            )
            self._embeddings = np.array(embeddings, dtype=np.float32)
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, self._embeddings)
            print(f"[retriever] cached embeddings to {cache_path.name}")

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

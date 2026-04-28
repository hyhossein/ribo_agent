"""Hybrid retriever — combines dense, sparse, and knowledge graph signals.

Reciprocal Rank Fusion (RRF) merges results from:
  1. FAISS dense retrieval  (semantic similarity)
  2. BM25 sparse retrieval  (keyword/exact match)
  3. Knowledge graph lookup  (structured cross-references)

RRF formula: score(d) = Σ 1 / (k + rank_i(d))
where k=60 (standard) and rank_i is the rank from retriever i.

This mirrors the approach from Karpathy's llm.c wiki and RAG papers —
ensemble retrieval consistently beats any single method.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from .chunker import Chunk
from .retriever import Retriever, RetrievalHit
from .bm25_retriever import BM25Retriever
from .knowledge_graph import KnowledgeGraph, build_knowledge_graph


@dataclass
class HybridHit:
    """A retrieval result with provenance from multiple retrievers."""
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    kg_rank: int | None = None
    source_signals: list[str] = None  # which retrievers contributed

    def __post_init__(self):
        if self.source_signals is None:
            self.source_signals = []

    @property
    def citation(self) -> str:
        return self.chunk.citation

    @property
    def source(self) -> str:
        return self.chunk.source

    @property
    def text(self) -> str:
        return self.chunk.text

    def to_citation_dict(self, rank: int) -> dict:
        return {
            "rank": rank,
            "source": self.source,
            "citation": self.citation,
            "section": self.chunk.section,
            "page_number": self.chunk.page_number,
            "score": round(self.score, 4),
            "snippet": self.text[:300],
            "retrieval_signals": self.source_signals,
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "kg_rank": self.kg_rank,
        }


def _reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, Chunk]]],
    k: int = 60,
) -> list[HybridHit]:
    """Fuse multiple ranked lists using RRF.

    Each input is a list of (chunk_id, Chunk) in rank order.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    ranks: dict[str, dict[str, int]] = {}  # chunk_id -> {signal: rank}
    signal_names = ["dense", "sparse", "kg"]

    for list_idx, ranked in enumerate(ranked_lists):
        signal = signal_names[list_idx] if list_idx < len(signal_names) else f"sig{list_idx}"
        for rank, (chunk_id, chunk) in enumerate(ranked, 1):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (k + rank)
            chunks[chunk_id] = chunk
            if chunk_id not in ranks:
                ranks[chunk_id] = {}
            ranks[chunk_id][signal] = rank

    # Sort by fused score
    sorted_ids = sorted(scores, key=lambda cid: -scores[cid])

    hits: list[HybridHit] = []
    for cid in sorted_ids:
        r = ranks.get(cid, {})
        signals = list(r.keys())
        hits.append(HybridHit(
            chunk=chunks[cid],
            score=scores[cid],
            dense_rank=r.get("dense"),
            sparse_rank=r.get("sparse"),
            kg_rank=r.get("kg"),
            source_signals=signals,
        ))
    return hits


class HybridRetriever:
    """Combines FAISS dense + BM25 sparse + KG for maximum recall."""

    def __init__(
        self,
        chunks: list[Chunk],
        *,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        self.dense = Retriever(chunks, model_name=embedding_model)
        self.sparse = BM25Retriever(chunks)
        self.kg = build_knowledge_graph(chunks)
        self.chunks = chunks
        self._chunk_map = {c.chunk_id: c for c in chunks}

    def search(self, query: str, k: int = 5) -> list[HybridHit]:
        """Retrieve top-k using RRF over all three signals."""
        # Dense retrieval (2x k for fusion headroom)
        dense_hits = self.dense.search(query, k=k * 2)
        dense_ranked = [(h.chunk.chunk_id, h.chunk) for h in dense_hits]

        # Sparse retrieval
        sparse_hits = self.sparse.search(query, k=k * 2)
        sparse_ranked = [(h.chunk.chunk_id, h.chunk) for h in sparse_hits]

        # KG retrieval — find matching nodes and their chunks
        kg_results = self.kg.search(query, k=k)
        kg_ranked: list[tuple[str, Chunk]] = []
        for node, _score in kg_results:
            for cid in node.chunk_ids:
                if cid in self._chunk_map:
                    kg_ranked.append((cid, self._chunk_map[cid]))
            # Also follow edges to get related chunks
            for neighbor in self.kg.get_neighbors(node.id)[:3]:
                for cid in neighbor.chunk_ids:
                    if cid in self._chunk_map:
                        kg_ranked.append((cid, self._chunk_map[cid]))

        # Fuse
        fused = _reciprocal_rank_fusion([dense_ranked, sparse_ranked, kg_ranked])
        return fused[:k]

    @classmethod
    def from_chunks_jsonl(
        cls,
        path: str | Path,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ) -> HybridRetriever:
        path = Path(path)
        chunks: list[Chunk] = []
        for line in path.open():
            d = json.loads(line)
            chunks.append(Chunk(**d))
        return cls(chunks, embedding_model=embedding_model)

"""Knowledge graph builder — extract entities and relations from KB chunks.

Builds a lightweight in-memory graph of legal concepts:
  - Nodes: Acts, Regulations, Sections, Definitions, Concepts
  - Edges: DEFINES, REQUIRES, REFERENCES, PART_OF

No external NLP model needed — uses domain-specific regex patterns
tuned for Ontario insurance law documents.

The graph enables:
  1. Following cross-references (e.g. "see s. 14" -> fetch that section)
  2. Answering "what does X mean?" via definition nodes
  3. Providing richer context than pure vector search
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict

from .chunker import Chunk


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class KGNode:
    id: str
    type: str                 # ACT, REGULATION, SECTION, DEFINITION, CONCEPT
    label: str                # human-readable label
    source: str               # source document
    chunk_ids: list[str] = field(default_factory=list)
    properties: dict = field(default_factory=dict)


@dataclass
class KGEdge:
    src: str                  # node id
    dst: str                  # node id
    relation: str             # DEFINES, REQUIRES, REFERENCES, PART_OF
    properties: dict = field(default_factory=dict)


class KnowledgeGraph:
    """In-memory knowledge graph over RIBO study corpus."""

    def __init__(self) -> None:
        self.nodes: dict[str, KGNode] = {}
        self.edges: list[KGEdge] = []
        self._adj: dict[str, list[KGEdge]] = defaultdict(list)
        self._chunks: dict[str, Chunk] = {}

    def add_node(self, node: KGNode) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: KGEdge) -> None:
        self.edges.append(edge)
        self._adj[edge.src].append(edge)
        self._adj[edge.dst].append(edge)

    def get_neighbors(self, node_id: str, relation: str | None = None) -> list[KGNode]:
        """Get all nodes connected to node_id, optionally filtered by relation."""
        result: list[KGNode] = []
        for edge in self._adj.get(node_id, []):
            other = edge.dst if edge.src == node_id else edge.src
            if relation is None or edge.relation == relation:
                if other in self.nodes:
                    result.append(self.nodes[other])
        return result

    def search(self, query: str, k: int = 5) -> list[tuple[KGNode, float]]:
        """Simple keyword search over node labels and properties."""
        query_lower = query.lower()
        query_tokens = set(re.findall(r"[a-z0-9]+", query_lower))
        scored: list[tuple[KGNode, float]] = []
        for node in self.nodes.values():
            label_tokens = set(re.findall(r"[a-z0-9]+", node.label.lower()))
            overlap = len(query_tokens & label_tokens)
            if overlap > 0:
                score = overlap / max(len(query_tokens), 1)
                scored.append((node, score))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def get_chunks_for_node(self, node_id: str) -> list[Chunk]:
        """Return the original chunks associated with a node."""
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [self._chunks[cid] for cid in node.chunk_ids if cid in self._chunks]

    def stats(self) -> dict:
        type_counts = defaultdict(int)
        for n in self.nodes.values():
            type_counts[n.type] += 1
        rel_counts = defaultdict(int)
        for e in self.edges:
            rel_counts[e.relation] += 1
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "node_types": dict(type_counts),
            "edge_types": dict(rel_counts),
        }

    def to_dict(self) -> dict:
        return {
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges],
        }


# ---------------------------------------------------------------------------
# Regex-based entity extraction
# ---------------------------------------------------------------------------

_DEFINITION_RE = re.compile(
    r'"([^"]{5,80})"\s+means\s+(.{20,300}?)(?:\.|;)',
    re.IGNORECASE,
)

_SECTION_REF_RE = re.compile(
    r"(?:section|s\.)\s*(\d+(?:\(\d+\))?(?:\.\d+)?)",
    re.IGNORECASE,
)

_REGULATION_REF_RE = re.compile(
    r"(?:Regulation|Reg\.?)\s*(\d{3})",
    re.IGNORECASE,
)

_ACT_REF_RE = re.compile(
    r"((?:Registered\s+Insurance\s+Brokers?\s+Act|RIB\s+Act|Insurance\s+Act)(?:\s*,?\s*\d{4})?)",
    re.IGNORECASE,
)

_BYLAW_REF_RE = re.compile(
    r"(By-?Law\s+(?:No\.?\s*)?\d+)",
    re.IGNORECASE,
)

_OAP_REF_RE = re.compile(
    r"(OAP\s+\d+|Ontario\s+Automobile\s+Policy)",
    re.IGNORECASE,
)

_DUTY_RE = re.compile(
    r"(?:shall|must|is required to|has a duty to)\s+(.{10,120}?)(?:\.|;)",
    re.IGNORECASE,
)


def build_knowledge_graph(chunks: list[Chunk]) -> KnowledgeGraph:
    """Extract entities and relations from chunks to build a KG."""
    kg = KnowledgeGraph()

    # Register all chunks
    for chunk in chunks:
        kg._chunks[chunk.chunk_id] = chunk

    # Create source document nodes
    sources_seen: set[str] = set()
    for chunk in chunks:
        if chunk.source not in sources_seen:
            sources_seen.add(chunk.source)
            doc_type = "ACT" if "Act" in chunk.source else (
                "REGULATION" if "Regulation" in chunk.source or "Reg" in chunk.source else
                "BYLAW" if "By-Law" in chunk.source or "ByLaw" in chunk.source else
                "POLICY"
            )
            kg.add_node(KGNode(
                id=f"doc:{chunk.source}",
                type=doc_type,
                label=chunk.source.replace("_", " "),
                source=chunk.source,
            ))

    # Process each chunk
    for chunk in chunks:
        # Create section node
        if chunk.section:
            sec_id = f"sec:{chunk.source}:{chunk.section}"
            if sec_id not in kg.nodes:
                kg.add_node(KGNode(
                    id=sec_id,
                    type="SECTION",
                    label=chunk.citation,
                    source=chunk.source,
                    chunk_ids=[chunk.chunk_id],
                    properties={
                        "section": chunk.section,
                        "title": chunk.title,
                        "page_number": chunk.page_number,
                    },
                ))
                # PART_OF edge to document
                kg.add_edge(KGEdge(
                    src=sec_id,
                    dst=f"doc:{chunk.source}",
                    relation="PART_OF",
                ))
            else:
                kg.nodes[sec_id].chunk_ids.append(chunk.chunk_id)

        # Extract definitions
        for match in _DEFINITION_RE.finditer(chunk.text):
            term = match.group(1).strip()
            definition = match.group(2).strip()
            def_id = f"def:{term.lower().replace(' ', '_')}"
            if def_id not in kg.nodes:
                kg.add_node(KGNode(
                    id=def_id,
                    type="DEFINITION",
                    label=term,
                    source=chunk.source,
                    chunk_ids=[chunk.chunk_id],
                    properties={"definition": definition[:300]},
                ))
            # DEFINES edge
            if chunk.section:
                sec_id = f"sec:{chunk.source}:{chunk.section}"
                kg.add_edge(KGEdge(
                    src=sec_id,
                    dst=def_id,
                    relation="DEFINES",
                ))

        # Extract cross-references to other sections
        if chunk.section:
            src_sec_id = f"sec:{chunk.source}:{chunk.section}"
            for m in _SECTION_REF_RE.finditer(chunk.text):
                ref_sec = m.group(1)
                if ref_sec != chunk.section:
                    ref_id = f"sec:{chunk.source}:{ref_sec}"
                    kg.add_edge(KGEdge(
                        src=src_sec_id,
                        dst=ref_id,
                        relation="REFERENCES",
                    ))

        # Extract references to regulations
        for m in _REGULATION_REF_RE.finditer(chunk.text):
            reg_num = m.group(1)
            reg_source = f"Ontario_Regulation_{reg_num}"
            if f"doc:{reg_source}" in kg.nodes and chunk.section:
                kg.add_edge(KGEdge(
                    src=f"sec:{chunk.source}:{chunk.section}",
                    dst=f"doc:{reg_source}",
                    relation="REFERENCES",
                ))

        # Extract duties/requirements as concept nodes
        for m in _DUTY_RE.finditer(chunk.text):
            duty_text = m.group(1).strip()
            duty_id = f"duty:{chunk.chunk_id}:{m.start()}"
            kg.add_node(KGNode(
                id=duty_id,
                type="CONCEPT",
                label=f"Duty: {duty_text[:80]}",
                source=chunk.source,
                chunk_ids=[chunk.chunk_id],
                properties={"full_text": duty_text},
            ))
            if chunk.section:
                kg.add_edge(KGEdge(
                    src=f"sec:{chunk.source}:{chunk.section}",
                    dst=duty_id,
                    relation="REQUIRES",
                ))

    print(f"[kg] built knowledge graph: {kg.stats()}")
    return kg


def build_from_jsonl(path: str | Path) -> KnowledgeGraph:
    """Build KG from a chunks.jsonl file."""
    path = Path(path)
    chunks: list[Chunk] = []
    for line in path.open():
        d = json.loads(line)
        chunks.append(Chunk(**d))
    return build_knowledge_graph(chunks)


def save_kg(kg: KnowledgeGraph, path: str | Path) -> None:
    """Save KG to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(kg.to_dict(), indent=2, ensure_ascii=False))
    print(f"[kg] saved to {path}")

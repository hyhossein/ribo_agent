"""Streamlit chat interface — interactive RIBO exam Q&A with full traceability.

Launch:
    streamlit run src/ribo_agent/chat.py

Features:
  - Multi-turn conversation with session history
  - Real-time citation display (doc → page → sentence)
  - Retrieval signal breakdown (dense / sparse / KG)
  - Cost & latency dashboard per question and cumulative
  - Knowledge graph context panel
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = ROOT / "data" / "kb" / "chunks.jsonl"


@st.cache_resource
def _load_retriever():
    """Load hybrid retriever (cached across reruns)."""
    from .kb.chunker import Chunk
    from .kb.hybrid_retriever import HybridRetriever

    chunks = [Chunk(**json.loads(line)) for line in CHUNKS_PATH.open()]
    return HybridRetriever(chunks, embedding_model="BAAI/bge-small-en-v1.5")


@st.cache_resource
def _load_llm():
    """Load Ollama LLM client."""
    from .llm.factory import make_client
    return make_client({
        "backend": "ollama",
        "model": "qwen2.5:7b-instruct",
        "base_url": "http://localhost:11434",
    })


SYSTEM_PROMPT = (
    "You are an expert Ontario insurance broker preparing for the RIBO Level 1 "
    "licensing exam. Answer questions accurately using the provided reference "
    "documents. Always cite the relevant document and page when possible. "
    "If the documents don't contain the answer, say so clearly."
)


def _ask(query: str, retriever, llm, top_k: int = 5) -> dict:
    """Run a single query through hybrid retrieval + LLM."""
    t0 = time.perf_counter()
    hits = retriever.search(query, k=top_k)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    # Build context
    context_blocks = []
    for i, hit in enumerate(hits, 1):
        page = f" [page {hit.chunk.page_number}]" if hit.chunk.page_number else ""
        signals = ", ".join(hit.source_signals)
        header = f"[{i}] {hit.citation}{page} ({signals})"
        context_blocks.append(f"{header}\n{hit.text[:500].rstrip()}")
    context = "\n\n".join(context_blocks)

    prompt = f"""{SYSTEM_PROMPT}

Reference documents:
{context}

---

User question: {query}

Provide a clear, accurate answer. Reference the document numbers [1], [2], etc. when citing information."""

    t1 = time.perf_counter()
    resp = llm.complete(prompt, temperature=0.0, max_tokens=512)
    generation_ms = (time.perf_counter() - t1) * 1000

    citations = [hit.to_citation_dict(rank=i) for i, hit in enumerate(hits, 1)]

    return {
        "answer": resp.text,
        "citations": citations,
        "retrieval_ms": round(retrieval_ms, 1),
        "generation_ms": round(generation_ms, 1),
        "total_ms": round(retrieval_ms + generation_ms, 1),
        "prompt_tokens": resp.prompt_tokens or 0,
        "completion_tokens": resp.completion_tokens or 0,
        "model": resp.model,
    }


def main():
    st.set_page_config(
        page_title="RIBO Agent Chat",
        page_icon="💬",
        layout="wide",
    )

    st.title("💬 RIBO Agent — Interactive Q&A")
    st.caption("Ask any Ontario insurance licensing question. Answers include document citations with page references.")

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0
    if "total_time_ms" not in st.session_state:
        st.session_state.total_time_ms = 0.0

    # Sidebar — cumulative stats
    with st.sidebar:
        st.header("📊 Session Stats")
        st.metric("Questions Asked", st.session_state.total_queries)
        st.metric("Total Tokens", f"{st.session_state.total_tokens:,}")
        st.metric("Total Time", f"{st.session_state.total_time_ms / 1000:.1f}s")
        if st.session_state.total_queries > 0:
            avg = st.session_state.total_time_ms / st.session_state.total_queries
            st.metric("Avg Latency", f"{avg:.0f} ms")

        st.divider()
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.total_tokens = 0
            st.session_state.total_queries = 0
            st.session_state.total_time_ms = 0.0
            st.rerun()

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "metadata" in msg:
                meta = msg["metadata"]
                # Citations expander
                with st.expander("📚 Citations & Sources"):
                    for cit in meta.get("citations", []):
                        signals = ", ".join(cit.get("retrieval_signals", []))
                        page = cit.get("page_number", "?")
                        st.markdown(
                            f"**[{cit['rank']}] {cit['citation']}** "
                            f"(page {page}, score: {cit['score']:.3f}, via: {signals})"
                        )
                        st.text(cit.get("snippet", "")[:250])
                # Performance
                with st.expander("⏱️ Performance"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Retrieval", f"{meta['retrieval_ms']} ms")
                    c2.metric("Generation", f"{meta['generation_ms']} ms")
                    c3.metric("Total", f"{meta['total_ms']} ms")
                    st.caption(
                        f"Tokens: {meta['prompt_tokens']} prompt + "
                        f"{meta['completion_tokens']} completion"
                    )

    # Input
    if query := st.chat_input("Ask a RIBO exam question..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving & generating..."):
                retriever = _load_retriever()
                llm = _load_llm()
                result = _ask(query, retriever, llm)

            st.markdown(result["answer"])

            with st.expander("📚 Citations & Sources"):
                for cit in result["citations"]:
                    signals = ", ".join(cit.get("retrieval_signals", []))
                    page = cit.get("page_number", "?")
                    st.markdown(
                        f"**[{cit['rank']}] {cit['citation']}** "
                        f"(page {page}, score: {cit['score']:.3f}, via: {signals})"
                    )
                    st.text(cit.get("snippet", "")[:250])

            with st.expander("⏱️ Performance"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Retrieval", f"{result['retrieval_ms']} ms")
                c2.metric("Generation", f"{result['generation_ms']} ms")
                c3.metric("Total", f"{result['total_ms']} ms")
                st.caption(
                    f"Tokens: {result['prompt_tokens']} prompt + "
                    f"{result['completion_tokens']} completion"
                )

        # Update session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "metadata": result,
        })
        st.session_state.total_queries += 1
        st.session_state.total_tokens += result["prompt_tokens"] + result["completion_tokens"]
        st.session_state.total_time_ms += result["total_ms"]


if __name__ == "__main__":
    main()

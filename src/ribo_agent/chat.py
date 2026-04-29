"""Streamlit chat interface — interactive RIBO exam Q&A with full traceability.

Launch:
    streamlit run src/ribo_agent/chat.py

Features:
  - Every answer starts with "✅ Answer: X is correct"
  - Expandable chain-of-thought reasoning
  - Document cards: Title → Section → Key sentence
  - Browse the 169 eval questions directly
  - Confidence badge + performance metrics
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

# Fix imports: add src/ to path so absolute imports work with `streamlit run`
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

from ribo_agent.kb.chunker import Chunk  # noqa: E402
from ribo_agent.kb.hybrid_retriever import HybridRetriever  # noqa: E402
from ribo_agent.llm.factory import make_client  # noqa: E402

CHUNKS_PATH = ROOT / "data" / "kb" / "chunks.jsonl"
EVAL_PATH = ROOT / "data" / "parsed" / "eval.jsonl"


@st.cache_resource
def _load_retriever():
    """Load hybrid retriever (cached across reruns)."""
    chunks = [Chunk(**json.loads(line)) for line in CHUNKS_PATH.open()]
    return HybridRetriever(chunks, embedding_model="BAAI/bge-small-en-v1.5")


@st.cache_resource
def _load_llm():
    """Load Ollama LLM client."""
    return make_client({
        "backend": "ollama",
        "model": "qwen2.5:7b-instruct",
        "base_url": "http://localhost:11434",
    })


def _load_eval_questions() -> list[dict]:
    """Load the 169 eval MCQs."""
    if not EVAL_PATH.exists():
        return []
    return [json.loads(line) for line in EVAL_PATH.open()]


_SOURCE_LABELS = {
    "RIB_Act_1990": "Registered Insurance Brokers Act",
    "Ont_Reg_991": "Ontario Regulation 991",
    "RIBO_Bylaws": "RIBO Bylaws",
    "OAP_1": "Ontario Auto Policy (OAP-1)",
}


def _clean_source(raw: str) -> str:
    return _SOURCE_LABELS.get(raw, raw.replace("_", " "))


SYSTEM_PROMPT = """You are an expert Ontario insurance professional.

You MUST structure your response EXACTLY like this:

ANSWER: <letter>
REASON: <1-2 sentence explanation of why this is correct>
THINKING: <your step-by-step chain of thought — examine each option, eliminate wrong ones, explain the principle>

Rules:
- ALWAYS give a definitive answer. Never say "I don't know".
- If the documents help, cite them as [1], [2] etc.
- If the documents don't cover the topic, answer from your knowledge.
- The ANSWER line must contain exactly one letter (A, B, C, or D) for MCQs, or a direct answer for open questions."""


def _parse_structured_response(raw: str) -> dict:
    """Parse the LLM output into answer / reason / thinking."""
    result = {"answer_line": "", "reason": "", "thinking": "", "raw": raw}

    # Try to extract ANSWER: line
    m = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", raw, re.IGNORECASE)
    if m:
        result["answer_line"] = m.group(1).strip()

    # Try to extract REASON: line
    m = re.search(r"REASON:\s*(.+?)(?:\nTHINKING:|\n\n|$)", raw, re.IGNORECASE | re.DOTALL)
    if m:
        result["reason"] = m.group(1).strip()

    # Try to extract THINKING: block
    m = re.search(r"THINKING:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
    if m:
        result["thinking"] = m.group(1).strip()

    # Fallback: if no structured format, use the whole thing
    if not result["answer_line"] and not result["reason"]:
        # Try <answer>X</answer> format
        m2 = re.search(r"<answer>\s*([A-D])\s*</answer>", raw, re.IGNORECASE)
        if m2:
            result["answer_line"] = m2.group(1).upper()
            result["reason"] = raw.replace(m2.group(0), "").strip()[:300]
        else:
            result["answer_line"] = "See below"
            result["reason"] = raw[:500]

    return result


def _extract_key_sentence(snippet: str) -> str:
    """Pull the most informative sentence from a chunk snippet."""
    sentences = re.split(r'(?<=[.!?])\s+', snippet.strip())
    # Pick the longest sentence (usually most informative)
    if sentences:
        best = max(sentences, key=len)
        return best[:200]
    return snippet[:200]


def _ask(query: str, retriever, llm, top_k: int = 5) -> dict:
    """Run a single query through hybrid retrieval + LLM."""
    t0 = time.perf_counter()
    hits = retriever.search(query, k=top_k)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    # Build context
    context_blocks = []
    for i, hit in enumerate(hits, 1):
        doc_name = _clean_source(hit.source)
        section = f"§ {hit.chunk.section}" if hit.chunk.section else ""
        page = f"Page {hit.chunk.page_number}" if hit.chunk.page_number else ""
        header_parts = [p for p in [doc_name, section, page] if p]
        header = f"[{i}] {' — '.join(header_parts)}"
        context_blocks.append(f"{header}\n{hit.text[:500].rstrip()}")
    context = "\n\n".join(context_blocks)

    prompt = f"""{SYSTEM_PROMPT}

Reference documents:
{context}

---

User question: {query}"""

    t1 = time.perf_counter()
    resp = llm.complete(prompt, temperature=0.0, max_tokens=400)
    generation_ms = (time.perf_counter() - t1) * 1000

    parsed = _parse_structured_response(resp.text)

    # Build document cards
    doc_cards = []
    for i, hit in enumerate(hits, 1):
        key_sentence = _extract_key_sentence(hit.text)
        doc_cards.append({
            "rank": i,
            "title": _clean_source(hit.source),
            "section": hit.chunk.section,
            "page": hit.chunk.page_number,
            "sentence": key_sentence,
            "full_text": hit.text[:600],
            "score": round(hit.score, 4),
            "signals": hit.source_signals,
        })

    top_score = hits[0].score if hits else 0
    confidence = round(min(1.0, top_score * 0.6 + 0.3), 3) if hits else 0.2

    return {
        "parsed": parsed,
        "doc_cards": doc_cards,
        "confidence": confidence,
        "retrieval_ms": round(retrieval_ms, 1),
        "generation_ms": round(generation_ms, 1),
        "total_ms": round(retrieval_ms + generation_ms, 1),
        "prompt_tokens": resp.prompt_tokens or 0,
        "completion_tokens": resp.completion_tokens or 0,
        "model": resp.model,
    }


def _render_answer(result: dict):
    """Render a single answer with all visual components."""
    parsed = result["parsed"]
    conf = result.get("confidence", 0)

    # ── 1. Big answer banner ──
    answer_letter = parsed["answer_line"]
    conf_icon = "🟢" if conf >= 0.7 else "🟡" if conf >= 0.4 else "🔴"
    st.markdown(
        f"### {conf_icon} Answer: **{answer_letter}**\n\n"
        f"{parsed['reason']}"
    )
    st.caption(f"Confidence: {conf:.0%}")

    # ── 2. Chain of thought (expandable) ──
    if parsed.get("thinking"):
        with st.expander("🧠 Chain of Thought — Step-by-step reasoning"):
            st.markdown(parsed["thinking"])

    # ── 3. Document cards ──
    docs = result.get("doc_cards", [])
    if docs:
        with st.expander(f"📚 Source Documents ({len(docs)} retrieved)"):
            for doc in docs:
                title = doc["title"]
                section = f"§ {doc['section']}" if doc["section"] else ""
                page = f"Page {doc['page']}" if doc["page"] else ""
                signals = ", ".join(doc["signals"])
                score_bar = "█" * int(doc["score"] * 20)

                # Document header
                header_parts = [f"**{title}**"]
                if section:
                    header_parts.append(section)
                if page:
                    header_parts.append(page)
                st.markdown(f"**[{doc['rank']}]** " + " — ".join(header_parts))

                # Key sentence in a quote block
                st.markdown(f"> *\"{doc['sentence']}\"*")

                # Meta info
                st.caption(
                    f"Relevance: {doc['score']:.3f} {score_bar}  •  "
                    f"Found via: {signals}"
                )

                # Full text on hover/expand
                with st.popover(f"📄 Full text from [{doc['rank']}]"):
                    st.markdown(f"**{title}**" + (f" — {section}" if section else "") + (f" — {page}" if page else ""))
                    st.text(doc["full_text"])

                st.divider()

    # ── 4. Performance ──
    with st.expander("⏱️ Performance"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Retrieval", f"{result['retrieval_ms']} ms")
        c2.metric("Generation", f"{result['generation_ms']} ms")
        c3.metric("Total", f"{result['total_ms']} ms")
        st.caption(
            f"Tokens: {result['prompt_tokens']} prompt + "
            f"{result['completion_tokens']} completion  •  "
            f"Model: {result.get('model', '?')}"
        )


def main():
    st.set_page_config(page_title="RIBO Agent Chat", page_icon="💬", layout="wide")

    st.title("💬 RIBO Agent — Interactive Q&A")
    st.caption(
        "Ask any Ontario insurance question or pick one from the 169 exam bank. "
        "Every answer: definitive answer + reasoning + source documents (document → section → sentence)."
    )

    # Session state
    for key, default in [("messages", []), ("total_tokens", 0),
                         ("total_queries", 0), ("total_time_ms", 0.0),
                         ("selected_q", None)]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Sidebar ──
    eval_qs = _load_eval_questions()

    with st.sidebar:
        st.header("📊 Session")
        c1, c2 = st.columns(2)
        c1.metric("Qs", st.session_state.total_queries)
        c2.metric("Tokens", f"{st.session_state.total_tokens:,}")
        if st.session_state.total_queries > 0:
            avg = st.session_state.total_time_ms / st.session_state.total_queries
            st.caption(f"Avg latency: {avg:.0f} ms")

        st.divider()
        st.header(f"📋 Exam Bank ({len(eval_qs)} Qs)")
        domains = sorted({q.get("content_domain") or "Other" for q in eval_qs})
        sel_domain = st.selectbox("Domain", ["All"] + domains)
        filtered = eval_qs if sel_domain == "All" else [
            q for q in eval_qs if q.get("content_domain") == sel_domain]

        if filtered:
            labels = [f"Q{i+1}: {q['stem'][:55]}..." for i, q in enumerate(filtered)]
            idx = st.selectbox("Question", range(len(labels)), format_func=lambda i: labels[i])
            if st.button("📨 Send this question"):
                q = filtered[idx]
                st.session_state.selected_q = (
                    f"{q['stem']}\n\nA. {q['options']['A']}\n"
                    f"B. {q['options']['B']}\nC. {q['options']['C']}\n"
                    f"D. {q['options']['D']}")
                st.rerun()

        st.divider()
        if st.button("Clear Chat"):
            for k in ["messages", "total_tokens", "total_queries", "total_time_ms"]:
                st.session_state[k] = [] if k == "messages" else 0
            st.rerun()

    # ── Chat history ──
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            elif "metadata" in msg:
                _render_answer(msg["metadata"])
            else:
                st.markdown(msg.get("content", ""))

    # ── Input ──
    query = st.chat_input("Ask a RIBO exam question...")
    if st.session_state.selected_q:
        query = st.session_state.selected_q
        st.session_state.selected_q = None

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving & answering..."):
                result = _ask(query, _load_retriever(), _load_llm())
            _render_answer(result)
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["parsed"]["answer_line"],
            "metadata": result,
        })
        st.session_state.total_queries += 1
        st.session_state.total_tokens += result["prompt_tokens"] + result["completion_tokens"]
        st.session_state.total_time_ms += result["total_ms"]


if __name__ == "__main__":
    main()

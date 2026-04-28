"""Streamlit app for exploring RIBO Agent evaluation results.

Launch:
    streamlit run src/ribo_agent/app.py

Browse per-question predictions, citations, faithfulness verdicts,
and drill down into the source documents that informed each answer.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import streamlit as st

# We import lazily so the app can at least start even if some deps are missing
try:
    from .eval.faithfulness import check_faithfulness, FaithfulnessVerdict
except ImportError:
    check_faithfulness = None  # type: ignore


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "runs"


def _find_runs() -> list[Path]:
    """Find all evaluation run directories."""
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        [d for d in RESULTS_DIR.iterdir() if d.is_dir() and (d / "predictions.jsonl").exists()],
        reverse=True,
    )


def _load_predictions(run_dir: Path) -> list[dict]:
    preds = []
    for line in (run_dir / "predictions.jsonl").open():
        preds.append(json.loads(line))
    return preds


def _load_metrics(run_dir: Path) -> dict | None:
    p = run_dir / "metrics.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def _verdict_color(verdict: str) -> str:
    return {
        "GROUNDED": "🟢",
        "PARTIAL": "🟡",
        "UNGROUNDED": "🔴",
        "NO_CONTEXT": "⚪",
    }.get(verdict, "❓")


def main():
    st.set_page_config(
        page_title="RIBO Agent Explorer",
        page_icon="📋",
        layout="wide",
    )
    st.title("📋 RIBO Agent — Evaluation Explorer")

    runs = _find_runs()
    if not runs:
        st.warning(
            "No evaluation runs found in `results/runs/`. "
            "Run `make eval` first to generate results."
        )
        st.stop()

    # --- Sidebar: run selector ---
    st.sidebar.header("Select Run")
    run_names = [r.name for r in runs]
    selected_name = st.sidebar.selectbox("Run", run_names)
    run_dir = RESULTS_DIR / selected_name

    preds = _load_predictions(run_dir)
    metrics = _load_metrics(run_dir)

    # --- Sidebar: filters ---
    st.sidebar.header("Filters")
    show_only = st.sidebar.multiselect(
        "Show",
        ["Correct", "Wrong", "Refused"],
        default=["Correct", "Wrong", "Refused"],
    )
    faith_filter = st.sidebar.multiselect(
        "Faithfulness",
        ["GROUNDED", "PARTIAL", "UNGROUNDED", "NO_CONTEXT"],
        default=["GROUNDED", "PARTIAL", "UNGROUNDED", "NO_CONTEXT"],
    )

    # --- Overview tab ---
    tab_overview, tab_questions, tab_citations = st.tabs(
        ["📊 Overview", "❓ Questions", "📚 Citations"]
    )

    with tab_overview:
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
            col2.metric("Macro-F1", f"{metrics['macro_f1']:.4f}")
            col3.metric("Questions", metrics["n"])
            col4.metric("Refusal Rate", f"{metrics['refusal_rate']:.1%}")

            # Faithfulness breakdown
            if check_faithfulness and preds and preds[0].get("citations"):
                st.subheader("Faithfulness Breakdown")
                verdicts = {"GROUNDED": 0, "PARTIAL": 0, "UNGROUNDED": 0, "NO_CONTEXT": 0}
                for pred in preds:
                    fr = check_faithfulness(pred.get("raw_response", ""), pred.get("citations", []))
                    verdicts[fr.verdict.value] += 1
                cols = st.columns(4)
                for col, (v, count) in zip(cols, verdicts.items()):
                    col.metric(f"{_verdict_color(v)} {v}", count)

            # Per-class breakdown
            if "per_class" in metrics:
                st.subheader("Per-Class Performance")
                rows = []
                for label, vals in metrics["per_class"].items():
                    rows.append({
                        "Class": label,
                        "Support": vals["support"],
                        "Precision": f"{vals['precision']:.4f}",
                        "Recall": f"{vals['recall']:.4f}",
                        "F1": f"{vals['f1']:.4f}",
                    })
                st.table(rows)

    with tab_questions:
        st.subheader(f"Per-Question Results ({len(preds)} questions)")

        for i, pred in enumerate(preds):
            # Apply filters
            is_correct = pred.get("predicted") == pred.get("correct")
            is_refused = pred.get("predicted") is None
            if is_correct and "Correct" not in show_only:
                continue
            if not is_correct and not is_refused and "Wrong" not in show_only:
                continue
            if is_refused and "Refused" not in show_only:
                continue

            # Faithfulness
            faith_verdict = "NO_CONTEXT"
            faith_score = 0.0
            if check_faithfulness:
                fr = check_faithfulness(
                    pred.get("raw_response", ""),
                    pred.get("citations", []),
                )
                faith_verdict = fr.verdict.value
                faith_score = fr.grounding_score

            if faith_verdict not in faith_filter:
                continue

            # Display
            status = "✅" if is_correct else ("⏭️" if is_refused else "❌")
            with st.expander(
                f"{status} Q{i+1}: {pred['qid']} — "
                f"Predicted: {pred.get('predicted', 'REFUSED')} "
                f"(correct: {pred['correct']}) "
                f"{_verdict_color(faith_verdict)} {faith_verdict}"
            ):
                col1, col2, col3 = st.columns(3)
                col1.metric("Answer", pred.get("predicted", "REFUSED"))
                col2.metric("Correct", pred["correct"])
                col3.metric("Grounding", f"{faith_score:.2f}")

                # Citations
                citations = pred.get("citations", [])
                if citations:
                    st.markdown("**📚 Retrieved Documents:**")
                    for j, cit in enumerate(citations, 1):
                        st.markdown(
                            f"**{j}. {cit.get('citation', '?')}** "
                            f"(source: `{cit.get('source', '?')}`, "
                            f"relevance: {cit.get('score', 0):.2f})"
                        )
                        snippet = cit.get("snippet", "")
                        if snippet:
                            st.text(snippet[:300])

                # Model reasoning
                raw = pred.get("raw_response", "")
                if raw:
                    st.markdown("**🧠 Model Reasoning:**")
                    st.code(raw[:800], language=None)

                # Latency
                lat = pred.get("latency_ms")
                if lat:
                    st.caption(f"⏱️ Latency: {lat:.0f} ms")

    with tab_citations:
        st.subheader("Citation Analysis")
        st.markdown(
            "Aggregated view of which documents are cited most frequently "
            "across all questions."
        )

        source_counts: dict[str, int] = {}
        citation_counts: dict[str, int] = {}
        for pred in preds:
            for cit in pred.get("citations", []):
                src = cit.get("source", "unknown")
                ref = cit.get("citation", "unknown")
                source_counts[src] = source_counts.get(src, 0) + 1
                citation_counts[ref] = citation_counts.get(ref, 0) + 1

        if source_counts:
            st.markdown("**Most-cited sources:**")
            for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1])[:15]:
                st.markdown(f"- `{src}`: {cnt} citations")

            st.markdown("**Most-cited sections:**")
            for ref, cnt in sorted(citation_counts.items(), key=lambda x: -x[1])[:20]:
                st.markdown(f"- **{ref}**: {cnt} citations")
        else:
            st.info("No citations found. Run a RAG evaluation to see citation data.")


if __name__ == "__main__":
    main()

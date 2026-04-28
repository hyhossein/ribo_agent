# RIBO Evidence Chat

Interactive chat interface for Ontario insurance exam QA with full pipeline traceability.

## Quick Start

    cd ~/ribo_agent
    source .venv/bin/activate
    export ANTHROPIC_API_KEY="your-key"

    # Local
    python chat_explorer.py          # http://localhost:5001

    # Network (share with team)
    python3 -c "import chat_explorer; chat_explorer.app.run(host='0.0.0.0', port=8888)"

## Architecture

    Question
        |
        +-> [1] Wiki Lookup (0ms)        -> 70K compiled knowledge, 3 best sections
        |
        +-> [2] KB Retrieval (0ms)       -> 297 statutory chunks, semantic (FAISS) or BM25
        |
        +-> [3] LLM Reasoning (Opus ~5s) -> answer + chain-of-thought + [SOURCE: doc | section | quote]
        |
        +-> [4] Majority Vote (3x ~10s)  -> self-consistency, confidence from agreement
                |
                v
          Answer + Evidence Trail + Confidence Score

## Three-Panel Interface

| Panel | What it shows |
|-------|--------------|
| Left: Chat | Answer with confidence + reasoning + evidence cards (wiki fed, KB fed, LLM citations, votes) |
| Middle: Trace | Expandable pipeline steps with timing per step |
| Right: Compare | Same question through zero-shot / wiki / ensemble side by side |

## Evidence Trail

Every answer shows:
- CITED BY LLM: sections the model explicitly referenced
- WIKI CONTEXT FED: which compiled wiki sections were input
- SOURCE DOCS FED: which statutory chunks were retrieved
- VOTE: majority voting results with agreement percentage

## Knowledge Layers

| Layer | Size | Source |
|-------|------|--------|
| Compiled Wiki | 70K chars | LLM-compiled from 8 study documents (Karpathy pattern) |
| KB Chunks | 297 chunks | Section-level splits from OAP, By-Laws, Act, Regulations |
| FAISS Index | 297 x 768d | BGE-base-en-v1.5 embeddings for semantic search |
| QLoRA Adapter | 7B params | Fine-tuned Qwen-2.5-7B on RIBO training data |

## Eval Results

| Architecture | Accuracy | Latency |
|-------------|----------|---------|
| 3-Way Vote (Opus) | 91.72% | ~15s |
| Rewrite + Wiki (Opus) | 88.76% | ~6s |
| Zero-shot (Opus) | 78.70% | ~5s |
| QLoRA v3 (Qwen 7B) | 65.68% | ~2s |

## Files

    chat_explorer.py              Main Flask app
    run_full_eval.py              Full 169-question eval with checkpoint
    explorer.py                   Basic evidence explorer (v1)
    data/kb/wiki_compiled.md      70K compiled knowledge wiki
    data/kb/chunks.jsonl          297 statutory chunks
    data/kb/faiss.index           FAISS semantic search index
    data/kb/embeddings.npy        BGE embeddings (297 x 768)
    data/kb/chunk_meta.json       Chunk metadata for FAISS
    src/ribo_agent/agents/multistep_agent.py   5-step reasoning agent
    configs/v4_multistep_opus.yaml             Multistep config

## Cost Per Query

| Operation | Cost |
|-----------|------|
| Single question (wiki only) | ~$0.05 |
| Single question (ensemble + vote) | ~$0.15 |
| Full 169-question eval | ~$25 |
| Wiki compilation (one-time) | ~$5 |

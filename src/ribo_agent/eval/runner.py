"""End-to-end eval runner.

Usage:
  python -m ribo_agent.eval.run_eval --config configs/v0_zeroshot_qwen25_7b.yaml
  python -m ribo_agent.eval.run_eval --config configs/...yaml --limit 10   # smoke

Produces in results/runs/<timestamp>_<safe_model>/:
  predictions.jsonl   one Prediction per eval question
  metrics.json        full Metrics as JSON
  report.md           human-readable markdown summary
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

import yaml

from ..agents import ZeroShotAgent, RAGAgent
from ..agents.base import Prediction
from ..llm import make_client
from ..parsers.schema import MCQ
from .metrics import compute_metrics, format_report
from .detailed_report import write_detailed_report


ROOT = Path(__file__).resolve().parents[3]
EVAL_PATH = ROOT / "data" / "parsed" / "eval.jsonl"
RESULTS = ROOT / "results" / "runs"


def _load_eval() -> list[MCQ]:
    if not EVAL_PATH.exists():
        raise FileNotFoundError(
            f"{EVAL_PATH} missing. Run `make parse` first."
        )
    out: list[MCQ] = []
    for line in EVAL_PATH.open():
        d = json.loads(line)
        out.append(MCQ(**d))
    return out


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _safe_name(model: str) -> str:
    # "qwen2.5:7b-instruct" -> "qwen2.5_7b-instruct"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_")


def run_eval(
    config: dict,
    *,
    limit: int | None = None,
    progress: bool = True,
) -> dict:
    llm_cfg = config["llm"]
    gen_cfg = config.get("generation", {})
    agent_cfg = config.get("agent", {})
    llm = make_client(llm_cfg)

    agent_type = agent_cfg.get("type", "zeroshot")

    if agent_type == "rag":
        ret_cfg = config.get("retrieval", {})
        chunks_path = Path(ret_cfg.get("chunks_path", "data/kb/chunks.jsonl"))
        if not chunks_path.is_absolute():
            chunks_path = ROOT / chunks_path
        embedding_model = ret_cfg.get("embedding_model", "BAAI/bge-small-en-v1.5")
        retrieval_mode = ret_cfg.get("mode", "dense")  # dense | hybrid

        if retrieval_mode == "hybrid":
            from ..kb.hybrid_retriever import HybridRetriever
            from ..kb.chunker import Chunk
            import json as _json
            _chunks = [Chunk(**_json.loads(l)) for l in chunks_path.open()]
            retriever = HybridRetriever(_chunks, embedding_model=embedding_model)
        else:
            from ..kb.retriever import Retriever
            retriever = Retriever.from_chunks_jsonl(
                chunks_path, model_name=embedding_model,
            )
        agent = RAGAgent(
            llm,
            retriever,
            top_k=agent_cfg.get("top_k", 5),
            temperature=gen_cfg.get("temperature", 0.0),
            max_tokens=gen_cfg.get("max_tokens", 512),
        )
    else:
        agent = ZeroShotAgent(
            llm,
            temperature=gen_cfg.get("temperature", 0.0),
            max_tokens=gen_cfg.get("max_tokens", 256),
        )

    mcqs = _load_eval()
    if limit:
        mcqs = mcqs[:limit]

    preds: list[Prediction] = []
    t0 = time.perf_counter()
    for i, mcq in enumerate(mcqs, 1):
        pred = agent.answer(mcq)
        preds.append(pred)
        if progress:
            n_correct = sum(1 for p in preds if p.is_correct)
            elapsed = time.perf_counter() - t0
            eta = elapsed / i * (len(mcqs) - i)
            print(
                f"\r[{i:>3}/{len(mcqs)}]  acc={n_correct / i:.3f}  "
                f"elapsed={elapsed:6.1f}s  eta={eta:6.1f}s  ",
                end="",
                file=sys.stderr,
                flush=True,
            )
    if progress:
        print("", file=sys.stderr)

    metrics = compute_metrics(preds, mcqs=mcqs)

    # write artifacts
    ts = time.strftime("%Y%m%d-%H%M%S")
    model_safe = _safe_name(llm_cfg.get("model", "unknown"))
    run_dir = RESULTS / f"{ts}_{config.get('name', 'agent')}_{model_safe}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "predictions.jsonl").open("w") as f:
        for p in preds:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
    (run_dir / "metrics.json").write_text(json.dumps(metrics.to_dict(), indent=2))

    title = (
        f"{config.get('name', 'run')} — {llm_cfg.get('model', '?')} "
        f"({llm_cfg.get('backend', '?')})"
    )
    (run_dir / "report.md").write_text(format_report(metrics, title=title))

    # Detailed per-question report with citations + faithfulness
    write_detailed_report(
        preds,
        run_dir / "detailed_report.md",
        title=f"Detailed: {title}",
    )

    # Profiling summary
    from .profiler import CostTracker
    tracker = CostTracker()
    for p in preds:
        extras = p.extras or {}
        tracker.record(
            qid=p.qid,
            retrieval_ms=extras.get("retrieval_ms", 0),
            generation_ms=extras.get("generation_ms", 0),
            prompt_tokens=p.prompt_tokens or 0,
            completion_tokens=p.completion_tokens or 0,
        )
    (run_dir / "profiling.md").write_text(tracker.format_summary())

    print(f"\nwrote {run_dir.relative_to(ROOT)}/")
    print(f"  accuracy  = {metrics.accuracy:.4f}")
    print(f"  macro-F1  = {metrics.macro_f1:.4f}")
    print(f"  n         = {metrics.n}")
    if metrics.latency_ms_mean:
        print(f"  latency   = {metrics.latency_ms_mean:.0f} ms mean "
              f"(p50 {metrics.latency_ms_p50:.0f}, p90 {metrics.latency_ms_p90:.0f})")
    return metrics.to_dict()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only evaluate the first N questions (smoke test).")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    config = _load_config(args.config)
    run_eval(config, limit=args.limit, progress=not args.no_progress)


if __name__ == "__main__":
    main()

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

from ..agents import ZeroShotAgent
from ..agents.base import Prediction
from ..llm import make_client
from ..parsers.schema import MCQ
from .metrics import compute_metrics, format_report


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


def _make_agent(config: dict, llm):
    """Build the agent specified by config['agent'] (default: zeroshot)."""
    agent_type = config.get("agent", "zeroshot")
    gen_cfg = config.get("generation", {})
    temp = gen_cfg.get("temperature", 0.0)
    max_tok = gen_cfg.get("max_tokens", 256)

    if agent_type == "zeroshot":
        return ZeroShotAgent(llm, temperature=temp, max_tokens=max_tok)

    if agent_type == "wiki":
        from ..agents.wiki_agent import WikiAgent
        return WikiAgent(
            llm, temperature=temp, max_tokens=max_tok,
            wiki_max_tokens=gen_cfg.get("wiki_max_tokens", 4096),
        )

    if agent_type == "rewrite":
        from ..agents.wiki_agent import WikiAgent
        from ..agents.rewrite_agent import RewriteAgent
        wiki = WikiAgent(llm, temperature=temp, max_tokens=max_tok)
        return RewriteAgent(llm, wiki_agent=wiki, temperature=temp, max_tokens=max_tok)

    if agent_type == "ensemble":
        from ..agents.ensemble_agent import EnsembleAgent
        return EnsembleAgent(
            llm, temperature=temp, max_tokens=max_tok,
            wiki_max_tokens=gen_cfg.get("wiki_max_tokens", 4096),
            sc_samples=gen_cfg.get("sc_samples", 5),
            sc_temperature=gen_cfg.get("sc_temperature", 0.7),
        )

    if agent_type == "multistep":
        from ..agents.multistep_agent import MultiStepAgent
        return MultiStepAgent(
            llm, temperature=temp, max_tokens=max_tok,
            top_k_retrieve=gen_cfg.get("top_k_retrieve", 5),
            similarity_threshold=gen_cfg.get("similarity_threshold", 0.70),
            wiki_max_tokens=gen_cfg.get("wiki_max_tokens", 4096),
            enable_voting=gen_cfg.get("enable_voting", False),
            vote_samples=gen_cfg.get("vote_samples", 3),
        )

    raise ValueError(f"unknown agent type: {agent_type!r}")


def run_eval(
    config: dict,
    *,
    limit: int | None = None,
    progress: bool = True,
) -> dict:
    llm_cfg = config["llm"]
    gen_cfg = config.get("generation", {})
    llm = make_client(llm_cfg)

    agent = _make_agent(config, llm)

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

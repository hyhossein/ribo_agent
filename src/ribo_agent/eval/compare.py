"""Compare every run under results/runs/ in one table.

Usage:
  python -m ribo_agent.eval.compare              # plain text table
  python -m ribo_agent.eval.compare --markdown   # for LEADERBOARD.md
  python -m ribo_agent.eval.compare --readme     # for the README section
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "results" / "runs"


def _collect() -> list[dict]:
    rows: list[dict] = []
    if not RUNS.exists():
        return rows
    for run_dir in sorted(RUNS.iterdir()):
        mpath = run_dir / "metrics.json"
        if not mpath.exists():
            continue
        m = json.loads(mpath.read_text())
        # dir format: "<ts>_<agent_type>_<model_safe>"
        # examples:
        #   20260423-195907_v0_zeroshot_claude-opus-4-20250514
        #   20260424-063248_v2_rewrite_wiki_claude-opus-4-20250514
        #   20260423-182402_v0_zeroshot_phi4-mini
        name = run_dir.name
        parts = name.split("_")
        ts_part = parts[0]

        # Extract agent prefix and model slug from the dir name
        # Agent prefix: v0_zeroshot, v1_wiki, v2_rewrite_wiki, v3_ensemble
        # Model slug: everything after the agent prefix
        suffix = "_".join(parts[1:])  # e.g. "v0_zeroshot_claude-opus-4-20250514"
        agent_type, model_slug = _parse_agent_and_model(suffix)

        display = _pretty_display(agent_type, model_slug)

        rows.append({
            "run": name,
            "timestamp": ts_part,
            "model": display,
            "model_raw": model_slug,
            "agent_type": agent_type,
            "dedup_key": f"{agent_type}_{model_slug}",
            "n": m.get("n"),
            "accuracy": m.get("accuracy"),
            "macro_f1": m.get("macro_f1"),
            "refusal": m.get("refusal_rate"),
            "lat_mean_ms": m.get("latency_ms_mean"),
            "per_domain": m.get("per_domain", {}),
        })
    return rows


_AGENT_PREFIXES = [
    "v3_ensemble",
    "v2_rewrite_wiki",
    "v1_wiki",
    "v0_zeroshot",
]


def _parse_agent_and_model(suffix: str) -> tuple[str, str]:
    """Split 'v2_rewrite_wiki_claude-opus-4-20250514' into
    ('v2_rewrite_wiki', 'claude-opus-4-20250514')."""
    for prefix in _AGENT_PREFIXES:
        if suffix.startswith(prefix + "_"):
            model = suffix[len(prefix) + 1:]
            return prefix, model
        if suffix == prefix:
            return prefix, "unknown"
    # Fallback: assume v0_zeroshot and take everything after first two tokens
    parts = suffix.split("_", 2)
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}", parts[2]
    return "v0_zeroshot", suffix


_MODEL_PRETTY = {
    "qwen2.5_7b-instruct": "Qwen 2.5 7B",
    "qwen3_8b": "Qwen 3 8B",
    "llama3.1_8b": "Llama 3.1 8B",
    "llama3.1_8b-instruct": "Llama 3.1 8B",
    "phi3.5_3.8b": "Phi 3.5 3.8B",
    "phi4-mini": "Phi-4 Mini 3.8B",
    "gemma3_12b": "Gemma 3 12B",
    "deepseek-r1_7b": "DeepSeek-R1 7B",
    "mistral_7b-instruct": "Mistral 7B",
    "claude-opus-4-20250514": "Opus 4",
    "claude-sonnet-4-20250514": "Sonnet 4",
}

_AGENT_PRETTY = {
    "v0_zeroshot": "",
    "v1_wiki": "Wiki +",
    "v2_rewrite_wiki": "Rewrite+Wiki +",
    "v3_ensemble": "Ensemble +",
}


def _pretty_model(slug: str) -> str:
    return _MODEL_PRETTY.get(slug, slug)


def _pretty_display(agent_type: str, model_slug: str) -> str:
    """Combine agent label and model name into a readable display string."""
    model = _MODEL_PRETTY.get(model_slug, model_slug)
    agent = _AGENT_PRETTY.get(agent_type, agent_type)
    if agent:
        return f"{agent} {model}"
    return model


def _latest_per_model(rows: list[dict]) -> list[dict]:
    """Deduplicate: keep only the most recent run per agent+model combo."""
    by_key: dict[str, dict] = {}
    for r in rows:
        key = r.get("dedup_key", r["model_raw"])
        existing = by_key.get(key)
        if existing is None or r["timestamp"] > existing["timestamp"]:
            by_key[key] = r
    return sorted(
        by_key.values(), key=lambda r: (r["accuracy"] or 0), reverse=True,
    )


def _format_plain(rows: list[dict]) -> str:
    if not rows:
        return "(no runs yet — run `make eval CONFIG=…` to get started)"
    headers = ["#", "model", "n", "accuracy", "macro_f1", "refusal", "lat_ms"]
    out_rows = []
    for i, r in enumerate(rows, 1):
        lat = f"{r['lat_mean_ms']:.0f}" if r["lat_mean_ms"] else "-"
        out_rows.append([
            str(i),
            r["model"][:30],
            str(r["n"]),
            f"{r['accuracy']:.4f}" if r["accuracy"] is not None else "-",
            f"{r['macro_f1']:.4f}" if r["macro_f1"] is not None else "-",
            f"{r['refusal']:.3f}" if r["refusal"] is not None else "-",
            lat,
        ])
    widths = [max(len(h), max(len(r[i]) for r in out_rows)) for i, h in enumerate(headers)]
    sep = "  "
    lines = [sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    lines.append(sep.join("-" * widths[i] for i in range(len(headers))))
    for row in out_rows:
        lines.append(sep.join(row[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines) + "\n"


def _medal(rank: int) -> str:
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"{rank}.")


def _format_markdown(rows: list[dict], *, compact: bool = False) -> str:
    """Markdown table with medals for the top three."""
    if not rows:
        return (
            "_No evaluation runs yet. Run `make sweep` to populate this "
            "leaderboard with real numbers._\n"
        )
    if compact:
        headers = ["", "Model", "Accuracy", "Macro-F1", "Latency (ms)"]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| :--- | :--- | ---: | ---: | ---: |",
        ]
        for i, r in enumerate(rows, 1):
            lat = f"{r['lat_mean_ms']:.0f}" if r["lat_mean_ms"] else "-"
            lines.append(
                f"| {_medal(i)} | **{r['model']}** | "
                f"`{r['accuracy']:.4f}` | `{r['macro_f1']:.4f}` | {lat} |"
            )
    else:
        headers = ["", "Model", "n", "Accuracy", "Macro-F1", "Refusal", "Latency mean (ms)"]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for i, r in enumerate(rows, 1):
            lat = f"{r['lat_mean_ms']:.0f}" if r["lat_mean_ms"] else "-"
            lines.append(
                f"| {_medal(i)} | **{r['model']}** | {r['n']} | "
                f"`{r['accuracy']:.4f}` | `{r['macro_f1']:.4f}` | "
                f"{r['refusal']:.3f} | {lat} |"
            )
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines += ["", f"_Updated {ts} · 169-question eval set · open-source + commercial models_"]
    return "\n".join(lines) + "\n"


def _format_readme_block(rows: list[dict]) -> str:
    """The block we inline into README.md between leaderboard markers."""
    if not rows:
        body = (
            "_No evaluation runs yet. Run `make sweep` to populate this "
            "leaderboard with real numbers._"
        )
    else:
        body = _format_markdown(rows, compact=True).rstrip()
    return body


README_START = "<!-- LEADERBOARD:START -->"
README_END = "<!-- LEADERBOARD:END -->"


def _update_readme(block: str) -> bool:
    """Splice the block into README.md between the two markers.
    Returns True if README was modified."""
    readme = ROOT / "README.md"
    text = readme.read_text()
    if README_START not in text or README_END not in text:
        return False
    pattern = re.compile(
        re.escape(README_START) + r".*?" + re.escape(README_END),
        re.DOTALL,
    )
    new_text = pattern.sub(
        f"{README_START}\n{block}\n{README_END}", text
    )
    if new_text == text:
        return False
    readme.write_text(new_text)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true",
                    help="Emit markdown for LEADERBOARD.md")
    ap.add_argument("--readme", action="store_true",
                    help="Splice a compact table into README.md between "
                         "the <!-- LEADERBOARD:START --> markers.")
    ap.add_argument("--all-runs", action="store_true",
                    help="Show every run, not just the latest per model.")
    args = ap.parse_args()

    rows = _collect()
    if not args.all_runs:
        rows = _latest_per_model(rows)
    else:
        rows.sort(key=lambda r: (r["accuracy"] or 0), reverse=True)

    if args.readme:
        block = _format_readme_block(rows)
        changed = _update_readme(block)
        print(
            "README.md updated" if changed
            else "README.md unchanged (no markers or block already current)"
        )
        return

    if args.markdown:
        print(_format_markdown(rows, compact=False))
    else:
        print(_format_plain(rows))


if __name__ == "__main__":
    main()

"""Compare every run under results/runs/ in one table.

Usage:
  python -m ribo_agent.eval.compare              # stdout table
  python -m ribo_agent.eval.compare --markdown > results/leaderboard.md
"""
from __future__ import annotations

import argparse
import json
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
        # parse the run dir name: "<ts>_<config_name>_<model_safe>"
        parts = run_dir.name.split("_", 2)
        run_id = run_dir.name
        rows.append({
            "run": run_id,
            "n": m.get("n"),
            "accuracy": m.get("accuracy"),
            "macro_f1": m.get("macro_f1"),
            "refusal": m.get("refusal_rate"),
            "lat_mean_ms": m.get("latency_ms_mean"),
            "per_domain": m.get("per_domain", {}),
        })
    return rows


def _format_table(rows: list[dict], *, markdown: bool = False) -> str:
    if not rows:
        return "(no runs yet)"
    headers = ["run", "n", "accuracy", "macro_f1", "refusal", "lat_ms"]
    out_rows = []
    for r in rows:
        lat = f"{r['lat_mean_ms']:.0f}" if r["lat_mean_ms"] else "-"
        out_rows.append([
            r["run"][:60],
            str(r["n"]),
            f"{r['accuracy']:.4f}" if r["accuracy"] is not None else "-",
            f"{r['macro_f1']:.4f}" if r["macro_f1"] is not None else "-",
            f"{r['refusal']:.3f}" if r["refusal"] is not None else "-",
            lat,
        ])
    if markdown:
        lines = ["| " + " | ".join(headers) + " |",
                 "| " + " | ".join(["---"] * len(headers)) + " |"]
        for row in out_rows:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines) + "\n"
    # plain text
    widths = [max(len(h), max(len(r[i]) for r in out_rows)) for i, h in enumerate(headers)]
    sep = "  "
    lines = [sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    lines.append(sep.join("-" * widths[i] for i in range(len(headers))))
    for row in out_rows:
        lines.append(sep.join(row[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    rows = _collect()
    rows.sort(key=lambda r: (r["accuracy"] or 0), reverse=True)
    print(_format_table(rows, markdown=args.markdown))


if __name__ == "__main__":
    main()

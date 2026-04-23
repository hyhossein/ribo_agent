"""Tests for the leaderboard compare module.

Uses a temp directory so we don't touch the real results/runs/.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ribo_agent.eval import compare


def _write_run(runs_dir: Path, name: str, metrics: dict) -> None:
    d = runs_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(metrics))


@pytest.fixture
def runs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "runs"
    d.mkdir()
    monkeypatch.setattr(compare, "RUNS", d)
    return d


def _fake(n: int = 169, accuracy: float = 0.5, f1: float = 0.5) -> dict:
    return {
        "n": n,
        "accuracy": accuracy,
        "macro_f1": f1,
        "micro_f1": accuracy,
        "refusal_rate": 0.0,
        "latency_ms_mean": 5000.0,
        "latency_ms_p50": 4800.0,
        "latency_ms_p90": 6100.0,
        "per_class": {},
        "per_domain": {},
        "per_cognitive": {},
        "confusion": {},
    }


def test_empty_runs_returns_friendly_message(runs_dir: Path) -> None:
    out = compare._format_markdown([])
    assert "No evaluation runs yet" in out


def test_latest_per_model_dedupes_older_runs(runs_dir: Path) -> None:
    _write_run(runs_dir, "20260101-120000_v0_zeroshot_qwen2.5_7b-instruct",
               _fake(accuracy=0.50))
    _write_run(runs_dir, "20260102-120000_v0_zeroshot_qwen2.5_7b-instruct",
               _fake(accuracy=0.65))
    rows = compare._collect()
    assert len(rows) == 2
    # Latest-per-model should keep only the newest
    latest = compare._latest_per_model(rows)
    assert len(latest) == 1
    assert latest[0]["accuracy"] == 0.65


def test_leaderboard_sorts_by_accuracy_desc(runs_dir: Path) -> None:
    _write_run(runs_dir, "20260101-120000_v0_zeroshot_phi3.5_3.8b",
               _fake(accuracy=0.42))
    _write_run(runs_dir, "20260102-120000_v0_zeroshot_qwen2.5_7b-instruct",
               _fake(accuracy=0.65))
    _write_run(runs_dir, "20260103-120000_v0_zeroshot_llama3.1_8b",
               _fake(accuracy=0.60))
    rows = compare._latest_per_model(compare._collect())
    assert [r["model_raw"] for r in rows] == [
        "qwen2.5_7b-instruct", "llama3.1_8b", "phi3.5_3.8b",
    ]


def test_markdown_emits_medals(runs_dir: Path) -> None:
    _write_run(runs_dir, "20260101-120000_v0_zeroshot_qwen2.5_7b-instruct",
               _fake(accuracy=0.65))
    _write_run(runs_dir, "20260102-120000_v0_zeroshot_llama3.1_8b",
               _fake(accuracy=0.55))
    rows = compare._latest_per_model(compare._collect())
    md = compare._format_markdown(rows)
    assert "🥇" in md
    assert "🥈" in md
    # Model names should be prettified, not raw slugs
    assert "Qwen 2.5 7B Instruct" in md
    assert "Llama 3.1 8B" in md


def test_readme_splice_replaces_block(runs_dir: Path, tmp_path: Path,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    # Point _update_readme at a temp README with the markers.
    fake_readme = tmp_path / "README.md"
    fake_readme.write_text(
        "# Proj\n\nintro\n\n"
        "<!-- LEADERBOARD:START -->\nold content\n<!-- LEADERBOARD:END -->\n\n"
        "rest\n"
    )
    monkeypatch.setattr(compare, "ROOT", tmp_path)
    _write_run(runs_dir, "20260101-120000_v0_zeroshot_qwen2.5_7b-instruct",
               _fake(accuracy=0.65))
    rows = compare._latest_per_model(compare._collect())
    block = compare._format_readme_block(rows)
    changed = compare._update_readme(block)
    assert changed
    new = fake_readme.read_text()
    assert "old content" not in new
    assert "Qwen 2.5 7B Instruct" in new
    # preserve surrounding content
    assert "intro" in new
    assert "rest" in new


def test_readme_splice_noop_when_no_markers(tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    fake_readme = tmp_path / "README.md"
    fake_readme.write_text("# Proj\n\nno markers here\n")
    monkeypatch.setattr(compare, "ROOT", tmp_path)
    changed = compare._update_readme("anything")
    assert changed is False

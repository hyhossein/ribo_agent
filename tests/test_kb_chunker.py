"""Tests for the knowledge-base chunker.

Uses the real raw/study corpus. The chunker is deterministic on a fixed
input, so we can assert on specific counts and invariants.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ribo_agent.kb import chunker


@pytest.fixture(scope="session")
def raw_study_dir(repo_root: Path) -> Path:
    p = repo_root / "data" / "raw" / "study"
    if not p.exists():
        pytest.skip(f"raw study dir missing: {p}")
    return p


@pytest.fixture(scope="module")
def chunks(raw_study_dir: Path, repo_root: Path):
    cache = repo_root / "data" / "interim" / "study_txt"
    return chunker.chunk_corpus(raw_study_dir, cache_dir=cache)


def test_all_eight_sources_present(chunks) -> None:
    # RIB Act + Reg 989/990/991 + By-Laws 1/2/3 + OAP = 8
    sources = {c.source for c in chunks}
    assert len(sources) == 8


def test_every_chunk_has_citation(chunks) -> None:
    for c in chunks:
        assert c.citation, c.chunk_id


def test_chunk_sizes_are_within_bounds(chunks) -> None:
    # Size normaliser enforces MIN_CHARS floor and MAX_CHARS ceiling
    # (with small overshoot allowed when a single sentence is big).
    for c in chunks:
        assert len(c.text) >= 50, c.chunk_id
        assert len(c.text) <= chunker.MAX_CHARS + 200, c.chunk_id


def test_reg_991_section_numbers_parse(chunks) -> None:
    # Regulation 991 is the meat of the exam; confirm we parsed it into
    # many sections rather than one big blob.
    reg991 = [c for c in chunks if c.source == "Ontario_Regulation_991"]
    assert len(reg991) >= 20


def test_rib_act_sections_have_titles(chunks) -> None:
    # The Act splitter promotes the previous short line to title; we
    # should see titles on at least half the Act chunks.
    act = [c for c in chunks if c.source == "RIB_Act_1990"]
    titled = [c for c in act if c.title]
    assert len(titled) >= len(act) // 2


def test_chunk_ids_unique(chunks) -> None:
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_bylaws_preserve_article_in_extras(chunks) -> None:
    bylaw = [c for c in chunks if c.source.startswith("RIBO_By-Law_1")]
    with_article = [c for c in bylaw if c.extras.get("article")]
    assert with_article  # at least some chunks have the article label

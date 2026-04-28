"""Shared pytest fixtures."""
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def raw_questions_dir(repo_root: Path) -> Path:
    p = repo_root / "data" / "raw" / "questions"
    if not p.exists():
        pytest.skip(f"raw questions dir missing: {p}")
    return p


@pytest.fixture(scope="session")
def sample_pdf(raw_questions_dir: Path) -> Path:
    p = raw_questions_dir / "Sample-Questions-RIBO-Level-1-Exam (1).pdf"
    if not p.exists():
        pytest.skip(f"sample pdf missing: {p}")
    return p

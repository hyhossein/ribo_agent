"""Tests for the storage abstraction."""
from __future__ import annotations

import pytest

from ribo_agent.io import LocalStorage, make_storage


def test_local_storage_roundtrip(tmp_path) -> None:
    s = LocalStorage(tmp_path)
    s.write_bytes("parsed/eval.jsonl", b'{"qid":"q1"}\n')
    assert s.exists("parsed/eval.jsonl")
    assert s.read_bytes("parsed/eval.jsonl") == b'{"qid":"q1"}\n'


def test_local_storage_lists_prefix(tmp_path) -> None:
    s = LocalStorage(tmp_path)
    s.write_bytes("parsed/a.jsonl", b"a")
    s.write_bytes("parsed/b.jsonl", b"b")
    s.write_bytes("kb/c.jsonl", b"c")
    parsed = s.list("parsed")
    assert set(parsed) == {"parsed/a.jsonl", "parsed/b.jsonl"}


def test_local_storage_rejects_path_escape(tmp_path) -> None:
    s = LocalStorage(tmp_path)
    with pytest.raises(ValueError, match="escapes storage root"):
        s.write_bytes("../sneaky.txt", b"nope")


def test_make_storage_defaults_to_local(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("RIBO_STORAGE", raising=False)
    s = make_storage(root=tmp_path)
    assert isinstance(s, LocalStorage)


def test_make_storage_honours_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RIBO_STORAGE", "local")
    s = make_storage(root=tmp_path)
    assert isinstance(s, LocalStorage)


def test_make_storage_rejects_unknown(monkeypatch) -> None:
    monkeypatch.setenv("RIBO_STORAGE", "mystery")
    with pytest.raises(ValueError, match="unknown RIBO_STORAGE"):
        make_storage()

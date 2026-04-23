"""Storage abstraction.

Code that reads or writes derived artifacts (parsed JSONL, KB chunks,
indices, eval results) should go through a `Storage` instance instead
of calling `open()` directly. That way Azure Blob storage is a one-line
config switch later, not a refactor.

Selection is via the `RIBO_STORAGE` env var, defaulting to `local`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol


class Storage(Protocol):
    """Minimal interface — read/write bytes, list keys under a prefix."""

    def read_bytes(self, key: str) -> bytes: ...
    def write_bytes(self, key: str, data: bytes) -> None: ...
    def list(self, prefix: str) -> list[str]: ...
    def exists(self, key: str) -> bool: ...


class LocalStorage:
    """Storage backed by a root directory on the local filesystem."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()

    def _p(self, key: str) -> Path:
        p = (self.root / key).resolve()
        # protect against path escape via '../'
        if not str(p).startswith(str(self.root)):
            raise ValueError(f"key escapes storage root: {key!r}")
        return p

    def read_bytes(self, key: str) -> bytes:
        return self._p(key).read_bytes()

    def write_bytes(self, key: str, data: bytes) -> None:
        p = self._p(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def list(self, prefix: str) -> list[str]:
        base = self._p(prefix)
        if not base.exists():
            return []
        if base.is_file():
            return [str(base.relative_to(self.root))]
        return [
            str(p.relative_to(self.root))
            for p in sorted(base.rglob("*"))
            if p.is_file()
        ]

    def exists(self, key: str) -> bool:
        return self._p(key).exists()


class AzureBlobStorage:
    """Azure Blob storage — stub.

    Ships in a later release along with the Azure ML LLM deployment.
    Keeping the class here reserves the name and signals the migration
    path to reviewers.
    """

    def __init__(self, account_url: str, container: str, credential=None) -> None:
        raise NotImplementedError(
            "AzureBlobStorage lands with the v0.x Azure ML promotion. "
            "Use LocalStorage for now (default)."
        )


def make_storage(root: Path | str | None = None) -> Storage:
    """Return the storage backend selected by the `RIBO_STORAGE` env var.

    `local` (default) backs onto `root` (or CWD if not given).
    `azureml` will back onto Azure Blob once implemented.
    """
    backend = os.environ.get("RIBO_STORAGE", "local").lower()
    if backend == "local":
        return LocalStorage(root or Path.cwd())
    if backend == "azureml":
        return AzureBlobStorage(
            account_url=os.environ["AZUREML_BLOB_ACCOUNT_URL"],
            container=os.environ.get("AZUREML_BLOB_CONTAINER", "ribo-agent"),
        )
    raise ValueError(f"unknown RIBO_STORAGE backend: {backend!r}")

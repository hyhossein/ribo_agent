"""Turn raw study documents into plain UTF-8 text.

Supports two formats:
  .doc   -> LibreOffice headless `--convert-to txt`, output as UTF-8
  .pdf   -> `pdftotext -layout`, preserves spatial layout important for
           recognising section headers

Cached output lives in data/interim/study_txt/ keyed by the source
filename. Rebuilds are idempotent: if the source hasn't changed since
the cached copy, we reuse it.

LibreOffice binary discovery:
  - Linux (Homebrew or apt): `libreoffice` is on PATH
  - macOS (Homebrew cask):  the binary is `soffice`, typically at
      /opt/homebrew/bin/soffice (Apple Silicon) or
      /Applications/LibreOffice.app/Contents/MacOS/soffice
  - We probe candidates in order and use the first that works.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


_SOFFICE_CANDIDATES = [
    "libreoffice",                                                     # linux
    "soffice",                                                         # macos homebrew symlink
    "/opt/homebrew/bin/soffice",                                       # apple silicon brew
    "/usr/local/bin/soffice",                                          # intel mac brew
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",            # mac .app bundle
]


def _find_soffice() -> str:
    for name in _SOFFICE_CANDIDATES:
        path = shutil.which(name) if "/" not in name else (name if Path(name).exists() else None)
        if path:
            return path
    raise RuntimeError(
        "LibreOffice not found. On macOS: `brew install --cask libreoffice`. "
        "On Ubuntu: `sudo apt-get install -y libreoffice`."
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _cache_path(cache_dir: Path, src: Path) -> Path:
    digest = _sha256(src)
    safe = src.stem.replace(" ", "_")
    return cache_dir / f"{safe}.{digest}.txt"


def doc_to_text(src: Path, cache_dir: Path | None = None) -> str:
    """Convert a .doc file to UTF-8 text via LibreOffice headless."""
    if cache_dir is not None:
        cached = _cache_path(cache_dir, src)
        if cached.exists():
            return cached.read_text(encoding="utf-8")

    soffice = _find_soffice()
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [
                soffice,
                "--headless",
                "--convert-to",
                "txt:Text (encoded):UTF8",
                "--outdir",
                tmp,
                str(src),
            ],
            check=True,
            capture_output=True,
            env={**os.environ, "HOME": tmp},  # isolate profile
        )
        produced = Path(tmp) / f"{src.stem}.txt"
        if not produced.exists():
            raise RuntimeError(f"libreoffice did not produce {produced}")
        text = produced.read_text(encoding="utf-8", errors="replace")

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        _cache_path(cache_dir, src).write_text(text, encoding="utf-8")
    return text


def pdf_to_text(src: Path, cache_dir: Path | None = None) -> str:
    """Convert a PDF to text via `pdftotext -layout`."""
    if cache_dir is not None:
        cached = _cache_path(cache_dir, src)
        if cached.exists():
            return cached.read_text(encoding="utf-8")

    r = subprocess.run(
        ["pdftotext", "-layout", str(src), "-"],
        capture_output=True, check=True,
    )
    text = r.stdout.decode("utf-8", errors="replace")

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        _cache_path(cache_dir, src).write_text(text, encoding="utf-8")
    return text

"""Knowledge-base construction: ingest study docs and chunk by section."""

from .ingest import doc_to_text, pdf_to_text
from .chunker import chunk_corpus, Chunk

__all__ = ["Chunk", "chunk_corpus", "doc_to_text", "pdf_to_text"]

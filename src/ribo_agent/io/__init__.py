"""Pluggable storage backends."""

from .storage import Storage, LocalStorage, make_storage

__all__ = ["Storage", "LocalStorage", "make_storage"]

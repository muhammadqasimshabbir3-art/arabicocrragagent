"""Embedding factory and compatibility helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from embeddings.base import Embedder
from embeddings.bge_m3 import BGEM3Embedder, model_dir


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Return the configured embedder (BGE-M3 by default)."""
    return BGEM3Embedder()


def embed_documents(texts: Sequence[str]) -> list[list[float]]:
    """Embed document / chunk passages."""
    return get_embedder().embed_documents(texts)


def embed_query(text: str) -> list[float]:
    """Embed a search query."""
    return get_embedder().embed_query(text)


def embed_text(text: str) -> list[float]:
    """Embed a single document chunk."""
    return get_embedder().embed_text(text)


# Back-compat constants for older imports.
MODEL_ID = "BAAI/bge-m3"
EMBEDDING_DIMENSIONS = 1024

__all__ = [
    "EMBEDDING_DIMENSIONS",
    "MODEL_ID",
    "embed_documents",
    "embed_query",
    "embed_text",
    "get_embedder",
    "model_dir",
]

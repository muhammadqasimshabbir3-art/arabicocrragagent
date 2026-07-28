"""Embedding model abstractions."""

from .base import Embedder
from .factory import embed_documents, embed_query, embed_text, get_embedder

__all__ = ["Embedder", "embed_documents", "embed_query", "embed_text", "get_embedder"]

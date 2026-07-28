"""Vector store abstractions."""

from .base import RetrievedChunk, VectorStore
from .factory import get_vector_store

__all__ = ["RetrievedChunk", "VectorStore", "get_vector_store"]

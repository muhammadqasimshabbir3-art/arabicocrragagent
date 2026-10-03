"""Configurable chunking strategies."""

from .base import TextChunk
from .factory import chunk_document, get_chunker

__all__ = ["TextChunk", "chunk_document", "get_chunker"]

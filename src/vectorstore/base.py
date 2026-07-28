"""Vector store interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

from chunking.base import TextChunk


@dataclass
class RetrievedChunk:
    """A chunk returned from similarity search."""

    chunk_id: str
    text: str
    score: float
    page_start: int = 0
    page_end: int = 0
    filename: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    """Abstract vector database backend."""

    name: str = "base"

    @abstractmethod
    def add_chunks(
        self,
        chunks: Sequence[TextChunk],
        embeddings: Sequence[Sequence[float]],
        *,
        filename: str,
    ) -> None:
        """Index chunks with precomputed embeddings."""

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Return top-k nearest chunks for a query embedding."""

    @abstractmethod
    def count(self) -> int:
        """Return number of indexed chunks."""

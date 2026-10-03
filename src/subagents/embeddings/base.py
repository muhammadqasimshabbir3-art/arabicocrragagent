"""Embedder interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class Embedder(ABC):
    """Abstract embedding backend."""

    model_id: str = ""
    dimensions: int = 0

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed document / chunk passages."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""

    def embed_text(self, text: str) -> list[float]:
        """Embed a single document chunk."""
        return self.embed_documents([text or ""])[0]

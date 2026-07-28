"""Chunk data model and base chunker interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ocr.base import OCRDocument


@dataclass(frozen=True)
class TextChunk:
    """A searchable text chunk with page/layout metadata."""

    chunk_id: str
    text: str
    page_start: int
    page_end: int
    metadata: dict[str, Any] = field(default_factory=dict)


class Chunker(ABC):
    """Abstract chunking strategy."""

    name: str = "base"

    @abstractmethod
    def chunk(self, document: OCRDocument) -> list[TextChunk]:
        """Split an OCR document into retrieval chunks."""

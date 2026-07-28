"""Chunker factory driven by configuration."""

from __future__ import annotations

from chunking.base import Chunker, TextChunk
from chunking.layout_aware import LayoutAwareChunker
from chunking.page_aware import PageAwareChunker
from chunking.recursive import RecursiveChunker
from chunking.semantic import SemanticChunker
from config.settings import get_settings
from ocr.base import OCRDocument


def get_chunker(strategy: str | None = None) -> Chunker:
    """Return a chunker for the configured strategy."""
    settings = get_settings()
    name = (strategy or settings.chunk_strategy or "page_aware").lower()
    size = settings.chunk_size
    overlap = settings.chunk_overlap

    if name == "recursive":
        return RecursiveChunker(chunk_size=size, overlap=overlap)
    if name == "layout_aware":
        return LayoutAwareChunker(chunk_size=size, overlap=overlap)
    if name == "semantic":
        return SemanticChunker(chunk_size=size, overlap=overlap)
    if name == "page_aware":
        return PageAwareChunker(chunk_size=size, overlap=overlap)
    raise ValueError(f"Unknown chunk strategy: {name}")


def chunk_document(document: OCRDocument, strategy: str | None = None) -> list[TextChunk]:
    """Chunk an OCR document using the selected strategy."""
    return get_chunker(strategy).chunk(document)

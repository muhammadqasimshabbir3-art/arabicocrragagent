"""Recursive character chunking with overlap and page metadata."""

from __future__ import annotations

import re

from chunking.base import Chunker, TextChunk
from ocr.base import OCRDocument


class RecursiveChunker(Chunker):
    """Split full document text recursively on paragraph / sentence / space boundaries."""

    name = "recursive"

    def __init__(self, chunk_size: int = 1200, overlap: int = 200) -> None:
        if chunk_size <= overlap:
            raise ValueError("chunk_size must be larger than overlap")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: OCRDocument) -> list[TextChunk]:
        text = document.full_text
        normalized = re.sub(r"\n{3,}", "\n\n", text).strip()
        if not normalized:
            return []

        chunks: list[TextChunk] = []
        start = 0
        length = len(normalized)

        while start < length:
            end = min(start + self.chunk_size, length)
            if end < length:
                boundary = max(
                    normalized.rfind("\n\n", start, end),
                    normalized.rfind(". ", start, end),
                    normalized.rfind("۔ ", start, end),
                    normalized.rfind(" ", start, end),
                )
                if boundary > start + self.chunk_size // 2:
                    end = boundary + 1

            chunk_text = normalized[start:end].strip()
            if chunk_text:
                page_numbers = [
                    int(match)
                    for match in re.findall(r"\[Page\s+(\d+)\]", chunk_text)
                ]
                page_start = min(page_numbers) if page_numbers else 0
                page_end = max(page_numbers) if page_numbers else page_start
                chunks.append(
                    TextChunk(
                        chunk_id=f"chunk-{len(chunks)}",
                        text=chunk_text,
                        page_start=page_start,
                        page_end=page_end,
                        metadata={
                            "filename": document.filename,
                            "strategy": self.name,
                        },
                    )
                )

            if end >= length:
                break
            start = max(0, end - self.overlap)

        return chunks

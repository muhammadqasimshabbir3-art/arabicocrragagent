"""Layout-aware chunking using OCR HTML blocks when available."""

from __future__ import annotations

import re

from core.chunking.base import Chunker, TextChunk
from core.chunking.page_aware import PageAwareChunker
from subagents.ocr.base import OCRDocument
from core.retriever.hybrid import make_chunk_id

_BLOCK_RE = re.compile(
    r"(<(?:h[1-6]|p|li|tr|div|table)[^>]*>.*?</(?:h[1-6]|p|li|tr|div|table)>)",
    re.IGNORECASE | re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """Remove HTML tags while preserving plain text for indexing."""
    plain = _TAG_RE.sub(" ", text or "")
    return re.sub(r"\s+", " ", plain).strip()


class LayoutAwareChunker(Chunker):
    """Split on HTML layout blocks; fall back to page-aware chunking."""

    name = "layout_aware"

    def __init__(self, chunk_size: int = 1200, overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._fallback = PageAwareChunker(chunk_size=chunk_size, overlap=overlap)

    def chunk(self, document: OCRDocument) -> list[TextChunk]:
        has_html = any((page.html or "").strip() for page in document.pages)
        if not has_html:
            return self._fallback.chunk(document)

        chunks: list[TextChunk] = []
        ordinal = 0
        fingerprint = document.fingerprint or document.filename
        for page in document.pages:
            html = (page.html or "").strip()
            text = (page.text or "").strip()
            source = html or text
            if not source:
                continue

            blocks = [m.group(1).strip() for m in _BLOCK_RE.finditer(html)] if html else []
            if not blocks:
                blocks = [source]

            buffer = ""
            for block in blocks:
                candidate = f"{buffer}\n{block}".strip() if buffer else block
                if len(candidate) <= self.chunk_size:
                    buffer = candidate
                    continue
                if buffer:
                    plain = _strip_html(buffer)
                    labeled = f"[Page {page.page_number}]\n{plain}"
                    chunks.append(
                        TextChunk(
                            chunk_id=make_chunk_id(
                                fingerprint, page.page_number, ordinal, labeled
                            ),
                            text=labeled,
                            page_start=page.page_number,
                            page_end=page.page_number,
                            metadata={
                                "filename": document.filename,
                                "strategy": self.name,
                            },
                        )
                    )
                    ordinal += 1
                buffer = block

            if buffer:
                plain = _strip_html(buffer)
                labeled = f"[Page {page.page_number}]\n{plain}"
                chunks.append(
                    TextChunk(
                        chunk_id=make_chunk_id(
                            fingerprint, page.page_number, ordinal, labeled
                        ),
                        text=labeled,
                        page_start=page.page_number,
                        page_end=page.page_number,
                        metadata={
                            "filename": document.filename,
                            "strategy": self.name,
                        },
                    )
                )
                ordinal += 1

        return chunks or self._fallback.chunk(document)

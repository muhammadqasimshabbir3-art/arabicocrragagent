"""Page-aware chunking — keep page boundaries when possible."""

from __future__ import annotations

from chunking.base import Chunker, TextChunk
from chunking.recursive import RecursiveChunker
from ocr.base import OCRDocument


class PageAwareChunker(Chunker):
    """Chunk each page independently, then fall back to recursive within long pages."""

    name = "page_aware"

    def __init__(self, chunk_size: int = 1200, overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._recursive = RecursiveChunker(chunk_size=chunk_size, overlap=overlap)

    def chunk(self, document: OCRDocument) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        for page in document.pages:
            text = (page.text or "").strip()
            if not text:
                continue
            labeled = f"[Page {page.page_number}]\n{text}"
            if len(labeled) <= self.chunk_size:
                chunks.append(
                    TextChunk(
                        chunk_id=f"chunk-{len(chunks)}",
                        text=labeled,
                        page_start=page.page_number,
                        page_end=page.page_number,
                        metadata={
                            "filename": document.filename,
                            "strategy": self.name,
                            "has_html": bool(page.html),
                        },
                    )
                )
                continue

            # Build a one-page OCRDocument for recursive splitting.
            from ocr.base import OCRDocument as OD
            from ocr.base import OCRPage

            mini = OD(
                filename=document.filename,
                fingerprint=document.fingerprint,
                pages=[
                    OCRPage(
                        page_number=page.page_number,
                        text=text,
                        html=page.html,
                    )
                ],
                engine=document.engine,
            )
            for part in self._recursive.chunk(mini):
                chunks.append(
                    TextChunk(
                        chunk_id=f"chunk-{len(chunks)}",
                        text=part.text,
                        page_start=page.page_number,
                        page_end=page.page_number,
                        metadata={
                            "filename": document.filename,
                            "strategy": self.name,
                        },
                    )
                )
        return chunks

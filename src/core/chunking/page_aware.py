"""Page-aware chunking — keep page boundaries when possible."""

from __future__ import annotations

from core.chunking.base import Chunker, TextChunk
from core.chunking.recursive import RecursiveChunker
from subagents.ocr.base import OCRDocument
from core.retriever.hybrid import make_chunk_id


class PageAwareChunker(Chunker):
    """Chunk each page independently, then fall back to recursive within long pages."""

    name = "page_aware"

    def __init__(self, chunk_size: int = 1200, overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._recursive = RecursiveChunker(chunk_size=chunk_size, overlap=overlap)

    def chunk(self, document: OCRDocument) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        ordinal = 0
        fingerprint = document.fingerprint or document.filename
        for page in document.pages:
            text = (page.text or "").strip()
            if not text:
                continue
            labeled = f"[Page {page.page_number}]\n{text}"
            if len(labeled) <= self.chunk_size:
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
                            "has_html": bool(page.html),
                        },
                    )
                )
                ordinal += 1
                continue

            # Build a one-page OCRDocument for recursive splitting.
            from subagents.ocr.base import OCRDocument as OD
            from subagents.ocr.base import OCRPage

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
                        chunk_id=make_chunk_id(
                            fingerprint, page.page_number, ordinal, part.text
                        ),
                        text=part.text,
                        page_start=page.page_number,
                        page_end=page.page_number,
                        metadata={
                            "filename": document.filename,
                            "strategy": self.name,
                        },
                    )
                )
                ordinal += 1
        return chunks

"""Semantic chunking — group sentences by embedding similarity."""

from __future__ import annotations

import re

from chunking.base import Chunker, TextChunk
from chunking.page_aware import PageAwareChunker
from ocr.base import OCRDocument

_SENTENCE_SPLIT = re.compile(r"(?<=[\.!\?؟۔])\s+|\n{2,}")


def _split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(text or "") if part.strip()]
    return parts or ([text.strip()] if text.strip() else [])


class SemanticChunker(Chunker):
    """Merge adjacent sentences while cosine similarity stays above a threshold.

    Falls back to page-aware chunking when embeddings are unavailable or the
    page is short.
    """

    name = "semantic"

    def __init__(
        self,
        chunk_size: int = 1200,
        overlap: int = 200,
        similarity_threshold: float = 0.55,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.similarity_threshold = similarity_threshold
        self._fallback = PageAwareChunker(chunk_size=chunk_size, overlap=overlap)

    def chunk(self, document: OCRDocument) -> list[TextChunk]:
        try:
            from embeddings.factory import embed_documents
        except Exception:
            return self._fallback.chunk(document)

        chunks: list[TextChunk] = []
        for page in document.pages:
            text = (page.text or "").strip()
            if not text:
                continue
            sentences = _split_sentences(text)
            if len(sentences) <= 1 or len(text) <= self.chunk_size:
                chunks.append(
                    TextChunk(
                        chunk_id=f"chunk-{len(chunks)}",
                        text=f"[Page {page.page_number}]\n{text}",
                        page_start=page.page_number,
                        page_end=page.page_number,
                        metadata={
                            "filename": document.filename,
                            "strategy": self.name,
                        },
                    )
                )
                continue

            try:
                vectors = embed_documents(sentences)
            except Exception:
                return self._fallback.chunk(document)

            current: list[str] = [sentences[0]]
            current_len = len(sentences[0])

            def flush() -> None:
                nonlocal current, current_len
                if not current:
                    return
                body = " ".join(current).strip()
                chunks.append(
                    TextChunk(
                        chunk_id=f"chunk-{len(chunks)}",
                        text=f"[Page {page.page_number}]\n{body}",
                        page_start=page.page_number,
                        page_end=page.page_number,
                        metadata={
                            "filename": document.filename,
                            "strategy": self.name,
                        },
                    )
                )
                current = []
                current_len = 0

            for index in range(1, len(sentences)):
                prev_vec = vectors[index - 1]
                cur_vec = vectors[index]
                # Dot product works because BGE-M3 embeddings are normalized.
                similarity = sum(a * b for a, b in zip(prev_vec, cur_vec, strict=False))
                candidate_len = current_len + 1 + len(sentences[index])
                if (
                    similarity >= self.similarity_threshold
                    and candidate_len <= self.chunk_size
                ):
                    current.append(sentences[index])
                    current_len = candidate_len
                else:
                    flush()
                    current = [sentences[index]]
                    current_len = len(sentences[index])
            flush()

        return chunks or self._fallback.chunk(document)

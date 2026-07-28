"""Compatibility shim — document analysis now lives in retriever + document_qa."""

from agent.document_qa import document_analysis_response
from chunking.recursive import RecursiveChunker
from chunking.base import TextChunk as PDFChunk

# Historical names used by tests.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 5
EMBEDDING_DIMENSIONS = 1024


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split labeled PDF text into overlapping chunks (test/compat helper)."""
    from ocr.base import OCRDocument, OCRPage

    doc = OCRDocument(
        filename="compat.pdf",
        fingerprint="compat",
        pages=[OCRPage(page_number=1, text=text)],
        engine="compat",
    )
    # If text already has [Page N] markers, recursive chunker preserves them.
    return RecursiveChunker(chunk_size=chunk_size, overlap=overlap).chunk(doc)


async def pdf_analysis_response(
    question: str,
    pdf_data_base64: str,
    pdf_filename: str = "uploaded.pdf",
    summarize_only: bool = False,
):
    """Backward-compatible PDF analysis entrypoint."""
    return await document_analysis_response(
        question=question,
        data_base64=pdf_data_base64,
        filename=pdf_filename,
        summarize_only=summarize_only,
    )


__all__ = [
    "CHUNK_OVERLAP",
    "CHUNK_SIZE",
    "EMBEDDING_DIMENSIONS",
    "PDFChunk",
    "TOP_K",
    "pdf_analysis_response",
    "split_text_into_chunks",
]

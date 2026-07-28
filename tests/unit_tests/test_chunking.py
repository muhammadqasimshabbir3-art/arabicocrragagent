"""Unit tests for chunking and preprocess modules."""

from chunking.page_aware import PageAwareChunker
from chunking.recursive import RecursiveChunker
from ocr.base import OCRDocument, OCRPage
from preprocess.arabic_normalize import dedupe_repeated_lines


def _doc(*pages: str) -> OCRDocument:
    return OCRDocument(
        filename="t.pdf",
        fingerprint="abc",
        pages=[
            OCRPage(page_number=i + 1, text=text)
            for i, text in enumerate(pages)
        ],
        engine="digital_text",
    )


def test_page_aware_keeps_short_pages_separate():
    chunks = PageAwareChunker(chunk_size=2000, overlap=50).chunk(
        _doc("الصفحة الأولى", "الصفحة الثانية")
    )
    assert len(chunks) == 2
    assert chunks[0].page_start == 1
    assert chunks[1].page_start == 2


def test_recursive_chunker_splits_long_text():
    long_text = ("جملة عربية للاختبار. " * 100).strip()
    chunks = RecursiveChunker(chunk_size=200, overlap=40).chunk(_doc(long_text))
    assert len(chunks) > 1


def test_dedupe_repeated_lines():
    text = "عنوان\nعنوان\nمحتوى"
    assert dedupe_repeated_lines(text) == "عنوان\nمحتوى"

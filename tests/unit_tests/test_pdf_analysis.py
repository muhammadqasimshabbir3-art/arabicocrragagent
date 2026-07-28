"""Unit tests for PDF / document analysis helpers."""

from agent.pdf_analysis import split_text_into_chunks
from preprocess.arabic_normalize import clean_ocr_text, normalize_arabic


def test_split_text_into_overlapping_page_chunks():
    text = (
        "[Page 1]\n"
        + "Alpha beta gamma. " * 80
        + "\n\n[Page 2]\n"
        + "Delta epsilon zeta. " * 80
    )

    chunks = split_text_into_chunks(text, chunk_size=500, overlap=80)

    assert len(chunks) > 1
    assert "[Page 1]" in chunks[0].text
    assert all(chunk.text for chunk in chunks)


def test_normalize_arabic_alef_variants():
    assert normalize_arabic("أحمد") == "احمد"
    assert "ـ" not in normalize_arabic("كــتاب")


def test_clean_ocr_text_whitespace():
    cleaned = clean_ocr_text("مرحبا   بالعالم\n\n\nاختبار")
    assert "  " not in cleaned
    assert "\n\n\n" not in cleaned

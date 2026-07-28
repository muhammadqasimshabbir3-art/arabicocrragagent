"""Regression tests for Arabic normalization on OCR/PDF quirks."""

from preprocess.arabic_normalize import (
    clean_ocr_text,
    looks_unreliable_arabic_extraction,
    normalize_arabic,
)


def test_presentation_forms_normalize_to_standard_arabic():
    raw = "ﻋﺎﻡ ﺑﻴﺎﻧﺎﺕ ﻭﻣﻬﻨﺪﺱ ﺫﻛﺎﺀ ﺍﺻﻄﻨﺎﻋﻲ"
    normalized = normalize_arabic(raw, remove_diacritics=False, normalize_teh_marbuta=False)
    assert "عام بيانات" in normalized
    assert "ذكاء" in normalized
    assert "اصطناعي" in normalized


def test_spaced_arabic_letters_collapsed():
    raw = "ا ل ت ع ل ي م : ج ا م ع ة"
    normalized = clean_ocr_text(raw, for_retrieval=True)
    assert "التعليم" in normalized
    assert "جامعة" in normalized


def test_visual_order_reversed_tokens_are_repaired():
    raw = "دمحم مساق ريبش سدنهم ءاكذ يعانطصا"
    normalized = clean_ocr_text(raw, for_retrieval=True)
    assert "محمد قاسم شبير" in normalized
    assert "مهندس" in normalized
    assert "ذكاء اصطناعي" in normalized


def test_normal_text_is_not_reversed():
    raw = "محمد قاسم شبير مهندس ذكاء اصطناعي"
    normalized = clean_ocr_text(raw, for_retrieval=True)
    assert normalized.startswith("محمد قاسم شبير")


def test_generic_reversed_phrase_repaired_without_name_glossary():
    raw = "ليلحت تانايب ةجذمنو ةيؤبنت"
    normalized = clean_ocr_text(raw, for_retrieval=True)
    assert "تحليل بيانات" in normalized
    assert "نمذجة" in normalized


def test_unreliable_extraction_heuristic_flags_spaced_letters():
    raw = "ا ل ت ع ل ي م ا ل ج ا م ع ة ا ل ا س ل ا م ي ة"
    assert looks_unreliable_arabic_extraction(raw) is True


def test_unreliable_extraction_heuristic_passes_clean_arabic():
    raw = "التعليم في الجامعة الإسلامية جيد وواضح."
    assert looks_unreliable_arabic_extraction(raw) is False


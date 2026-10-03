"""Spelling Guardian — sure fixes only; unsure tokens left alone."""

from core.config.settings import get_settings
from subagents.spelling_corrector import correct_spelling


def test_dictionary_fixes_english_and_arabic() -> None:
    get_settings.cache_clear()
    result = correct_spelling("Who is Scheherazd in الف ليله?", use_llm=False)
    assert result.applied
    assert "scheherazade" in result.corrected.lower()
    assert "الف ليلة" in result.corrected or "ألف ليلة" in result.corrected


def test_unsure_token_left_unchanged() -> None:
    get_settings.cache_clear()
    # Made-up word not in sure dictionary — must stay as-is when LLM is off.
    raw = "Tell me about Zzzyx and قمرالزمان"
    result = correct_spelling(raw, use_llm=False)
    assert "Zzzyx" in result.corrected
    assert "قمر الزمان" in result.corrected


def test_disabled_leaves_text_untouched() -> None:
    get_settings.cache_clear()
    import os

    os.environ["ENABLE_SPELLING_CORRECTOR"] = "false"
    get_settings.cache_clear()
    try:
        result = correct_spelling("Scheherazd", use_llm=False)
        assert result.corrected == "Scheherazd"
        assert result.applied is False
    finally:
        os.environ.pop("ENABLE_SPELLING_CORRECTOR", None)
        get_settings.cache_clear()


def test_empty_input() -> None:
    result = correct_spelling("   ")
    assert result.applied is False
    assert result.corrected.strip() == ""

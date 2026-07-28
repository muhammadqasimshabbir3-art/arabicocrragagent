"""Tests for web-search helper utilities."""

from tools.web_search import (
    extract_links,
    screenshot_markdown,
    wants_profile_screenshot,
)


def test_wants_profile_screenshot_in_english_and_arabic():
    assert wants_profile_screenshot("Please include a screenshot of the profile")
    assert wants_profile_screenshot("اعرض لقطة شاشة للحساب")


def test_extract_links_and_screenshot_markdown():
    blob = "1) https://github.com/a\n2) https://kaggle.com/b"
    links = extract_links(blob)
    assert links == ["https://github.com/a", "https://kaggle.com/b"]
    md = screenshot_markdown(links, max_items=1)
    assert "image.thum.io" in md
    assert "github.com/a" in md


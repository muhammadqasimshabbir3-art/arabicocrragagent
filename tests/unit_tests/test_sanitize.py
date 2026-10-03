"""Unit tests for filename and collection-name sanitization."""

import pytest

from core.utils.sanitize import sanitize_filename, validate_collection_name


def test_sanitize_strips_path_traversal() -> None:
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("foo/../bar/secret.pdf") == "secret.pdf"
    assert sanitize_filename("C:\\Windows\\System32\\drivers") == "drivers"


def test_sanitize_rejects_dots_and_empty() -> None:
    assert sanitize_filename("..") == "uploaded.bin"
    assert sanitize_filename(".") == "uploaded.bin"
    assert sanitize_filename("") == "uploaded.bin"
    assert sanitize_filename(None) == "uploaded.bin"


def test_sanitize_strips_unsafe_chars() -> None:
    assert "/" not in sanitize_filename("a<b>:c|d?.pdf")
    assert "\\" not in sanitize_filename("x\\y.txt")


def test_validate_collection_name_ok() -> None:
    assert validate_collection_name("wathiqa_knowledge") == "wathiqa_knowledge"
    assert validate_collection_name("A1-b_2") == "A1-b_2"


def test_validate_collection_name_rejects_bad() -> None:
    with pytest.raises(ValueError):
        validate_collection_name("../evil")
    with pytest.raises(ValueError):
        validate_collection_name("")
    with pytest.raises(ValueError):
        validate_collection_name("_leading")
    with pytest.raises(ValueError):
        validate_collection_name("has space")

"""Unit tests for trust-layer relevant-index parsing."""

from subagents.document_qa import _parse_relevant_indexes


def test_parse_json_relevant_indexes() -> None:
    analysis = """
    {
      "sources": [
        {"index": 1, "verdict": "RELEVANT"},
        {"index": 2, "verdict": "NOT_RELEVANT"},
        {"index": 3, "verdict": "relevant"}
      ]
    }
    """
    assert _parse_relevant_indexes(analysis, 3) == {1, 3}


def test_parse_json_inside_fence() -> None:
    analysis = """```json
    {"sources": [{"index": 2, "verdict": "RELEVANT"}]}
    ```"""
    assert _parse_relevant_indexes(analysis, 5) == {2}


def test_parse_regex_fallback() -> None:
    analysis = """
    ### [1]
    Quote: something
    Verdict: RELEVANT

    ### [2]
    Quote: other
    Verdict: NOT_RELEVANT

    ### [3]
    Quote: more
    Verdict: relevant
    """
    assert _parse_relevant_indexes(analysis, 3) == {1, 3}


def test_parse_ignores_out_of_range() -> None:
    analysis = '{"sources": [{"index": 9, "verdict": "RELEVANT"}]}'
    assert _parse_relevant_indexes(analysis, 3) == set()


def test_parse_empty() -> None:
    assert _parse_relevant_indexes("", 3) == set()
    assert _parse_relevant_indexes("noise", 0) == set()

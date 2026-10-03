"""Unit tests for hybrid retrieval helpers."""

from core.retriever.hybrid import bm25_scores, make_chunk_id, tokenize_for_bm25


def test_make_chunk_id_unique_across_fingerprints() -> None:
    text = "نفس النص للاختبار"
    a = make_chunk_id("fp_aaaa_11111111", 1, 0, text)
    b = make_chunk_id("fp_bbbb_22222222", 1, 0, text)
    assert a != b
    assert a.startswith("fp_aaaa_11111111:")
    assert b.startswith("fp_bbbb_22222222:")


def test_make_chunk_id_stable_for_same_inputs() -> None:
    kwargs = {"fingerprint": "abc123", "page_start": 2, "ordinal": 1, "text": "hello"}
    assert make_chunk_id(**kwargs) == make_chunk_id(**kwargs)


def test_tokenize_for_bm25_filters_stopwords() -> None:
    tokens = tokenize_for_bm25("the Arabic book في البيت and more")
    assert "the" not in tokens
    assert "and" not in tokens
    assert "في" not in tokens
    assert "arabic" in tokens or "Arabic".lower() in tokens
    assert "book" in tokens
    assert "بيت" in tokens
    assert "البيت" not in tokens


def test_tokenize_strips_al_prefix() -> None:
    tokens = tokenize_for_bm25("التعليم في المدرسة")
    assert "تعليم" in tokens
    # ته مربوطة → ه under normalize_arabic(remove_diacritics=True)
    assert "مدرسه" in tokens or "مدرسة" in tokens
    assert "التعليم" not in tokens
    assert "المدرسة" not in tokens
    assert "المدرسه" not in tokens


def test_rrf_fuse_prefers_consensus() -> None:
    from core.retriever.hybrid import rrf_fuse

    scores = rrf_fuse(
        [
            ["a", "b", "c"],
            ["b", "a", "d"],
        ],
        k=60,
    )
    assert scores["b"] > scores["c"]
    assert scores["a"] > scores["c"]
    assert "d" in scores


def test_bm25_scores_basic() -> None:
    query = tokenize_for_bm25("camel desert")
    docs = [
        tokenize_for_bm25("the camel walks in the desert"),
        tokenize_for_bm25("unrelated ocean fish"),
        tokenize_for_bm25("desert wind and camel caravan"),
    ]
    scores = bm25_scores(query, docs)
    assert len(scores) == 3
    assert scores[0] > scores[1]
    assert scores[2] > scores[1]


def test_bm25_empty_query_or_corpus() -> None:
    assert bm25_scores([], [["a", "b"]]) == [0.0]
    assert bm25_scores(["x"], []) == []

"""Shared helpers for stable chunk IDs and hybrid retrieval constants."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence

from core.preprocess.arabic_normalize import normalize_arabic

# Hybrid fusion (semantic vs BM25) — used when HYBRID_FUSION=weighted.
# Favor dense cosine/IP semantic rank slightly over BM25 for Arabic story recall.
SEMANTIC_WEIGHT = 0.7
BM25_WEIGHT = 0.3
LEXICAL_RERANK_WEIGHT = 0.7
SEMANTIC_RERANK_WEIGHT = 0.3

# BM25 defaults.
BM25_K1 = 1.5
BM25_B = 0.75
DEFAULT_RRF_K = 60

# Default chunking (mirrored in settings).
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200

_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "is",
    "are",
    "was",
    "were",
    "with",
    "from",
    "by",
    "this",
    "that",
    "it",
    "as",
    "at",
    "be",
    "of",
    "من",
    "في",
    "على",
    "إلى",
    "الى",
    "عن",
    "مع",
    "هذا",
    "هذه",
    "ذلك",
    "تلك",
    "التي",
    "الذي",
    "الذين",
    "اللذين",
    "ما",
    "لا",
    "لم",
    "لن",
    "إن",
    "ان",
    "أن",
    "أو",
    "او",
    "و",
    "ثم",
    "قد",
    "كان",
    "كانت",
    "يكون",
    "بين",
    "بعد",
    "قبل",
    "عند",
    "حول",
    "كل",
    "بعض",
    "غير",
    "أي",
    "اي",
    "هناك",
    "هنا",
}


def make_chunk_id(
    fingerprint: str,
    page_start: int,
    ordinal: int,
    text: str,
) -> str:
    """Stable, document-namespaced chunk id for vector upserts."""
    fp = (fingerprint or "nofp")[:16]
    digest = hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:8]
    return f"{fp}:{page_start}:{ordinal}:{digest}"


def _strip_al_prefix(token: str) -> str:
    """Strip Arabic definite article ال when token is long enough."""
    if len(token) > 3 and token.startswith("ال"):
        return token[2:]
    return token


def tokenize_for_bm25(text: str) -> list[str]:
    """Normalize and tokenize Arabic/English text for BM25."""
    cleaned = normalize_arabic(text or "", remove_diacritics=True)
    cleaned = cleaned.translate(_ARABIC_INDIC)
    tokens = re.findall(r"[A-Za-z0-9\u0600-\u06FF]+", cleaned.lower())
    out: list[str] = []
    for token in tokens:
        token = _strip_al_prefix(token)
        if not token or token in _STOPWORDS or len(token) <= 1:
            continue
        out.append(token)
    return out


def normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize a score list; all-equal → zeros."""
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi - lo < 1e-12:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def bm25_scores(
    query_tokens: list[str],
    docs_tokens: list[list[str]],
    *,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> list[float]:
    """Classic BM25 over an in-memory corpus."""
    n_docs = len(docs_tokens)
    if n_docs == 0 or not query_tokens:
        return [0.0] * n_docs
    avgdl = sum(len(d) for d in docs_tokens) / n_docs
    df: Counter[str] = Counter()
    for doc in docs_tokens:
        df.update(set(doc))
    scores: list[float] = []
    for doc in docs_tokens:
        tf = Counter(doc)
        score = 0.0
        dl = len(doc) or 1
        for term in query_tokens:
            if term not in tf:
                continue
            n_q = df.get(term, 0)
            idf = max(0.0, ((n_docs - n_q + 0.5) / (n_q + 0.5)))
            idf = math.log(1.0 + idf)
            freq = tf[term]
            denom = freq + k1 * (1.0 - b + b * dl / max(avgdl, 1e-9))
            score += idf * (freq * (k1 + 1.0)) / max(denom, 1e-9)
        scores.append(score)
    return scores


def rrf_fuse(
    ranked_id_lists: Sequence[Sequence[str]],
    *,
    k: int = DEFAULT_RRF_K,
) -> dict[str, float]:
    """Reciprocal Rank Fusion over ordered id lists (best rank = 0)."""
    scores: dict[str, float] = {}
    for ranked in ranked_id_lists:
        for rank, item_id in enumerate(ranked):
            if not item_id:
                continue
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return scores

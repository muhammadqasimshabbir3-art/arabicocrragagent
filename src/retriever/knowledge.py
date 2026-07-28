"""Knowledge-base retrieval against the seeded vector collection (READ-only)."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from config.settings import get_settings
from embeddings.factory import embed_query
from preprocess.arabic_normalize import normalize_arabic
from vectorstore.base import RetrievedChunk, VectorStore
from vectorstore.factory import get_vector_store

_STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "to", "of", "in", "on",
        "for", "and", "or", "what", "which", "who", "how", "do", "does", "did",
        "with", "from", "that", "this", "it", "as", "at", "by", "if", "about",
        "me", "my", "please", "tell",
        "في", "من", "إلى", "على", "عن", "هذا", "هذه", "ذلك", "التي", "الذي",
        "ما", "هل", "او", "أو", "و", "ثم", "قد", "لم", "لن", "إن", "ان",
    }
)


def _tokenize(text: str) -> list[str]:
    normalized = normalize_arabic(text or "", remove_diacritics=True).lower()
    tokens = re.findall(r"[\w\u0600-\u06FF]+", normalized, flags=re.UNICODE)
    return [token for token in tokens if token not in _STOPWORDS and len(token) > 1]


def _bm25_scores(query: str, documents: list[str]) -> list[float]:
    if not documents:
        return []
    q_tokens = _tokenize(query)
    if not q_tokens:
        return [0.0] * len(documents)

    doc_tokens = [_tokenize(doc) for doc in documents]
    df: Counter[str] = Counter()
    for tokens in doc_tokens:
        df.update(set(tokens))

    n_docs = len(documents)
    avgdl = sum(len(tokens) for tokens in doc_tokens) / max(1, n_docs)
    k1, b = 1.5, 0.75
    scores: list[float] = []
    for tokens in doc_tokens:
        tf = Counter(tokens)
        score = 0.0
        dl = len(tokens) or 1
        for term in q_tokens:
            if term not in tf:
                continue
            idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            freq = tf[term]
            score += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * dl / avgdl))
        scores.append(score)
    return scores


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def knowledge_collection_name() -> str:
    settings = get_settings()
    return (settings.knowledge_collection or "wathiqa_knowledge").strip()


def get_knowledge_store() -> VectorStore:
    """Open the configured READ-only knowledge collection."""
    return get_vector_store(collection_name=knowledge_collection_name())


def knowledge_available() -> bool:
    try:
        return get_knowledge_store().count() > 0
    except Exception:  # noqa: BLE001
        return False


def retrieve_from_knowledge(
    query: str,
    *,
    top_k: int | None = None,
    keywords: list[str] | None = None,
) -> list[RetrievedChunk]:
    """Hybrid semantic + BM25 retrieval over the seeded knowledge collection.

    Always READ-only — never writes, updates, or deletes vectors.
    """
    settings = get_settings()
    k = top_k or settings.retrieval_top_k
    search_text = normalize_arabic(query or "", remove_diacritics=False).strip()
    if keywords:
        extras = " ".join(kw for kw in keywords if kw and kw.strip())
        if extras:
            merged = f"{search_text} {extras}".strip()
            search_text = normalize_arabic(merged, remove_diacritics=False).strip()
    if not search_text:
        return []

    store = get_knowledge_store()
    total = store.count()
    if total <= 0:
        return []

    query_vec = embed_query(search_text)
    fetch_k = min(max(k * 5, k), total)
    semantic_hits = store.similarity_search(query_vec, top_k=fetch_k)
    if not semantic_hits:
        return []

    if settings.retrieval_mode != "hybrid" or len(semantic_hits) <= 1:
        return semantic_hits[:k]

    corpus = [hit.text for hit in semantic_hits]
    bm25 = _bm25_scores(search_text, corpus)
    bm25_n = _normalize(bm25)
    fused: list[RetrievedChunk] = []
    for hit, bm25_score in zip(semantic_hits, bm25_n, strict=False):
        score = 0.6 * float(hit.score or 0.0) + 0.4 * bm25_score
        meta = dict(hit.metadata or {})
        meta["retrieval"] = "hybrid_bm25"
        meta["bm25"] = bm25_score
        fused.append(
            RetrievedChunk(
                chunk_id=hit.chunk_id,
                text=hit.text,
                score=score,
                page_start=hit.page_start,
                page_end=hit.page_end,
                filename=hit.filename,
                metadata=meta,
            )
        )
    fused.sort(key=lambda item: item.score, reverse=True)
    return fused[:k]


def knowledge_status() -> dict[str, Any]:
    name = knowledge_collection_name()
    try:
        count = get_knowledge_store().count()
    except Exception as exc:  # noqa: BLE001
        return {"collection": name, "available": False, "count": 0, "error": str(exc)}
    return {"collection": name, "available": count > 0, "count": count}

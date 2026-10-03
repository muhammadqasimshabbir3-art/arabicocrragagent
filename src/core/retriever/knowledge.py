"""Knowledge-base retrieval against the seeded vector collection (READ-only)."""

from __future__ import annotations

from typing import Any

from core.config.settings import get_settings
from subagents.embeddings.factory import embed_query
from core.preprocess.arabic_normalize import normalize_arabic
from subagents.reranker import rerank
from core.retriever.hybrid import (
    BM25_WEIGHT,
    SEMANTIC_WEIGHT,
    bm25_scores,
    normalize_scores,
    rrf_fuse,
    tokenize_for_bm25,
)
from core.vectorstore.base import RetrievedChunk, VectorStore
from core.vectorstore.factory import get_vector_store


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

    Over-fetches semantic candidates, then fuses with BM25 via RRF (default) or
    weighted sum. Always READ-only — never writes, updates, or deletes vectors.
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
    mult = max(1, settings.retrieval_candidate_multiplier)
    fetch_k = min(max(k * mult, k), total)
    semantic_hits = store.similarity_search(query_vec, top_k=fetch_k)
    if not semantic_hits:
        return []

    if settings.retrieval_mode != "hybrid" or len(semantic_hits) <= 1:
        hits = semantic_hits
    else:
        corpus_tokens = [tokenize_for_bm25(hit.text) for hit in semantic_hits]
        query_tokens = tokenize_for_bm25(search_text)
        bm25 = bm25_scores(query_tokens, corpus_tokens)

        if settings.hybrid_fusion == "weighted":
            bm25_n = normalize_scores(bm25)
            fused: list[RetrievedChunk] = []
            for hit, bm25_score in zip(semantic_hits, bm25_n, strict=False):
                score = SEMANTIC_WEIGHT * float(hit.score or 0.0) + BM25_WEIGHT * bm25_score
                meta = dict(hit.metadata or {})
                meta["retrieval"] = "hybrid_weighted"
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
            hits = fused
        else:
            sem_ranked = [hit.chunk_id for hit in semantic_hits]
            bm25_order = sorted(
                range(len(semantic_hits)),
                key=lambda i: bm25[i],
                reverse=True,
            )
            bm25_ranked = [
                semantic_hits[i].chunk_id for i in bm25_order if bm25[i] > 0
            ]
            seen = set(bm25_ranked)
            bm25_ranked.extend(
                h.chunk_id for h in semantic_hits if h.chunk_id not in seen
            )
            rrf_scores = rrf_fuse([sem_ranked, bm25_ranked], k=settings.rrf_k)
            by_id = {hit.chunk_id: hit for hit in semantic_hits}
            fused = []
            for chunk_id, score in rrf_scores.items():
                hit = by_id.get(chunk_id)
                if hit is None:
                    continue
                meta = dict(hit.metadata or {})
                meta["retrieval"] = "hybrid_rrf"
                fused.append(
                    RetrievedChunk(
                        chunk_id=hit.chunk_id,
                        text=hit.text,
                        score=float(score),
                        page_start=hit.page_start,
                        page_end=hit.page_end,
                        filename=hit.filename,
                        metadata=meta,
                    )
                )
            fused.sort(key=lambda item: item.score, reverse=True)
            hits = fused

    if settings.retrieval_min_score > 0:
        hits = [h for h in hits if float(h.score or 0.0) >= settings.retrieval_min_score]

    if settings.enable_reranker:
        return rerank(search_text, hits, top_k=k)
    return hits[:k]


def knowledge_status() -> dict[str, Any]:
    name = knowledge_collection_name()
    try:
        count = get_knowledge_store().count()
    except Exception as exc:  # noqa: BLE001
        return {"collection": name, "available": False, "count": 0, "error": str(exc)}
    return {"collection": name, "available": count > 0, "count": count}

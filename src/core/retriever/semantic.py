"""Document indexing and retrieval orchestration."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from core.chunking.base import TextChunk
from core.chunking.factory import chunk_document
from core.config.settings import get_settings
from subagents.embeddings.factory import embed_documents, embed_query
from core.loaders.factory import load_document
from subagents.ocr.base import OCRDocument
from subagents.ocr.factory import extract_document_text
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
from core.vectorstore.base import RetrievedChunk
from core.vectorstore.factory import get_vector_store

_INDEX_CACHE: OrderedDict[str, DocumentIndex] = OrderedDict()
_INDEX_CACHE_LOCK = threading.Lock()


@dataclass
class DocumentIndex:
    """In-memory indexed document ready for RAG."""

    fingerprint: str
    filename: str
    ocr: OCRDocument
    chunks: list[TextChunk]
    store: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return self.ocr.full_text


def _cache_get(fingerprint: str) -> DocumentIndex | None:
    with _INDEX_CACHE_LOCK:
        cached = _INDEX_CACHE.get(fingerprint)
        if cached is not None:
            _INDEX_CACHE.move_to_end(fingerprint)
        return cached


def _cache_put(fingerprint: str, index: DocumentIndex) -> None:
    settings = get_settings()
    with _INDEX_CACHE_LOCK:
        _INDEX_CACHE[fingerprint] = index
        _INDEX_CACHE.move_to_end(fingerprint)
        while len(_INDEX_CACHE) > settings.index_cache_size:
            _INDEX_CACHE.popitem(last=False)


def build_index_from_payload(
    data_base64: str,
    filename: str = "uploaded.pdf",
    mime_type: str | None = None,
) -> DocumentIndex:
    """Load → OCR → chunk → embed → index an uploaded document."""
    loaded = load_document(data_base64, filename=filename, mime_type=mime_type)
    cached = _cache_get(loaded.fingerprint)
    if cached:
        return cached

    ocr_doc = extract_document_text(loaded)
    chunks = chunk_document(ocr_doc)
    if not chunks:
        raise ValueError("No searchable text chunks could be produced from the document.")

    embeddings = embed_documents([chunk.text for chunk in chunks])
    store = get_vector_store(
        collection_name=f"doc_{loaded.fingerprint[:16]}",
    )
    store.add_chunks(chunks, embeddings, filename=loaded.filename)

    index = DocumentIndex(
        fingerprint=loaded.fingerprint,
        filename=loaded.filename,
        ocr=ocr_doc,
        chunks=chunks,
        store=store,
        metadata={
            "engine": ocr_doc.engine,
            "chunk_count": len(chunks),
            "page_count": len(ocr_doc.pages),
        },
    )
    _cache_put(loaded.fingerprint, index)
    return index


def get_or_build_index(
    data_base64: str,
    filename: str = "uploaded.pdf",
    mime_type: str | None = None,
) -> DocumentIndex:
    """Public alias for build_index_from_payload."""
    return build_index_from_payload(data_base64, filename=filename, mime_type=mime_type)


def upsert_index_into_knowledge(index: DocumentIndex) -> int:
    """Copy an indexed document into the shared READ-only knowledge collection."""
    from subagents.embeddings.factory import embed_documents
    from core.retriever.knowledge import get_knowledge_store

    if not index.chunks:
        return 0
    store = get_knowledge_store()
    embeddings = embed_documents([chunk.text for chunk in index.chunks])
    store.add_chunks(index.chunks, embeddings, filename=index.filename)
    return store.count()


def retrieve(
    index: DocumentIndex,
    query: str,
    *,
    top_k: int | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
    filename: str | None = None,
) -> list[RetrievedChunk]:
    """Retrieve relevant chunks (semantic or hybrid), with optional metadata filters."""
    settings = get_settings()
    k = top_k or settings.retrieval_top_k
    search_query = query or ""
    if settings.enable_query_rewrite:
        try:
            from subagents.query_planner import to_arabic_retrieval_query
            from core.routing.language import detect_language

            lang = detect_language(search_query)
            if lang in {"ar", "mixed"}:
                rewritten = to_arabic_retrieval_query(search_query)
                if rewritten and rewritten.strip():
                    search_query = rewritten.strip()
            elif lang == "en":
                # Optional: Arabic retrieval query helps Arabic corpora.
                rewritten = to_arabic_retrieval_query(search_query)
                if rewritten and rewritten.strip() and rewritten.strip() != search_query:
                    search_query = f"{search_query} {rewritten.strip()}"
        except Exception:  # noqa: BLE001
            pass

    normalized_query = normalize_arabic(search_query, remove_diacritics=False).strip()
    if not normalized_query:
        return []
    query_vec = embed_query(normalized_query)
    mult = max(1, settings.retrieval_candidate_multiplier)
    fetch_k = max(k * mult, k)
    semantic_hits = index.store.similarity_search(query_vec, top_k=fetch_k)

    if settings.retrieval_mode != "hybrid" or len(index.chunks) <= 1:
        hits = semantic_hits
    else:
        corpus_tokens = [tokenize_for_bm25(chunk.text) for chunk in index.chunks]
        query_tokens = tokenize_for_bm25(normalized_query)
        bm25 = bm25_scores(query_tokens, corpus_tokens)
        chunk_by_id = {chunk.chunk_id: chunk for chunk in index.chunks}
        semantic_by_id = {hit.chunk_id: hit for hit in semantic_hits}

        if settings.hybrid_fusion == "weighted":
            bm25_n = normalize_scores(bm25)
            fused: list[RetrievedChunk] = []
            for i, chunk in enumerate(index.chunks):
                sem = semantic_by_id.get(chunk.chunk_id)
                sem_score = sem.score if sem else 0.0
                score = SEMANTIC_WEIGHT * sem_score + BM25_WEIGHT * bm25_n[i]
                fused.append(
                    RetrievedChunk(
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        score=score,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        filename=index.filename,
                        metadata={**dict(chunk.metadata), "fusion": "weighted"},
                    )
                )
            fused.sort(key=lambda item: item.score, reverse=True)
            hits = fused
        else:
            # RRF (default): fuse semantic rank list + BM25 rank list.
            sem_ranked = [hit.chunk_id for hit in semantic_hits]
            bm25_order = sorted(
                range(len(index.chunks)),
                key=lambda i: bm25[i],
                reverse=True,
            )
            bm25_ranked = [index.chunks[i].chunk_id for i in bm25_order if bm25[i] > 0]
            # Include zero-BM25 chunks at the end so ids are complete for union.
            seen = set(bm25_ranked)
            bm25_ranked.extend(
                c.chunk_id for c in index.chunks if c.chunk_id not in seen
            )
            rrf_scores = rrf_fuse([sem_ranked, bm25_ranked], k=settings.rrf_k)
            fused = []
            for chunk_id, score in rrf_scores.items():
                chunk = chunk_by_id.get(chunk_id)
                if chunk is None:
                    continue
                fused.append(
                    RetrievedChunk(
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        score=float(score),
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        filename=index.filename,
                        metadata={**dict(chunk.metadata), "fusion": "rrf"},
                    )
                )
            fused.sort(key=lambda item: item.score, reverse=True)
            hits = fused

    def _passes_filters(chunk: RetrievedChunk) -> bool:
        if filename and chunk.filename and chunk.filename != filename:
            return False
        if page_start is not None and chunk.page_end and chunk.page_end < page_start:
            return False
        if page_end is not None and chunk.page_start and chunk.page_start > page_end:
            return False
        return True

    hits = [hit for hit in hits if _passes_filters(hit)][: max(fetch_k, k)]
    if settings.retrieval_min_score > 0:
        hits = [h for h in hits if float(h.score or 0.0) >= settings.retrieval_min_score]

    if settings.enable_reranker:
        hits = rerank(normalized_query, hits, top_k=k)
    else:
        hits = hits[:k]
    return hits


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks for the LLM context window."""
    if not chunks:
        return ""
    parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        if chunk.page_start and chunk.page_end and chunk.page_start != chunk.page_end:
            page_label = f"Pages {chunk.page_start}-{chunk.page_end}"
        elif chunk.page_start:
            page_label = f"Page {chunk.page_start}"
        else:
            page_label = "Page ?"
        parts.append(
            f"[{index}] {chunk.filename} · {page_label} · score={chunk.score:.3f}\n"
            f"{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)

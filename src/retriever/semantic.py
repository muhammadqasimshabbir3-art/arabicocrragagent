"""Document indexing and retrieval orchestration."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from chunking.base import TextChunk
from chunking.factory import chunk_document
from config.settings import get_settings
from embeddings.factory import embed_documents, embed_query
from loaders.factory import load_document
from ocr.base import OCRDocument
from ocr.factory import extract_document_text
from preprocess.arabic_normalize import normalize_arabic
from reranker import rerank
from vectorstore.base import RetrievedChunk
from vectorstore.factory import get_vector_store

_INDEX_CACHE: dict[str, "DocumentIndex"] = {}

_STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "to", "of", "in", "on",
        "for", "and", "or", "what", "which", "who", "how", "do", "does", "did",
        "with", "from", "that", "this", "it", "as", "at", "by", "if", "about",
        "me", "my", "please", "tell",
        # Arabic function words
        "في", "من", "إلى", "على", "عن", "هذا", "هذه", "ذلك", "التي", "الذي",
        "ما", "هل", "او", "أو", "و", "ثم", "قد", "لم", "لن", "إن", "ان",
    }
)


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


def build_index_from_payload(
    data_base64: str,
    filename: str = "uploaded.pdf",
    mime_type: str | None = None,
) -> DocumentIndex:
    """Load → OCR → chunk → embed → index an uploaded document."""
    settings = get_settings()
    loaded = load_document(data_base64, filename=filename, mime_type=mime_type)
    cached = _INDEX_CACHE.get(loaded.fingerprint)
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
    _INDEX_CACHE[loaded.fingerprint] = index

    # Bound memory for long-running sessions.
    while len(_INDEX_CACHE) > settings.index_cache_size:
        oldest = next(iter(_INDEX_CACHE))
        if oldest == loaded.fingerprint:
            break
        _INDEX_CACHE.pop(oldest, None)

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
    from embeddings.factory import embed_documents
    from retriever.knowledge import get_knowledge_store

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
    normalized_query = normalize_arabic(query or "", remove_diacritics=False).strip()
    if not normalized_query:
        return []
    query_vec = embed_query(normalized_query)
    # Over-fetch before filters / rerank.
    fetch_k = max(k * 3, k)
    semantic_hits = index.store.similarity_search(query_vec, top_k=fetch_k)

    if settings.retrieval_mode != "hybrid" or len(index.chunks) <= 1:
        hits = semantic_hits
    else:
        corpus = [chunk.text for chunk in index.chunks]
        bm25 = _bm25_scores(normalized_query, corpus)
        bm25_n = _normalize(bm25)
        semantic_by_id = {hit.chunk_id: hit for hit in semantic_hits}
        fused: list[RetrievedChunk] = []
        for i, chunk in enumerate(index.chunks):
            sem = semantic_by_id.get(chunk.chunk_id)
            sem_score = sem.score if sem else 0.0
            score = 0.6 * sem_score + 0.4 * bm25_n[i]
            fused.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=score,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    filename=index.filename,
                    metadata=dict(chunk.metadata),
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

    if settings.enable_reranker:
        hits = rerank(query, hits, top_k=k)
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

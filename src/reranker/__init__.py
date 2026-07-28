"""Optional reranking of retrieved chunks."""

from __future__ import annotations

import math
import re
from functools import lru_cache

from config.settings import get_settings
from preprocess.arabic_normalize import normalize_arabic
from vectorstore.base import RetrievedChunk
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def _tokenize(text: str) -> set[str]:
    normalized = normalize_arabic(text or "", remove_diacritics=True).lower()
    return set(re.findall(r"[\w\u0600-\u06FF]+", normalized, flags=re.UNICODE))


def _lexical_overlap_score(query: str, passage: str) -> float:
    q = _tokenize(query)
    p = _tokenize(passage)
    if not q or not p:
        return 0.0
    return len(q & p) / math.sqrt(len(q) * len(p))


@lru_cache(maxsize=1)
def _cross_encoder():
    """Lazy-load a multilingual cross-encoder when available."""
    settings = get_settings()
    model_id = settings.reranker_model_id
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        logger.warning("sentence-transformers CrossEncoder unavailable; using lexical rerank")
        return None
    try:
        return CrossEncoder(model_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load reranker %s: %s", model_id, exc)
        return None


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Rerank retrieved chunks with cross-encoder when possible, else lexical overlap."""
    if not chunks:
        return []

    settings = get_settings()
    limit = top_k or settings.retrieval_top_k
    model = None
    if settings.reranker_backend == "cross_encoder":
        model = _cross_encoder()

    scored: list[RetrievedChunk] = []
    if model is not None:
        pairs = [(query, chunk.text) for chunk in chunks]
        try:
            scores = model.predict(pairs)
            for chunk, score in zip(chunks, scores, strict=False):
                scored.append(
                    RetrievedChunk(
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        score=float(score),
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        filename=chunk.filename,
                        metadata={**chunk.metadata, "rerank": "cross_encoder"},
                    )
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cross-encoder rerank failed (%s); falling back to lexical", exc)
            model = None

    if model is None:
        for chunk in chunks:
            lexical = _lexical_overlap_score(query, chunk.text)
            fused = 0.7 * float(chunk.score) + 0.3 * lexical
            scored.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=fused,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    filename=chunk.filename,
                    metadata={**chunk.metadata, "rerank": "lexical"},
                )
            )

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:limit]

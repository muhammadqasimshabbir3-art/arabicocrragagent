"""Compatibility shim — prefer `from subagents.embeddings import …`."""

from subagents.embeddings.bge_m3 import model_dir
from subagents.embeddings.factory import (
    EMBEDDING_DIMENSIONS,
    MODEL_ID,
    embed_documents,
    embed_query,
    embed_text,
    get_embedder,
)


def get_embedding_model():
    """Lazy-load the local embedding model (downloads on first use)."""
    from subagents.embeddings.bge_m3 import _load_sentence_transformer

    return _load_sentence_transformer()


__all__ = [
    "EMBEDDING_DIMENSIONS",
    "MODEL_ID",
    "embed_documents",
    "embed_query",
    "embed_text",
    "get_embedder",
    "get_embedding_model",
    "model_dir",
]

"""Embeddings subagent package — BGE-M3 dense vector model."""
from subagents.embeddings.factory import (
    embed_documents,
    embed_query,
    embed_text,
    get_embedder,
    EMBEDDING_DIMENSIONS,
)

__all__ = ["embed_documents", "embed_query", "embed_text", "get_embedder", "EMBEDDING_DIMENSIONS"]

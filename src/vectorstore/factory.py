"""Vector store factory — swap backends via configuration."""

from __future__ import annotations

from config.settings import get_settings
from vectorstore.base import VectorStore
from vectorstore.chroma_store import ChromaVectorStore


def get_vector_store(
    backend: str | None = None,
    *,
    collection_name: str | None = None,
) -> VectorStore:
    """Create a vector store for the configured backend."""
    settings = get_settings()
    name = (backend or settings.vectorstore_backend or "chroma").lower()

    if name == "chroma":
        persist = (settings.vectorstore_persist_dir or "").strip()
        if persist.lower() in {"", "memory", "none", "ephemeral"}:
            return ChromaVectorStore(collection_name=collection_name)
        return ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=persist,
        )

    if name == "faiss":
        from vectorstore.faiss_store import FaissVectorStore

        return FaissVectorStore(collection_name=collection_name)

    if name == "qdrant":
        from vectorstore.qdrant_store import QdrantVectorStore

        return QdrantVectorStore(collection_name=collection_name)

    if name == "milvus":
        raise NotImplementedError(
            "Milvus backend is reserved for a future adapter. "
            "Use VECTORSTORE_BACKEND=chroma, faiss, or qdrant."
        )

    raise ValueError(f"Unknown vectorstore backend: {name}")

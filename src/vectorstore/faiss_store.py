"""In-memory FAISS vector store backend."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from chunking.base import TextChunk
from vectorstore.base import RetrievedChunk, VectorStore


class FaissVectorStore(VectorStore):
    """Ephemeral FAISS index for uploaded document sessions."""

    name = "faiss"

    def __init__(self, collection_name: str | None = None) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "FAISS backend requires `faiss-cpu` (or `faiss-gpu`). "
                "Install it or set VECTORSTORE_BACKEND=chroma."
            ) from exc

        self._collection_name = collection_name or "docs"
        self._index = None
        self._records: list[dict[str, Any]] = []
        self._dim: int | None = None

    def add_chunks(
        self,
        chunks: Sequence[TextChunk],
        embeddings: Sequence[Sequence[float]],
        *,
        filename: str,
    ) -> None:
        import faiss

        if not chunks:
            return
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
            raise ValueError("embeddings must match chunks one-to-one")

        # Cosine similarity via inner product on L2-normalized vectors.
        faiss.normalize_L2(matrix)
        dim = matrix.shape[1]
        if self._index is None:
            self._dim = dim
            self._index = faiss.IndexFlatIP(dim)
        elif self._dim != dim:
            raise ValueError(f"Embedding dim mismatch: expected {self._dim}, got {dim}")

        self._index.add(matrix)
        for chunk in chunks:
            self._records.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "filename": filename,
                    "metadata": {
                        "filename": filename,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        **{
                            key: value
                            for key, value in chunk.metadata.items()
                            if isinstance(value, (str, int, float, bool))
                        },
                    },
                }
            )

    def similarity_search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        import faiss

        if self._index is None or self.count() <= 0:
            return []

        query = np.asarray([list(query_embedding)], dtype=np.float32)
        faiss.normalize_L2(query)
        k = min(top_k, self.count())
        scores, indices = self._index.search(query, k)

        retrieved: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0 or idx >= len(self._records):
                continue
            record = self._records[idx]
            retrieved.append(
                RetrievedChunk(
                    chunk_id=str(record["chunk_id"]),
                    text=str(record["text"]),
                    score=float(score),
                    page_start=int(record["page_start"] or 0),
                    page_end=int(record["page_end"] or 0),
                    filename=str(record["filename"] or ""),
                    metadata=dict(record["metadata"]),
                )
            )
        return retrieved

    def count(self) -> int:
        return len(self._records)

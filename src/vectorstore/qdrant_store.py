"""Qdrant vector store — production-style remote or local server."""

from __future__ import annotations

import uuid
from typing import Sequence

from chunking.base import TextChunk
from config.settings import get_settings
from vectorstore.base import RetrievedChunk, VectorStore


class QdrantVectorStore(VectorStore):
    """Qdrant collection for durable, production-like vector search."""

    name = "qdrant"

    def __init__(
        self,
        collection_name: str | None = None,
        *,
        url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qmodels
        except ImportError as exc:
            raise RuntimeError(
                "Qdrant backend requires `qdrant-client`. "
                "Install with: uv sync --extra qdrant"
            ) from exc

        settings = get_settings()
        self._qmodels = qmodels
        self._url = (url or settings.qdrant_url or "http://localhost:6333").strip()
        self._api_key = (api_key or settings.qdrant_api_key or "").strip() or None
        self._collection = collection_name or f"wathiqa_{uuid.uuid4().hex[:12]}"
        self._dims = settings.embedding_dimensions

        self._client = QdrantClient(url=self._url, api_key=self._api_key, timeout=180)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        qmodels = self._qmodels
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qmodels.VectorParams(
                size=self._dims,
                distance=qmodels.Distance.COSINE,
            ),
        )

    @property
    def collection_name(self) -> str:
        return self._collection

    def add_chunks(
        self,
        chunks: Sequence[TextChunk],
        embeddings: Sequence[Sequence[float]],
        *,
        filename: str,
    ) -> None:
        if not chunks:
            return
        import time

        qmodels = self._qmodels
        points = []
        for chunk, vector in zip(chunks, embeddings, strict=False):
            payload = {
                "text": chunk.text,
                "filename": filename,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                **{
                    key: value
                    for key, value in chunk.metadata.items()
                    if isinstance(value, (str, int, float, bool))
                },
            }
            points.append(
                qmodels.PointStruct(
                    id=self._point_id(chunk.chunk_id),
                    vector=list(vector),
                    payload=payload,
                )
            )
        batch_size = 8
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            last_error: Exception | None = None
            for attempt in range(1, 6):
                try:
                    self._client.upsert(
                        collection_name=self._collection,
                        points=batch,
                        wait=True,
                    )
                    last_error = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    time.sleep(min(30, 2 ** attempt))
            if last_error is not None:
                raise last_error

    def _point_id(self, chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    def similarity_search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        total = self.count()
        if total <= 0:
            return []
        limit = min(top_k, total)
        if hasattr(self._client, "search"):
            hits = self._client.search(
                collection_name=self._collection,
                query_vector=list(query_embedding),
                limit=limit,
                with_payload=True,
            )
        else:
            result = self._client.query_points(
                collection_name=self._collection,
                query=list(query_embedding),
                limit=limit,
                with_payload=True,
            )
            hits = getattr(result, "points", [])
        retrieved: list[RetrievedChunk] = []
        for hit in hits:
            payload = hit.payload or {}
            retrieved.append(
                RetrievedChunk(
                    chunk_id=str(hit.id),
                    text=str(payload.get("text") or ""),
                    score=float(hit.score or 0.0),
                    page_start=int(payload.get("page_start") or 0),
                    page_end=int(payload.get("page_end") or 0),
                    filename=str(payload.get("filename") or ""),
                    metadata=dict(payload),
                )
            )
        return retrieved

    def count(self) -> int:
        info = self._client.get_collection(self._collection)
        return int(info.points_count or 0)

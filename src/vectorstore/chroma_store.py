"""Chroma vector store — persistent on disk by default (free & open source).

Chroma: https://github.com/chroma-core/chroma (Apache-2.0)
Default path: data/chroma under the repo (VECTORSTORE_PERSIST_DIR).
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Sequence

from chunking.base import TextChunk
from vectorstore.base import RetrievedChunk, VectorStore


class ChromaVectorStore(VectorStore):
    """Chroma collection backed by a local persist directory when configured."""

    name = "chroma"

    def __init__(
        self,
        collection_name: str | None = None,
        *,
        persist_directory: str | Path | None = None,
    ) -> None:
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise RuntimeError(
                "Chroma backend requires the `chromadb` package."
            ) from exc

        settings = Settings(anonymized_telemetry=False)
        name = collection_name or f"docs_{uuid.uuid4().hex[:12]}"

        if persist_directory:
            path = Path(persist_directory).expanduser().resolve()
            path.mkdir(parents=True, exist_ok=True)
            self._persist_directory = path
            self._client = chromadb.PersistentClient(
                path=str(path),
                settings=settings,
            )
            self._collection = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        else:
            self._persist_directory = None
            self._client = chromadb.Client(
                Settings(anonymized_telemetry=False, is_persistent=False)
            )
            self._collection = self._client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )

    @property
    def persist_directory(self) -> Path | None:
        return self._persist_directory

    @property
    def collection_name(self) -> str:
        return self._collection.name

    def add_chunks(
        self,
        chunks: Sequence[TextChunk],
        embeddings: Sequence[Sequence[float]],
        *,
        filename: str,
    ) -> None:
        if not chunks:
            return
        ids = [chunk.chunk_id for chunk in chunks]
        # Upsert so re-seeding the same document refreshes vectors.
        self._collection.upsert(
            ids=ids,
            documents=[chunk.text for chunk in chunks],
            embeddings=[list(vector) for vector in embeddings],
            metadatas=[
                {
                    "filename": filename,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    **{
                        key: value
                        for key, value in chunk.metadata.items()
                        if isinstance(value, (str, int, float, bool))
                    },
                }
                for chunk in chunks
            ],
        )

    def similarity_search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        total = self.count()
        if total <= 0:
            return []
        results = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=min(top_k, total),
            include=["documents", "metadatas", "distances"],
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for doc_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=False
        ):
            meta = metadata if isinstance(metadata, dict) else {}
            score = 1.0 - float(distance) if distance is not None else 0.0
            retrieved.append(
                RetrievedChunk(
                    chunk_id=str(doc_id),
                    text=str(document),
                    score=score,
                    page_start=int(meta.get("page_start") or 0),
                    page_end=int(meta.get("page_end") or 0),
                    filename=str(meta.get("filename") or ""),
                    metadata=meta,
                )
            )
        return retrieved

    def count(self) -> int:
        return int(self._collection.count())

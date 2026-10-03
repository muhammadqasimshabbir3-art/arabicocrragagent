"""Retrieval package."""

from __future__ import annotations

from typing import Any

__all__ = [
    "DocumentIndex",
    "get_or_build_index",
    "knowledge_available",
    "knowledge_status",
    "retrieve",
    "retrieve_from_knowledge",
    "upsert_index_into_knowledge",
]


def __getattr__(name: str) -> Any:
    # Lazy exports avoid circular imports with chunking → hybrid.
    if name in {
        "DocumentIndex",
        "get_or_build_index",
        "retrieve",
        "upsert_index_into_knowledge",
    }:
        from . import semantic

        return getattr(semantic, name)
    if name in {
        "knowledge_available",
        "knowledge_status",
        "retrieve_from_knowledge",
    }:
        from . import knowledge

        return getattr(knowledge, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

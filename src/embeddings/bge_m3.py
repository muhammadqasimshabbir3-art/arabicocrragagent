"""BGE-M3 multilingual embeddings (Arabic + English) via sentence-transformers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

from config.settings import get_settings
from embeddings.base import Embedder
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def _model_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / "modules.json").is_file() or (path / "config.json").is_file()


def model_dir() -> Path:
    """Return on-disk model directory (override with EMBEDDING_MODEL_PATH)."""
    settings = get_settings()
    if settings.embedding_model_path:
        return Path(settings.embedding_model_path).expanduser().resolve()
    safe = settings.embedding_model_id.replace("/", "__")
    # Prefer human-friendly bge-m3 folder for the default model.
    if settings.embedding_model_id == "BAAI/bge-m3":
        return settings.models_dir / "bge-m3"
    return settings.models_dir / safe


@lru_cache(maxsize=1)
def _load_sentence_transformer():
    from sentence_transformers import SentenceTransformer

    settings = get_settings()
    path = model_dir()
    if _model_ready(path):
        logger.info("Loading embeddings from %s", path)
        return SentenceTransformer(str(path))

    path.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading embeddings %s → %s", settings.embedding_model_id, path)
    model = SentenceTransformer(settings.embedding_model_id, trust_remote_code=True)
    model.save(str(path))
    return model


class BGEM3Embedder(Embedder):
    """BAAI/bge-m3 dense embeddings for multilingual Arabic RAG."""

    def __init__(self) -> None:
        settings = get_settings()
        self.model_id = settings.embedding_model_id
        self.dimensions = settings.embedding_dimensions

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = _load_sentence_transformer()
        vectors = model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        model = _load_sentence_transformer()
        vector = model.encode(
            (text or "").strip(),
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vector.tolist()

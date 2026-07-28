"""Central configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the Arabic Document Intelligence Agent."""

    # LLM
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    llm_model: str = ""  # optional override via LLM_MODEL
    llm_provider: str = "groq"  # groq | openai | ollama | gemini
    llm_temperature: float = 0.0

    # OCR
    ocr_engine: str = "qari"  # qari | ain | none
    ocr_model_id: str = "NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct"
    ocr_model_path: str = ""
    ocr_max_tokens: int = 2000
    ocr_device: str = "auto"  # auto | cpu | cuda

    # Embeddings
    embedding_model_id: str = "BAAI/bge-m3"
    embedding_model_path: str = ""
    embedding_dimensions: int = 1024

    # Chunking
    chunk_strategy: str = "page_aware"  # recursive | page_aware | layout_aware | semantic
    chunk_size: int = 1200
    chunk_overlap: int = 200

    # Retrieval
    vectorstore_backend: str = "chroma"  # chroma | faiss | qdrant | milvus
    vectorstore_persist_dir: str = ""
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    # Seeded corpus collection used for chat questions without PDF upload.
    knowledge_collection: str = "wathiqa_knowledge"
    retrieval_top_k: int = 5
    retrieval_mode: str = "hybrid"  # semantic | hybrid
    enable_reranker: bool = False
    reranker_backend: str = "lexical"  # lexical | cross_encoder
    reranker_model_id: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    # Documents
    max_document_bytes: int = 50 * 1024 * 1024
    index_cache_size: int = 8

    # Paths
    models_dir: Path = field(default_factory=lambda: _REPO_ROOT / "models")
    repo_root: Path = field(default_factory=lambda: _REPO_ROOT)

    # Logging
    log_level: str = "INFO"


def _env(key: str, default: str = "") -> str:
    return (os.getenv(key) or default).strip()


def _env_int(key: str, default: int) -> int:
    raw = _env(key)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = _env(key)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = _env(key).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings loaded from the environment."""
    models_dir = Path(_env("MODELS_DIR") or str(_REPO_ROOT / "models")).expanduser()
    return Settings(
        groq_api_key=_env("GROQ_API_KEY"),
        groq_model=_env("GROQ_MODEL", "llama-3.1-8b-instant"),
        llm_model=_env("LLM_MODEL"),
        llm_provider=_env("LLM_PROVIDER", "groq").lower(),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.0),
        ocr_engine=_env("OCR_ENGINE", "qari").lower(),
        ocr_model_id=_env(
            "OCR_MODEL_ID",
            "NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct",
        ),
        ocr_model_path=_env("OCR_MODEL_PATH"),
        ocr_max_tokens=_env_int("OCR_MAX_TOKENS", 2000),
        ocr_device=_env("OCR_DEVICE", "auto").lower(),
        embedding_model_id=_env("EMBEDDING_MODEL_ID", "BAAI/bge-m3"),
        embedding_model_path=_env("EMBEDDING_MODEL_PATH"),
        embedding_dimensions=_env_int("EMBEDDING_DIMENSIONS", 1024),
        chunk_strategy=_env("CHUNK_STRATEGY", "page_aware").lower(),
        chunk_size=_env_int("CHUNK_SIZE", 1200),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 200),
        vectorstore_backend=_env("VECTORSTORE_BACKEND", "chroma").lower(),
        vectorstore_persist_dir=_env(
            "VECTORSTORE_PERSIST_DIR",
            str(_REPO_ROOT / "data" / "chroma"),
        ),
        qdrant_url=_env("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=_env("QDRANT_API_KEY"),
        knowledge_collection=_env("KNOWLEDGE_COLLECTION", "wathiqa_knowledge"),
        retrieval_top_k=_env_int("RETRIEVAL_TOP_K", 5),
        retrieval_mode=_env("RETRIEVAL_MODE", "hybrid").lower(),
        enable_reranker=_env_bool("ENABLE_RERANKER", False),
        reranker_backend=_env("RERANKER_BACKEND", "lexical").lower(),
        reranker_model_id=_env(
            "RERANKER_MODEL_ID",
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        ),
        max_document_bytes=_env_int("MAX_DOCUMENT_BYTES", 50 * 1024 * 1024),
        index_cache_size=_env_int("INDEX_CACHE_SIZE", 8),
        models_dir=models_dir,
        repo_root=_REPO_ROOT,
        log_level=_env("LOG_LEVEL", "INFO").upper(),
    )

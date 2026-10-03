"""Central configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Backend root = `src/` (packages + models + local chroma data).
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
# Git / monorepo root (frontend, scripts, root .env).
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Prefer repo-root .env for local ./setup.sh; allow src/.env overrides.
load_dotenv(_REPO_ROOT / ".env")
load_dotenv(_BACKEND_ROOT / ".env", override=True)


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the Arabic Document Intelligence Agent."""

    # LLM
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    llm_model: str = ""  # optional override via LLM_MODEL
    llm_provider: str = "groq"  # groq | openai | ollama | gemini
    llm_temperature: float = 0.0

    # Text extraction — digital only (PDF text layer / TXT / DOCX)
    ocr_engine: str = "digital"
    ocr_max_pages: int = 120

    # Embeddings
    embedding_model_id: str = "BAAI/bge-m3"
    embedding_model_path: str = ""
    embedding_dimensions: int = 1024

    # Chunking
    chunk_strategy: str = "page_aware"  # recursive | page_aware | layout_aware | semantic
    chunk_size: int = 1200
    chunk_overlap: int = 200

    # Retrieval
    vectorstore_backend: str = "chroma"  # chroma | faiss | qdrant
    vectorstore_persist_dir: str = ""
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    # Seeded corpus collection used for chat questions without PDF upload.
    knowledge_collection: str = "wathiqa_knowledge"
    # LLM / search timeouts
    llm_timeout_seconds: int = 90
    web_search_timeout_seconds: int = 30
    retrieval_top_k: int = 8
    retrieval_mode: str = "hybrid"  # semantic | hybrid
    hybrid_fusion: str = "rrf"  # rrf | weighted
    rrf_k: int = 60
    retrieval_candidate_multiplier: int = 15
    retrieval_min_score: float = 0.0
    enable_query_rewrite: bool = True
    enable_reranker: bool = True
    reranker_backend: str = "cross_encoder"  # lexical | cross_encoder
    reranker_model_id: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    # Spelling Guardian: correct only high-confidence typos (ar/en)
    enable_spelling_corrector: bool = True
    spelling_min_confidence: float = 0.85
    spelling_use_llm: bool = True
    # Query Booster: add 2-line context notes for short/ambiguous queries.
    enable_query_booster: bool = True
    query_booster_use_llm: bool = True
    query_booster_min_words: int = 6

    # Documents
    max_document_bytes: int = 50 * 1024 * 1024
    index_cache_size: int = 8
    # How much digital text to feed the summarizer (500–900 line docs need more)
    summary_max_chars: int = 48000

    # Paths
    models_dir: Path = field(default_factory=lambda: _BACKEND_ROOT / "models")
    backend_root: Path = field(default_factory=lambda: _BACKEND_ROOT)
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


def validate_startup_settings(settings: Settings | None = None) -> None:
    """Fail fast on unsafe or incomplete production configuration."""
    from core.utils.sanitize import validate_collection_name

    cfg = settings or get_settings()
    if cfg.llm_provider == "groq" and not cfg.groq_api_key:
        raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER=groq.")
    if cfg.vectorstore_backend == "qdrant" and not cfg.qdrant_url:
        raise ValueError("QDRANT_URL is required when VECTORSTORE_BACKEND=qdrant.")
    validate_collection_name(cfg.knowledge_collection)
    if cfg.ocr_engine not in {"digital", "digital_text", "none"}:
        raise ValueError(
            f"Unsupported OCR_ENGINE={cfg.ocr_engine!r}. "
            "This agent supports digital text extraction only "
            "(text-layer PDF / TXT / DOCX)."
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings loaded from the environment."""
    from core.utils.sanitize import validate_collection_name

    models_dir = Path(
        _env("MODELS_DIR") or str(_BACKEND_ROOT / "models")
    ).expanduser()
    collection = validate_collection_name(
        _env("KNOWLEDGE_COLLECTION", "wathiqa_knowledge")
    )
    return Settings(
        groq_api_key=_env("GROQ_API_KEY"),
        groq_model=_env("GROQ_MODEL", "openai/gpt-oss-120b"),
        llm_model=_env("LLM_MODEL"),
        llm_provider=_env("LLM_PROVIDER", "groq").lower(),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.0),
        ocr_engine=_env("OCR_ENGINE", "digital").lower(),
        ocr_max_pages=_env_int("OCR_MAX_PAGES", 120),
        embedding_model_id=_env("EMBEDDING_MODEL_ID", "BAAI/bge-m3"),
        embedding_model_path=_env("EMBEDDING_MODEL_PATH"),
        embedding_dimensions=_env_int("EMBEDDING_DIMENSIONS", 1024),
        chunk_strategy=_env("CHUNK_STRATEGY", "page_aware").lower(),
        chunk_size=_env_int("CHUNK_SIZE", 1200),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 200),
        vectorstore_backend=_env("VECTORSTORE_BACKEND", "chroma").lower(),
        vectorstore_persist_dir=_env(
            "VECTORSTORE_PERSIST_DIR",
            str(_BACKEND_ROOT / "data" / "chroma"),
        ),
        qdrant_url=_env("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=_env("QDRANT_API_KEY"),
        knowledge_collection=collection,
        llm_timeout_seconds=_env_int("LLM_TIMEOUT_SECONDS", 90),
        web_search_timeout_seconds=_env_int("WEB_SEARCH_TIMEOUT_SECONDS", 30),
        retrieval_top_k=_env_int("RETRIEVAL_TOP_K", 8),
        retrieval_mode=_env("RETRIEVAL_MODE", "hybrid").lower(),
        hybrid_fusion=_env("HYBRID_FUSION", "rrf").lower(),
        rrf_k=_env_int("RRF_K", 60),
        retrieval_candidate_multiplier=_env_int("RETRIEVAL_CANDIDATE_MULTIPLIER", 15),
        retrieval_min_score=_env_float("RETRIEVAL_MIN_SCORE", 0.0),
        enable_query_rewrite=_env_bool("ENABLE_QUERY_REWRITE", True),
        enable_reranker=_env_bool("ENABLE_RERANKER", True),
        reranker_backend=_env("RERANKER_BACKEND", "cross_encoder").lower(),
        reranker_model_id=_env(
            "RERANKER_MODEL_ID",
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        ),
        enable_spelling_corrector=_env_bool("ENABLE_SPELLING_CORRECTOR", True),
        spelling_min_confidence=_env_float("SPELLING_MIN_CONFIDENCE", 0.85),
        spelling_use_llm=_env_bool("SPELLING_USE_LLM", True),
        enable_query_booster=_env_bool("ENABLE_QUERY_BOOSTER", True),
        query_booster_use_llm=_env_bool("QUERY_BOOSTER_USE_LLM", True),
        query_booster_min_words=_env_int("QUERY_BOOSTER_MIN_WORDS", 6),
        max_document_bytes=_env_int("MAX_DOCUMENT_BYTES", 50 * 1024 * 1024),
        index_cache_size=_env_int("INDEX_CACHE_SIZE", 8),
        summary_max_chars=_env_int("SUMMARY_MAX_CHARS", 48_000),
        models_dir=models_dir,
        backend_root=_BACKEND_ROOT,
        repo_root=_REPO_ROOT,
        log_level=_env("LOG_LEVEL", "INFO").upper(),
    )

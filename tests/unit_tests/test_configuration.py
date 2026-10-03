"""Graph and settings smoke tests."""

from langgraph.pregel import Pregel

from agent.graph import graph
from core.config.settings import get_settings


def test_graph_is_compiled() -> None:
    assert isinstance(graph, Pregel)
    nodes = set(graph.nodes)
    assert "decision_agent" in nodes
    assert "query_documents" in nodes
    assert "run_calculator" not in nodes


def test_settings_defaults(monkeypatch) -> None:
    monkeypatch.delenv("OCR_ENGINE", raising=False)
    monkeypatch.delenv("MODELS_DIR", raising=False)
    monkeypatch.delenv("VECTORSTORE_PERSIST_DIR", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.embedding_model_id == "BAAI/bge-m3"
    assert settings.ocr_engine == "digital"
    assert settings.models_dir.name == "models"
    assert settings.backend_root.name == "src"
    assert "chroma" in settings.vectorstore_persist_dir
    assert settings.vectorstore_backend in {"chroma", "faiss", "qdrant"}
    assert settings.hybrid_fusion in {"rrf", "weighted"}
    assert settings.hybrid_fusion == "rrf" or settings.hybrid_fusion == "weighted"
    assert settings.enable_spelling_corrector is True
    assert settings.spelling_min_confidence >= 0.5
    assert settings.enable_query_booster is True
    assert settings.query_booster_min_words >= 3
    # Defaults favor RRF excellence profile unless env overrides.
    assert settings.retrieval_top_k >= 5
    assert "milvus" not in settings.vectorstore_backend
    nodes = set(graph.nodes)
    assert "ingest_document" not in nodes
    assert "query_planner" in nodes
    assert "query_knowledge_base" in nodes
    assert "web_search" in nodes
    assert "chroma" in settings.vectorstore_persist_dir or settings.vectorstore_persist_dir in {
        "",
        "memory",
        "none",
        "ephemeral",
    }